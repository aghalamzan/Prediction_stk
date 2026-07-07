"""News sentiment features for price forecasting (Phase 1).

Pipeline: fetch company news from Finnhub -> score each headline with Claude
(Haiku) -> bucket the scored headlines onto the price bars with a causal
exponential decay. The resulting per-bar features (sentiment, headline count,
event flag) are fed to ARIMAX and the linear model as exogenous regressors.

Design constraints:
  * **No look-ahead.** A feature at bar ``t`` is built only from headlines whose
    publish time is ``<= t``. The decay is one-directional in time, so a
    backtest fold never sees future news. ``assert_no_lookahead`` enforces it.
  * **Graceful degradation.** Missing API keys, a missing ``anthropic`` SDK, or a
    network error never break the pipeline — the affected step logs a warning
    and yields neutral (zero) features, so forecasting proceeds as before.
  * **Cached scoring.** Claude scores are cached on disk keyed by a headline
    hash, so repeated backtests over the same headlines don't re-pay the API.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCORER_MODEL = "claude-haiku-4-5"
_CACHE_DIR = Path(os.environ.get("PREDICTION_STK_NEWS_CACHE", ".news_cache"))


@dataclass
class Headline:
    """One news item with its publish time (tz-naive UTC) and Claude score."""

    published: pd.Timestamp
    text: str
    score: float = 0.0            # sentiment in [-1, 1]
    event_type: str = "none"      # e.g. guidance / contract / dilution / none
    confidence: float = 0.0       # scorer confidence in [0, 1]


# --------------------------------------------------------------------------- #
# 1. Fetch — Finnhub company-news
# --------------------------------------------------------------------------- #
class NewsFetcher:
    """Fetch company news from Finnhub. Requires ``FINNHUB_API_KEY``.

    Finnhub stamps a real Unix ``datetime`` on every article, which is what
    makes point-in-time backtesting possible. Without a key the fetch returns an
    empty list and logs a warning (the pipeline then produces neutral features).
    """

    BASE_URL = "https://finnhub.io/api/v1/company-news"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY")

    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> List[Headline]:
        if not self.api_key:
            logger.warning("FINNHUB_API_KEY not set — news features will be neutral.")
            return []
        try:
            import requests

            resp = requests.get(
                self.BASE_URL,
                params={
                    "symbol": symbol.upper(),
                    "from": pd.Timestamp(start).strftime("%Y-%m-%d"),
                    "to": pd.Timestamp(end).strftime("%Y-%m-%d"),
                    "token": self.api_key,
                },
                timeout=15,
            )
            resp.raise_for_status()
            items = resp.json()
        except Exception as exc:  # network, auth, rate limit, bad JSON
            logger.warning("Finnhub fetch failed for %s: %s", symbol, exc)
            return []

        out: List[Headline] = []
        for it in items:
            ts = it.get("datetime")
            headline = (it.get("headline") or "").strip()
            if not ts or not headline:
                continue
            published = pd.Timestamp(int(ts), unit="s")  # tz-naive UTC
            summary = (it.get("summary") or "").strip()
            text = f"{headline}. {summary}".strip() if summary else headline
            out.append(Headline(published=published, text=text))
        out.sort(key=lambda h: h.published)
        return out


# --------------------------------------------------------------------------- #
# 2. Score — Claude (Haiku) sentiment, cached on disk
# --------------------------------------------------------------------------- #
class ClaudeSentimentScorer:
    """Score headlines in the [-1, 1] range with Claude, caching by headline hash.

    Uses ``claude-haiku-4-5`` with structured outputs so the response is always a
    valid JSON array. Requires ``ANTHROPIC_API_KEY`` (or an ``ant auth`` profile)
    and the ``anthropic`` SDK; absent either, headlines fall back to score 0.0.
    """

    _SCHEMA = {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "sentiment": {"type": "number"},
                        "event_type": {
                            "type": "string",
                            "enum": ["guidance", "contract", "product", "legal",
                                     "dilution", "macro", "other", "none"],
                        },
                        "confidence": {"type": "number"},
                    },
                    "required": ["index", "sentiment", "event_type", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    }

    def __init__(self, model: str = SCORER_MODEL, batch_size: int = 40,
                 cache_dir: Path = _CACHE_DIR):
        self.model = model
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_path(self, text: str) -> Path:
        return self.cache_dir / f"{self._key(text)}.json"

    def _load_cached(self, h: Headline) -> bool:
        path = self._cache_path(h.text)
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            h.score = float(data["sentiment"])
            h.event_type = str(data.get("event_type", "none"))
            h.confidence = float(data.get("confidence", 0.0))
            return True
        except Exception:
            return False

    def _store_cached(self, h: Headline) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(h.text).write_text(json.dumps(
                {"sentiment": h.score, "event_type": h.event_type,
                 "confidence": h.confidence}
            ))
        except Exception as exc:
            logger.debug("Could not cache score: %s", exc)

    def score(self, headlines: Sequence[Headline]) -> List[Headline]:
        pending = [h for h in headlines if not self._load_cached(h)]
        if not pending:
            return list(headlines)

        client = self._make_client()
        if client is None:
            logger.warning("Claude scorer unavailable — headlines default to neutral (0.0).")
            return list(headlines)

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start:start + self.batch_size]
            self._score_batch(client, batch)
            for h in batch:
                self._store_cached(h)
        return list(headlines)

    @staticmethod
    def _make_client():
        try:
            import anthropic
        except ImportError:
            logger.warning("`anthropic` SDK not installed — run `pip install anthropic`.")
            return None
        try:
            return anthropic.Anthropic()  # resolves key or `ant auth` profile
        except Exception as exc:
            logger.warning("Could not construct Anthropic client: %s", exc)
            return None

    def _score_batch(self, client, batch: List[Headline]) -> None:
        numbered = "\n".join(f"{i}. {h.text}" for i, h in enumerate(batch))
        prompt = (
            "You are a financial-news sentiment analyst. For each numbered "
            "headline, return its short-term price sentiment for the mentioned "
            "stock as a number in [-1, 1] (negative = bearish, positive = "
            "bullish), an event_type, and your confidence in [0, 1].\n\n"
            f"Headlines:\n{numbered}"
        )
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=2048,
                output_config={"format": {"type": "json_schema", "schema": self._SCHEMA}},
                messages=[{"role": "user", "content": prompt}],
            )
            if resp.stop_reason == "refusal":
                logger.warning("Scorer refused a batch; leaving it neutral.")
                return
            text = next(b.text for b in resp.content if b.type == "text")
            scores = json.loads(text)["scores"]
        except Exception as exc:
            logger.warning("Claude scoring failed for a batch: %s", exc)
            return

        by_index = {int(s["index"]): s for s in scores}
        for i, h in enumerate(batch):
            s = by_index.get(i)
            if not s:
                continue
            h.score = max(-1.0, min(1.0, float(s.get("sentiment", 0.0))))
            h.event_type = str(s.get("event_type", "none"))
            h.confidence = max(0.0, min(1.0, float(s.get("confidence", 0.0))))


# --------------------------------------------------------------------------- #
# 3. Align — bucket scored headlines onto price bars with causal decay
# --------------------------------------------------------------------------- #
def build_news_features(
    headlines: Sequence[Headline],
    index: pd.DatetimeIndex,
    half_life_bars: float = 12.0,
) -> pd.DataFrame:
    """Return per-bar ``[sentiment, headline_count, event_flag]`` for ``index``.

    Each headline's influence at bar ``t`` decays as ``0.5 ** (age / half_life)``
    where ``age`` is the time from the headline to ``t`` and ``half_life`` is
    ``half_life_bars`` times the median bar spacing. Only headlines published at
    or before ``t`` contribute, so the features are strictly causal.
    """
    cols = ["sentiment", "headline_count", "event_flag"]
    feats = pd.DataFrame(0.0, index=index, columns=cols)
    if len(headlines) == 0 or len(index) < 2:
        return feats

    bar_delta = pd.Series(index).diff().median()
    half_life = bar_delta * half_life_bars
    if pd.isna(half_life) or half_life.total_seconds() <= 0:
        return feats

    times = np.array([h.published.value for h in headlines])  # ns since epoch
    scores = np.array([h.score for h in headlines], dtype=float)
    strong = np.array([abs(h.score) >= 0.5 and h.confidence >= 0.5 for h in headlines])
    hl_ns = float(half_life.value)

    # Normalise to nanoseconds: Timestamp.value is always ns, but a
    # microsecond-resolution index reports asi8 in microseconds.
    bar_ns = pd.DatetimeIndex(index).as_unit("ns").asi8
    for i, t in enumerate(bar_ns):
        past = times <= t
        if not past.any():
            continue
        age = (t - times[past]).astype(float)
        weight = np.power(0.5, age / hl_ns)
        feats.iloc[i, 0] = float(np.sum(scores[past] * weight))
        feats.iloc[i, 1] = float(np.sum(weight))
        feats.iloc[i, 2] = float(np.sum(weight[strong[past]]) > 0.5)
    return feats


def project_news_features(features: pd.DataFrame, horizon: int,
                          half_life_bars: float = 12.0) -> np.ndarray:
    """Decay the last known feature row forward over the forecast horizon.

    No future headlines are known at forecast time, so each feature decays from
    its last observed value toward zero at the same half-life — the honest
    "no new news" assumption. Returns an ``(horizon, n_features)`` array.
    """
    if features.empty:
        return np.zeros((horizon, 3))
    last = features.iloc[-1].to_numpy(dtype=float)
    steps = np.arange(1, horizon + 1)
    decay = np.power(0.5, steps / half_life_bars)
    return np.outer(decay, last)


def assert_no_lookahead(headlines: Sequence[Headline], index: pd.DatetimeIndex,
                        half_life_bars: float = 12.0) -> None:
    """Raise ``AssertionError`` if any bar's features depend on a future headline.

    Zeroes out every headline published after the last bar and checks the
    features are unchanged — a direct test that future news cannot leak in.
    """
    base = build_news_features(headlines, index, half_life_bars)
    cutoff = index[-1]
    past_only = [h for h in headlines if h.published <= cutoff]
    trimmed = build_news_features(past_only, index, half_life_bars)
    if not np.allclose(base.to_numpy(), trimmed.to_numpy()):
        raise AssertionError("News features leak future headlines into past bars.")


# --------------------------------------------------------------------------- #
# 4. Convenience — full pipeline for one symbol aligned to a price index
# --------------------------------------------------------------------------- #
@dataclass
class NewsConfig:
    enabled: bool = False
    half_life_bars: float = 12.0
    lookback_days: int = 30
    fetcher: NewsFetcher = field(default_factory=NewsFetcher)
    scorer: ClaudeSentimentScorer = field(default_factory=ClaudeSentimentScorer)


def news_exog_for_symbol(symbol: str, index: pd.DatetimeIndex,
                         config: NewsConfig) -> Optional[pd.DataFrame]:
    """Fetch, score, and align news features for ``symbol`` over ``index``.

    Returns ``None`` when news is disabled so callers can skip exogenous inputs
    entirely. On any failure it returns neutral (all-zero) features rather than
    raising, so forecasting never breaks on news problems.
    """
    if not config.enabled:
        return None
    end = pd.Timestamp(index[-1])
    start = end - pd.Timedelta(days=config.lookback_days)
    headlines = config.fetcher.fetch(symbol, start, end)
    headlines = config.scorer.score(headlines)
    return build_news_features(headlines, index, config.half_life_bars)
