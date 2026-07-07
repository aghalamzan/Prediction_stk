import argparse

from .core import StockPipeline
from .daily import DailyAdvisor, format_report
from .horizon import PROFILES
from .visualize import format_summary, run_forecast


def _run_pipeline(args):
    pipeline = StockPipeline()
    print(pipeline.run())


def _run_forecast(args):
    config_symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    results, chart_path = run_forecast(
        symbols=config_symbols,
        profile=args.profile,
        output_dir=None if args.no_chart else args.output_dir,
        steps=args.horizon,
        prefer_gpu=not args.cpu,
        use_transformer=not args.no_transformer,
    )
    print(format_summary(results))
    if chart_path is not None:
        print(f"\nChart saved to: {chart_path.resolve()}")


def _run_daily(args):
    config = {"forecast_length": args.horizon, "posture": args.posture}
    if args.cpu:
        config["prefer_gpu"] = False
    if args.no_transformer:
        config["use_transformer"] = False
    advisor = DailyAdvisor(config)
    result = advisor.run(
        symbol=args.symbol,
        shares=args.shares,
        unrealized_pl=args.unrealized_pl,
        cost_basis=args.cost_basis,
        posture=args.posture,
    )
    print(format_report(result))


def main():
    parser = argparse.ArgumentParser(description="Stock prediction pipeline")
    parser.add_argument("--config", help="Path to config file", default=None)
    sub = parser.add_subparsers(dest="command")

    for name, default_profile in (("forecast", "week"), ("visualize", "intraday")):
        p = sub.add_parser(
            name,
            help=f"Run models and plot forecasts (default profile: {default_profile})",
        )
        p.add_argument("--symbols", default=None,
                       help="Comma-separated tickers (default: mega-cap + sector set)")
        p.add_argument("--profile", default=default_profile, choices=sorted(PROFILES),
                       help="Horizon profile: intraday (~5h), week (5 working days), month (~21)")
        p.add_argument("--output-dir", default="outputs", dest="output_dir")
        p.add_argument("--horizon", type=int, default=None,
                       help="Override the profile's forecast length (bars)")
        p.add_argument("--no-chart", action="store_true", dest="no_chart",
                       help="Skip the PNG chart, print the summary only")
        p.add_argument("--cpu", action="store_true", help="Force CPU instead of GPU")
        p.add_argument("--no-transformer", action="store_true", dest="no_transformer")
        p.set_defaults(func=_run_forecast)

    daily = sub.add_parser("daily", help="Daily position-aware recommendation for one symbol")
    daily.add_argument("--symbol", required=True, help="Ticker, e.g. SERV")
    daily.add_argument("--shares", type=float, required=True)
    daily.add_argument("--unrealized-pl", type=float, default=None,
                       dest="unrealized_pl",
                       help="Current unrealized P&L in USD (loss is negative, e.g. -3500)")
    daily.add_argument("--cost-basis", type=float, default=None, dest="cost_basis",
                       help="Average cost per share (alternative to --unrealized-pl)")
    daily.add_argument("--horizon", type=int, default=20, help="Forecast length in 15-min bars")
    daily.add_argument("--posture", default="balanced",
                       choices=["preservation", "balanced", "accumulate"])
    daily.add_argument("--cpu", action="store_true", help="Force CPU instead of GPU")
    daily.add_argument("--no-transformer", action="store_true", dest="no_transformer")
    daily.set_defaults(func=_run_daily)

    args = parser.parse_args()
    if getattr(args, "func", None):
        args.func(args)
    else:
        _run_pipeline(args)


if __name__ == "__main__":
    main()
