import argparse

from .core import StockPipeline
from .daily import DailyAdvisor, format_report
from .visualize import format_summary, run_visualization


def _run_pipeline(args):
    pipeline = StockPipeline()
    print(pipeline.run())


def _run_visualize(args):
    config_symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    results, chart_path = run_visualization(
        symbols=config_symbols,
        output_dir=args.output_dir,
        forecast_length=args.horizon,
        prefer_gpu=not args.cpu,
        use_transformer=not args.no_transformer,
    )
    print(format_summary(results))
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

    viz = sub.add_parser("visualize", help="Run models and plot intraday forecasts for today")
    viz.add_argument("--symbols", default=None,
                     help="Comma-separated tickers (default: mega-cap + sector set)")
    viz.add_argument("--output-dir", default="outputs", dest="output_dir")
    viz.add_argument("--horizon", type=int, default=20, help="Forecast length in 15-min bars")
    viz.add_argument("--cpu", action="store_true", help="Force CPU instead of GPU")
    viz.add_argument("--no-transformer", action="store_true", dest="no_transformer")
    viz.set_defaults(func=_run_visualize)

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
