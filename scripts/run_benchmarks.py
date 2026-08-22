"""Run every benchmark strategy through the simulator on stored market data.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_benchmarks.py --timeframe 15m
"""

from __future__ import annotations

import argparse

from darwin.agents import default_benchmarks
from darwin.config import load_config
from darwin.data.storage import DataStorage
from darwin.environment.simulator import SimulatorConfig, TradingSimulator
from darwin.evaluation.metrics import MetricsReport, format_header, format_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="development")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bars", type=int, default=None,
                        help="use only the most recent N candles")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    symbol = data_cfg["symbol"]

    ohlcv = DataStorage(data_cfg["processed_dir"]).load(symbol, args.timeframe)
    if args.bars is not None:
        ohlcv = ohlcv.tail(args.bars).reset_index(drop=True)

    sim_cfg = SimulatorConfig(
        initial_capital=cfg["simulator"]["initial_capital"],
        taker_fee_pct=cfg["simulator"]["taker_fee_pct"],
        slippage_pct=cfg["simulator"]["slippage_pct"],
        position_size_pct=cfg["simulator"]["fixed_position_size_pct"],
    )
    sim = TradingSimulator(sim_cfg)

    print(
        f"\n{symbol} {args.timeframe} | {len(ohlcv)} candles | "
        f"fee={sim_cfg.taker_fee_pct}% slip={sim_cfg.slippage_pct}% "
        f"size={sim_cfg.position_size_pct}% | capital={sim_cfg.initial_capital}"
    )
    print(format_header())

    for strategy in default_benchmarks(seed=args.seed):
        actions = strategy.generate_actions(ohlcv)
        result = sim.run(ohlcv, actions)
        report = MetricsReport.from_result(result, args.timeframe)
        print(format_row(strategy.name, report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
