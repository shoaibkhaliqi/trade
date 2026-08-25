"""Run forward tracking for ALL active assets on a loop.

Leave this running in a terminal. It checks every hour (on bar close)
and generates signals for SOL + ETH + BTC automatically.

Usage:
    .venv\\Scripts\\python.exe scripts\\forward_runner.py
    .venv\\Scripts\\python.exe scripts\\forward_runner.py --interval 3600
"""
import argparse
import sys
import time
from datetime import UTC, datetime

sys.path.insert(0, "scripts")
from forward_tracker import generate_and_log, report  # noqa: E402

ASSETS = ["SOLUSDT", "ETHUSDT", "BTCUSDT", "BNBUSDT"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=3600,
                        help="seconds between checks (default 3600 = 1h)")
    parser.add_argument("--assets", nargs="+", default=ASSETS)
    args = parser.parse_args()

    print("=" * 60)
    print("  DARWIN LAB — FORWARD RUNNER")
    print(f"  assets: {', '.join(args.assets)}")
    print(f"  interval: {args.interval}s")
    print(f"  started: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)
    print("\nLeave this terminal running. Ctrl+C to stop.\n")

    iteration = 0
    while True:
        iteration += 1
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n--- iteration {iteration} | {now} ---")
        for symbol in args.assets:
            try:
                generate_and_log(symbol)
            except Exception as e:
                print(f"  {symbol}: ERROR {e}")

        if iteration % 24 == 0:
            # daily summary every 24 checks
            print("\n--- daily summary ---")
            for symbol in args.assets:
                report(symbol)

        print(f"\nnext check in {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
