"""Combined multi-indicator strategy: SMC + T3 + Squeeze + UT Bot + Ichimoku.

Each indicator votes LONG/SHORT/HOLD independently. The combined strategy:
- CONSERVATIVE: all 5 must agree (very rare, very selective)
- MAJORITY: 3+ must agree
- TREND+TIMING: T3 gives regime, SMC or Squeeze gives entry, Ichimoku confirms

Also runs each indicator solo for comparison.
"""
# ruff: noqa: E702
from darwin.agents.strategies import (
    SmartMoneyConceptsStrategy,
    T3SqueezeMomentumStrategy,
    T3Strategy,
    UTBotStrategy,
)
from darwin.config import load_config
from darwin.data.storage import DataStorage
from darwin.environment.simulator import Action, SimulatorConfig, TradingSimulator
from darwin.evaluation.metrics import MetricsReport, format_row
from darwin.evolution.baseline import directional_baseline

cfg = load_config("development")
sim_cfg = SimulatorConfig(
    initial_capital=1000.0, taker_fee_pct=cfg["simulator"]["taker_fee_pct"],
    slippage_pct=cfg["simulator"]["slippage_pct"],
    position_size_pct=cfg["simulator"]["fixed_position_size_pct"])
src = DataStorage("data/processed")


def combine_signals(signal_lists, mode="majority"):
    """Combine multiple action lists into one."""
    n = len(signal_lists[0])
    combined = [Action.HOLD] * n
    for t in range(n):
        votes = []
        for lst in signal_lists:
            if lst[t] == Action.LONG:
                votes.append(1)
            elif lst[t] == Action.SHORT:
                votes.append(-1)
            elif lst[t] == Action.CLOSE:
                votes.append(0.5)  # exit signal
            else:
                votes.append(0)

        long_votes = sum(1 for v in votes if v == 1)
        short_votes = sum(1 for v in votes if v == -1)
        exit_votes = sum(1 for v in votes if v == 0.5)
        total = len(votes)

        if mode == "conservative":
            need = total  # ALL must agree
        elif mode == "majority":
            need = (total // 2) + 1  # simple majority
        elif mode == "strong":
            need = total - 1  # all but one
        else:
            need = (total // 2) + 1

        if long_votes >= need:
            combined[t] = Action.LONG
        elif short_votes >= need:
            combined[t] = Action.SHORT
        elif exit_votes >= 2:
            combined[t] = Action.CLOSE
    return combined


for symbol in ("SOLUSDT", "BTCUSDT", "ETHUSDT"):
    ohlcv = src.load(symbol, "1h")
    print(f"\n{'='*60}")
    print(f"  {symbol} 1h — Multi-Indicator Combination")
    print(f"{'='*60}")

    # generate all signals
    smc = SmartMoneyConceptsStrategy(swing_lookback=5)
    t3 = T3Strategy(mode="slope", period=14)
    squeeze = T3SqueezeMomentumStrategy(t3_period=14)
    utbot = UTBotStrategy(atr_period=10, atr_multiplier=2.0)

    sig_smc = smc.generate_actions(ohlcv)
    sig_t3 = t3.generate_actions(ohlcv)
    sig_squeeze = squeeze.generate_actions(ohlcv)
    sig_utbot = utbot.generate_actions(ohlcv)

    # solo results
    print("\n  solo:")
    solo_results = {}
    for name, sig in [("smc", sig_smc), ("t3_slope", sig_t3),
                      ("squeeze", sig_squeeze), ("ut_bot", sig_utbot)]:
        res = TradingSimulator(sim_cfg).run(ohlcv, sig)
        rep = MetricsReport.from_result(res, "1h")
        solo_results[name] = rep
        print(format_row("  " + name, rep) +
              "  t=" + str(rep.n_trades) + " WR=" + format(rep.win_rate, ".0%"))

    # combinations
    print("\n  combined:")
    combos = {
        "majority(3/4)": combine_signals(
            [sig_smc, sig_t3, sig_squeeze, sig_utbot], "majority"),
        "conservative(4/4)": combine_signals(
            [sig_smc, sig_t3, sig_squeeze, sig_utbot], "conservative"),
        "strong(3/4)": combine_signals(
            [sig_smc, sig_t3, sig_squeeze, sig_utbot], "strong"),
        "smc+t3 (regime+structure)": combine_signals(
            [sig_smc, sig_t3, sig_t3, sig_squeeze], "majority"),
    }

    combo_results = {}
    for name, sig in combos.items():
        res = TradingSimulator(sim_cfg).run(ohlcv, sig)
        rep = MetricsReport.from_result(res, "1h")
        combo_results[name] = rep
        n = len(ohlcv)
        val_end = int(n * 0.85)
        tail = res.equity_curve.iloc[val_end + 5000:].reset_index(drop=True)
        tail_rep = MetricsReport.from_parts(tail, (), "1h")
        marker = " <<<" if rep.total_return > 0 else ""
        print(format_row("  " + name, rep) +
              "  t=" + str(rep.n_trades)
              + " WR=" + format(rep.win_rate, ".0%")
              + " PF=" + format(rep.profit_factor, ".2f")
              + " tail=" + format(tail_rep.total_return, "+.1%")
              + marker)

    # best combo
    best_name = max(combo_results, key=lambda k: combo_results[k].total_return)
    best_rep = combo_results[best_name]
    print(f"\n  best combo: {best_name} {best_rep.total_return:+.2%}")
    baseline, side = directional_baseline(ohlcv, sim_cfg)
    print(f"  passive baseline: {baseline:+.2%} ({side})")
