"""Darwin Trading Lab — Dashboard (M19).

Usage:
    .venv\\Scripts\\python.exe -m streamlit run dashboard/app.py
"""
import sqlite3
from datetime import UTC, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Darwin Trading Lab", page_icon="🧬",
                   layout="wide")

DB_FORWARD = "experiments/forward_tracking.sqlite"
DB_META = "experiments/metadata.sqlite"


# ====================================================================
# data access
# ====================================================================
def get_forward_signals(symbol=None):
    conn = sqlite3.connect(DB_FORWARD)
    query = "SELECT * FROM signals"
    params = ()
    if symbol:
        query += " WHERE symbol = ?"
        params = (symbol,)
    query += " ORDER BY bar_time"
    try:
        df = pd.read_sql(query, conn, params=params)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def get_forward_trades(symbol=None, status=None):
    conn = sqlite3.connect(DB_FORWARD)
    query = "SELECT * FROM trades WHERE 1=1"
    params = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY entry_time"
    try:
        df = pd.read_sql(query, conn, params=params)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def get_experiments():
    conn = sqlite3.connect(DB_META)
    try:
        rows = conn.execute(
            "SELECT id, created_at, kind, payload FROM experiments "
            "ORDER BY created_at DESC LIMIT 50").fetchall()
    except Exception:
        rows = []
    conn.close()
    return rows


# ====================================================================
# sidebar
# ====================================================================
st.sidebar.title("🧬 Darwin Lab")
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "📡 Forward Tracker",
    "📈 Backtest Results",
    "🔬 Research",
])

symbol_filter = st.sidebar.selectbox(
    "Asset", ["All", "SOLUSDT", "ETHUSDT", "BTCUSDT",
              "XRPUSDT", "AVAXUSDT", "LINKUSDT", "DOGEUSDT"],
    index=0)

# ====================================================================
# PAGE: OVERVIEW
# ====================================================================
if page == "📊 Overview":
    st.title("📊 Darwin Trading Lab — Overview")

    signals = get_forward_signals(symbol_filter if symbol_filter != "All" else None)
    trades = get_forward_trades(symbol_filter if symbol_filter != "All" else None,
                                status="closed")
    open_trades = get_forward_trades(symbol_filter if symbol_filter != "All" else None,
                                     status="open")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Signals Logged", len(signals))
    with col2:
        st.metric("Closed Trades", len(trades))
    with col3:
        st.metric("Open Positions", len(open_trades))
    with col4:
        if not trades.empty and "pnl_pct" in trades.columns:
            pnls = trades["pnl_pct"].dropna()
            if len(pnls) > 0:
                st.metric("Forward P&L", f"{pnls.sum():+.2%}")
            else:
                st.metric("Forward P&L", "—")
        else:
            st.metric("Forward P&L", "—")

    # forward equity curve
    if not trades.empty and "pnl_pct" in trades.columns and len(trades) > 0:
        pnls = trades["pnl_pct"].dropna().tolist()
        if pnls:
            equity = [1000.0]
            for p in pnls:
                equity.append(equity[-1] * (1 + p))
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=equity, mode="lines+markers",
                                     name="Forward Equity",
                                     line=dict(color="#00cc96")))
            fig.update_layout(title="Forward Test Equity Curve",
                              yaxis_title="Equity ($)", xaxis_title="Trade #")
            st.plotly_chart(fig, use_container_width=True)

    # trade table
    if not trades.empty and len(trades) > 0:
        st.subheader("Closed Trades")
        display_cols = ["symbol", "direction", "entry_time", "entry_price",
                        "exit_time", "exit_price", "pnl_pct", "status"]
        available = [c for c in display_cols if c in trades.columns]
        st.dataframe(trades[available], use_container_width=True)

    if not open_trades.empty and len(open_trades) > 0:
        st.subheader("Open Positions")
        st.dataframe(open_trades, use_container_width=True)

    if signals.empty or len(signals) == 0:
        st.info("No forward signals yet. Run the forward tracker to start collecting data.")
        st.code(".venv\\Scripts\\python.exe scripts\\forward_tracker.py --symbol SOLUSDT")

# ====================================================================
# PAGE: FORWARD TRACKER
# ====================================================================
elif page == "📡 Forward Tracker":
    st.title("📡 Forward Tracker")

    st.subheader("Generate Signal")
    col1, col2 = st.columns(2)
    with col1:
        signal_symbol = st.selectbox(
            "Symbol", ["SOLUSDT", "ETHUSDT", "BTCUSDT", "XRPUSDT",
                       "AVAXUSDT", "LINKUSDT", "DOGEUSDT"])
    with col2:
        if st.button("📡 Generate Signal", type="primary"):
            with st.spinner("Fetching data + training meta-model..."):
                import subprocess
                result = subprocess.run(
                    [r".venv\Scripts\python.exe", "scripts/forward_tracker.py",
                     "--symbol", signal_symbol],
                    capture_output=True, text=True, timeout=120)
                st.code(result.stdout if result.stdout else result.stderr)
                st.rerun()

    st.subheader("Signal History")
    signals = get_forward_signals(symbol_filter if symbol_filter != "All" else None)
    if not signals.empty and len(signals) > 0:
        # P(win) over time
        if "p_win" in signals.columns:
            fig = go.Figure()
            for sym in signals["symbol"].unique():
                sub = signals[signals["symbol"] == sym]
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(sub["bar_time"]),
                    y=sub["p_win"], mode="lines+markers",
                    name=sym, line=dict(width=1.5)))
            fig.add_hline(y=0.55, line_dash="dash", line_color="red",
                          annotation_text="threshold")
            fig.update_layout(title="P(win) Over Time", yaxis_title="P(win)")
            st.plotly_chart(fig, use_container_width=True)

        display_cols = ["bar_time", "symbol", "close",
                        "primary_signal", "p_win", "final_signal"]
        available = [c for c in display_cols if c in signals.columns]
        st.dataframe(signals[available].sort_values("bar_time", ascending=False),
                     use_container_width=True)
    else:
        st.info("No signals logged yet.")

# ====================================================================
# PAGE: BACKTEST RESULTS
# ====================================================================
elif page == "📈 Backtest Results":
    st.title("📈 Backtest Results")

    # hardcoded from our EH-v8 run — the most comprehensive comparison
    st.subheader("Ichimoku + Meta-Filter: Bybit Top 10 (EH-v8)")
    data = {
        "Asset": ["SOL", "ETH", "SUI", "XRP", "AAVE", "BTC", "SAND",
                  "DOGE", "AVAX", "ZEC", "LINK", "HYPE", "PAXG"],
        "Unfiltered": ["+115.2%", "-9.8%", "-4.3%", "-14.8%", "-42.4%",
                       "+11.0%", "+60.3%", "+12.6%", "+47.2%", "+29.5%",
                       "-16.4%", "-12.3%", "-16.8%"],
        "Filtered": ["+129.6%", "+53.3%", "+46.7%", "+38.0%", "+24.8%",
                     "+22.0%", "+18.0%", "+17.6%", "+20.0%", "-90.4%",
                     "-1.6%", "-2.5%", "-3.4%"],
        "DD": ["-10.7%", "-15.1%", "-21.0%", "-28.8%", "-19.1%",
               "-14.6%", "-20.5%", "-38.0%", "-32.0%", "-99.4%",
               "-31.9%", "-7.7%", "-9.4%"],
        "PF": ["2.17", "2.14", "1.89", "1.54", "1.27",
               "1.24", "1.21", "1.16", "1.12", "0.31",
               "0.99", "0.62", "0.74"],
        "Sharpe": ["1.248", "0.876", "0.822", "0.531", "0.355",
                   "0.403", "0.322", "0.254", "0.306", "0.794",
                   "0.068", "-0.211", "-0.137"],
        "Trades": ["132", "70", "40", "106", "97",
                   "177", "146", "97", "286", "100",
                   "206", "4", "68"],
        "Tail Pass": ["❌", "❌", "n/a", "❌", "❌",
                      "❌", "❌", "❌", "❌", "❌",
                      "❌", "n/a", "❌"],
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # highlight positive
    st.info("🟢 SOL and ETH are the strongest candidates. "
            "🔴 ZEC shows the filter can backfire. "
            "PAXG is confirmed untradeable with technical analysis.")

    # stress test results
    st.subheader("Stress Test Results")
    stress_data = {
        "Scenario": ["Baseline", "2x Fees", "3x Fees", "2x Slippage",
                     "5x Slippage", "3x Fees + 5x Slip"],
        "SOL Return": ["+129.6%", "+121.4%", "+113.5%", "+126.6%",
                       "+117.7%", "+102.5%"],
        "SOL Sharpe": ["1.248", "1.196", "1.145", "1.229",
                       "1.173", "1.069"],
        "ETH Return": ["+53.3%", "+50.4%", "+47.5%", "+52.2%",
                       "+49.0%", "+43.4%"],
        "ETH Sharpe": ["0.876", "0.839", "0.801", "0.862",
                       "0.821", "0.746"],
    }
    st.dataframe(pd.DataFrame(stress_data), use_container_width=True,
                 hide_index=True)

    # Monte Carlo
    st.subheader("Monte Carlo (5000 bootstraps)")
    mc_col1, mc_col2 = st.columns(2)
    with mc_col1:
        st.metric("SOL P(loss)", "0.2%")
        st.metric("SOL 5th percentile", "+$552")
    with mc_col2:
        st.metric("ETH P(loss)", "3.3%")
        st.metric("ETH 5th percentile", "+$53")

# ====================================================================
# PAGE: RESEARCH
# ====================================================================
elif page == "🔬 Research":
    st.title("🔬 Research")

    st.subheader("Experiment History")
    experiments = get_experiments()
    if experiments:
        exp_data = []
        for exp_id, created, kind, payload_json in experiments:
            import json
            try:
                p = json.loads(payload_json)
                exp_data.append({
                    "id": exp_id[:12], "created": created[:19],
                    "kind": kind,
                    "symbol": p.get("symbol", ""),
                    "timeframe": p.get("timeframe", ""),
                    "seed": p.get("seed", ""),
                })
            except Exception:
                pass
        if exp_data:
            st.dataframe(pd.DataFrame(exp_data), use_container_width=True,
                         hide_index=True)
    else:
        st.info("No experiments recorded.")

    st.subheader("Quick Commands")
    st.code("""# Generate signal
.venv\\Scripts\\python.exe scripts\\forward_tracker.py --symbol SOLUSDT

# Run supervised hunt
.venv\\Scripts\\python.exe scripts\\supervised_hunt.py --model lightgbm

# Run meta-labeling hunt
.venv\\Scripts\\python.exe scripts\\meta_hunt.py

# Run population
.venv\\Scripts\\python.exe scripts\\run_population.py --size 8

# Walk-forward benchmarks
.venv\\Scripts\\python.exe scripts\\walk_forward.py --timeframe 15m

# Fitness report
.venv\\Scripts\\python.exe scripts\\fitness_report.py

# Diversity report
.venv\\Scripts\\python.exe scripts\\diversity_report.py

# Lineage tree
.venv\\Scripts\\python.exe scripts\\lineage.py --hof

# Regime report
.venv\\Scripts\\python.exe scripts\\regime_report.py --top 3

# Parameter optimization
.venv\\Scripts\\python.exe experiments\\param_opt.py

# Indicator sweep
.venv\\Scripts\\python.exe experiments\\indicator_sweep.py
""")

# sidebar footer
st.sidebar.markdown("---")
st.sidebar.caption(f"Darwin Lab v0.1.0 | {datetime.now(UTC).strftime('%Y-%m-%d %H:%M')} UTC")
