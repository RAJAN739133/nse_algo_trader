"""
Streamlit Dashboard - Real-time trading monitoring and analytics.

Features:
- Live P&L tracking
- Position monitoring
- Risk metrics display
- Performance charts
- Alert management
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="NSE Algo Trader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_sample_data():
    """Load sample data for demonstration."""
    # Sample positions
    positions = pd.DataFrame({
        "Symbol": ["RELIANCE", "HDFCBANK", "TCS", "INFY", "ICICIBANK"],
        "Side": ["LONG", "LONG", "SHORT", "LONG", "SHORT"],
        "Qty": [50, 100, 25, 75, 40],
        "Entry Price": [2450.50, 1580.25, 3890.00, 1520.75, 985.30],
        "Current Price": [2485.75, 1565.40, 3855.20, 1545.90, 992.15],
        "P&L": [1762.50, -1485.00, 870.00, 1886.25, -274.00],
        "P&L %": [1.44, -0.94, 0.89, 1.65, -0.70]
    })
    
    # Sample trades history
    trades = pd.DataFrame({
        "Time": pd.date_range(start=datetime.now() - timedelta(hours=6), periods=10, freq="30min"),
        "Symbol": ["RELIANCE", "HDFCBANK", "TCS", "INFY", "ICICIBANK", 
                   "SBIN", "BHARTIARTL", "WIPRO", "AXISBANK", "KOTAKBANK"],
        "Side": ["BUY", "BUY", "SELL", "BUY", "SELL", "BUY", "BUY", "SELL", "BUY", "SELL"],
        "Qty": [50, 100, 25, 75, 40, 60, 30, 45, 80, 35],
        "Entry": [2450.50, 1580.25, 3890.00, 1520.75, 985.30, 625.40, 1180.50, 485.25, 1120.75, 1850.30],
        "Exit": [2485.75, 1565.40, 3855.20, 1545.90, 992.15, 638.60, 1195.25, 478.50, 1135.40, 1835.80],
        "P&L": [1762.50, -1485.00, 870.00, 1886.25, -274.00, 792.00, 442.50, 303.75, 1172.00, 507.50],
        "Strategy": ["ORB", "VWAP", "ORB", "ML", "VWAP", "ORB", "ML", "VWAP", "ORB", "ML"]
    })
    
    # Sample P&L curve
    pnl_curve = pd.DataFrame({
        "Time": pd.date_range(start=datetime.now() - timedelta(hours=6), periods=100, freq="3min"),
        "Cumulative P&L": [0] + list(pd.Series(range(99)).apply(
            lambda x: (x * 50 - 20 * (x % 5)) + (100 if x > 50 else 0)
        ))
    })
    
    return positions, trades, pnl_curve


def create_pnl_chart(pnl_curve: pd.DataFrame) -> go.Figure:
    """Create P&L curve chart."""
    fig = go.Figure()
    
    # Add P&L line
    fig.add_trace(go.Scatter(
        x=pnl_curve["Time"],
        y=pnl_curve["Cumulative P&L"],
        mode="lines",
        name="P&L",
        line=dict(color="green", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,255,0,0.1)"
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="Intraday P&L Curve",
        xaxis_title="Time",
        yaxis_title="P&L (₹)",
        template="plotly_dark",
        height=300
    )
    
    return fig


def create_strategy_pie(trades: pd.DataFrame) -> go.Figure:
    """Create strategy contribution pie chart."""
    strategy_pnl = trades.groupby("Strategy")["P&L"].sum().reset_index()
    
    fig = px.pie(
        strategy_pnl,
        values="P&L",
        names="Strategy",
        title="P&L by Strategy",
        template="plotly_dark",
        hole=0.4
    )
    
    fig.update_layout(height=300)
    return fig


def create_risk_gauge(var_pct: float, limit: float) -> go.Figure:
    """Create VaR gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=var_pct,
        title={"text": "VaR (95%)"},
        gauge={
            "axis": {"range": [0, limit * 1.5]},
            "bar": {"color": "orange"},
            "steps": [
                {"range": [0, limit * 0.5], "color": "green"},
                {"range": [limit * 0.5, limit], "color": "yellow"},
                {"range": [limit, limit * 1.5], "color": "red"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": limit
            }
        }
    ))
    
    fig.update_layout(height=250, template="plotly_dark")
    return fig


def main():
    # Header
    st.title("📈 NSE Algo Trader Dashboard")
    st.markdown("---")
    
    # Load data
    positions, trades, pnl_curve = load_sample_data()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Trading status
        trading_active = st.toggle("Trading Active", value=True)
        if trading_active:
            st.success("🟢 System Active")
        else:
            st.error("🔴 System Paused")
        
        st.markdown("---")
        
        # Filters
        st.subheader("Filters")
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now().date(), datetime.now().date())
        )
        
        strategies = st.multiselect(
            "Strategies",
            options=["ORB", "VWAP", "ML", "All"],
            default=["All"]
        )
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Today's Trades", len(trades))
        with col2:
            st.metric("Active Positions", len(positions))
    
    # Main content
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_pnl = positions["P&L"].sum()
    win_rate = len(trades[trades["P&L"] > 0]) / len(trades) * 100
    
    with col1:
        st.metric(
            "Day P&L",
            f"₹{total_pnl:,.2f}",
            delta=f"{total_pnl/100000*100:.2f}% of capital",
            delta_color="normal" if total_pnl >= 0 else "inverse"
        )
    
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col3:
        st.metric("Gross Exposure", "₹4.85L", delta="48.5%")
    
    with col4:
        st.metric("Net Exposure", "₹2.15L", delta="21.5%")
    
    with col5:
        st.metric("VaR (95%)", "₹1,850", delta="1.85%")
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(create_pnl_chart(pnl_curve), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_strategy_pie(trades), use_container_width=True)
    
    # Positions and Risk
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📊 Open Positions")
        
        # Style the dataframe
        def color_pnl(val):
            color = 'green' if val > 0 else 'red'
            return f'color: {color}'
        
        styled_positions = positions.style.applymap(
            color_pnl, subset=['P&L', 'P&L %']
        ).format({
            'Entry Price': '₹{:.2f}',
            'Current Price': '₹{:.2f}',
            'P&L': '₹{:,.2f}',
            'P&L %': '{:.2f}%'
        })
        
        st.dataframe(styled_positions, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Risk Metrics")
        
        # Risk gauges
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(create_risk_gauge(1.85, 3.0), use_container_width=True)
        with col_b:
            # Drawdown gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=2.3,
                title={"text": "Drawdown"},
                gauge={
                    "axis": {"range": [0, 10]},
                    "bar": {"color": "red"},
                    "steps": [
                        {"range": [0, 3], "color": "green"},
                        {"range": [3, 7], "color": "yellow"},
                        {"range": [7, 10], "color": "red"}
                    ]
                }
            ))
            fig.update_layout(height=250, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk limits status
        st.markdown("**Limit Status**")
        limits = {
            "Daily Loss": ("✅", "₹2,150 / ₹3,000"),
            "Weekly Loss": ("✅", "₹3,800 / ₹7,000"),
            "Max Position": ("✅", "8.5% / 10%"),
            "Sector Exposure": ("⚠️", "28% / 30%")
        }
        
        for limit_name, (status, value) in limits.items():
            st.text(f"{status} {limit_name}: {value}")
    
    st.markdown("---")
    
    # Trade History
    st.subheader("📜 Trade History")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    with col1:
        side_filter = st.selectbox("Side", ["All", "BUY", "SELL"])
    with col2:
        strategy_filter = st.selectbox("Strategy", ["All"] + list(trades["Strategy"].unique()))
    with col3:
        pnl_filter = st.selectbox("P&L", ["All", "Profitable", "Losing"])
    
    # Filter trades
    filtered_trades = trades.copy()
    if side_filter != "All":
        filtered_trades = filtered_trades[filtered_trades["Side"] == side_filter]
    if strategy_filter != "All":
        filtered_trades = filtered_trades[filtered_trades["Strategy"] == strategy_filter]
    if pnl_filter == "Profitable":
        filtered_trades = filtered_trades[filtered_trades["P&L"] > 0]
    elif pnl_filter == "Losing":
        filtered_trades = filtered_trades[filtered_trades["P&L"] <= 0]
    
    st.dataframe(
        filtered_trades.style.format({
            'Entry': '₹{:.2f}',
            'Exit': '₹{:.2f}',
            'P&L': '₹{:,.2f}'
        }),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Alerts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔔 Recent Alerts")
        
        alerts = [
            {"time": "10:45:32", "type": "INFO", "message": "Trade entered: RELIANCE BUY 50 @ ₹2,450.50"},
            {"time": "10:52:18", "type": "WARNING", "message": "Sector exposure approaching limit: Banking at 28%"},
            {"time": "11:15:45", "type": "INFO", "message": "SL hit: HDFCBANK SELL 100 @ ₹1,565.40"},
            {"time": "11:30:00", "type": "INFO", "message": "Target hit: TCS BUY 25 @ ₹3,855.20"},
        ]
        
        for alert in alerts:
            if alert["type"] == "WARNING":
                st.warning(f"⚠️ [{alert['time']}] {alert['message']}")
            elif alert["type"] == "ERROR":
                st.error(f"🔴 [{alert['time']}] {alert['message']}")
            else:
                st.info(f"ℹ️ [{alert['time']}] {alert['message']}")
    
    with col2:
        st.subheader("📊 Strategy Performance")
        
        strategy_stats = trades.groupby("Strategy").agg({
            "P&L": ["sum", "count", lambda x: (x > 0).mean() * 100]
        }).round(2)
        strategy_stats.columns = ["Total P&L", "Trades", "Win Rate %"]
        
        st.dataframe(
            strategy_stats.style.format({
                'Total P&L': '₹{:,.2f}',
                'Win Rate %': '{:.1f}%'
            }),
            use_container_width=True
        )
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh: 5 seconds")


if __name__ == "__main__":
    main()
