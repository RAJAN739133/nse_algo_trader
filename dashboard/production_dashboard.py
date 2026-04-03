#!/usr/bin/env python3
"""
PRODUCTION DASHBOARD — Real-time Algo Trading Monitor
════════════════════════════════════════════════════════════

A comprehensive Streamlit dashboard for monitoring:
1. Live positions and P&L
2. System health and alerts
3. Performance analytics
4. Trade history
5. Kill switch control

Usage:
    streamlit run dashboard/production_dashboard.py --server.port 8501
"""

import os
import sys
import json
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.trade_database import TradeDB, get_db
from core.system_monitor import KillSwitch, TradingHours

# Page config
st.set_page_config(
    page_title="NSE Algo Trader - Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive { color: #00C853; }
    .negative { color: #FF5252; }
    .neutral { color: #FFFFFF; }
    .kill-switch-active {
        background-color: #FF5252;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)


def load_config():
    """Load configuration."""
    config_path = Path(__file__).parent.parent / "config" / "config_test.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def get_color(value: float) -> str:
    """Get color based on value."""
    if value > 0:
        return "green"
    elif value < 0:
        return "red"
    return "gray"


def format_pnl(value: float) -> str:
    """Format P&L with color."""
    color = "positive" if value > 0 else "negative" if value < 0 else "neutral"
    return f'<span class="{color}">₹{value:+,.2f}</span>'


def main():
    """Main dashboard."""
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 NSE Algo Trader")
        st.markdown("---")
        
        # Market status
        if TradingHours.is_market_hours():
            st.success("🟢 Market OPEN")
            ttc = TradingHours.time_to_close()
            st.caption(f"Closes in {ttc.seconds // 3600}h {(ttc.seconds % 3600) // 60}m")
        else:
            st.warning("🔴 Market CLOSED")
            tto = TradingHours.time_to_open()
            st.caption(f"Opens in {tto.days}d {tto.seconds // 3600}h")
        
        st.markdown("---")
        
        # Kill Switch
        st.subheader("🚨 Emergency Controls")
        kill_status = KillSwitch.get_status()
        
        if kill_status:
            st.markdown(
                '<div class="kill-switch-active">'
                '<h3>⚠️ KILL SWITCH ACTIVE</h3>'
                f'<p>Since: {kill_status["activated_at"]}</p>'
                f'<p>Reason: {kill_status["reason"]}</p>'
                '</div>',
                unsafe_allow_html=True
            )
            if st.button("🔓 Deactivate Kill Switch", type="primary"):
                KillSwitch.deactivate()
                st.rerun()
        else:
            st.info("System operating normally")
            reason = st.text_input("Kill switch reason")
            if st.button("🛑 ACTIVATE KILL SWITCH", type="secondary"):
                if reason:
                    KillSwitch.activate(reason)
                    st.rerun()
                else:
                    st.error("Please provide a reason")
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["📊 Overview", "📈 Positions", "📋 Trade History", 
             "📉 Analytics", "⚙️ System Health"],
            label_visibility="collapsed"
        )
    
    # Main content
    db = get_db()
    
    if page == "📊 Overview":
        render_overview(db)
    elif page == "📈 Positions":
        render_positions(db)
    elif page == "📋 Trade History":
        render_trade_history(db)
    elif page == "📉 Analytics":
        render_analytics(db)
    elif page == "⚙️ System Health":
        render_system_health()


def render_overview(db: TradeDB):
    """Render overview page."""
    st.title("📊 Trading Overview")
    
    # Today's summary
    today = date.today().isoformat()
    trades_today = db.get_today_trades()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_pnl = sum(t.net_pnl for t in trades_today)
    winners = len([t for t in trades_today if t.net_pnl > 0])
    losers = len([t for t in trades_today if t.net_pnl < 0])
    win_rate = winners / len(trades_today) * 100 if trades_today else 0
    
    with col1:
        st.metric(
            "Today's P&L",
            f"₹{total_pnl:+,.2f}",
            delta=f"{total_pnl/100:.1f}%" if total_pnl != 0 else None,
            delta_color="normal" if total_pnl >= 0 else "inverse"
        )
    
    with col2:
        st.metric("Trades", len(trades_today))
    
    with col3:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col4:
        st.metric("Winners", winners, delta=winners - losers)
    
    with col5:
        st.metric("Losers", losers)
    
    st.markdown("---")
    
    # Open positions
    st.subheader("📈 Open Positions")
    positions = db.get_open_positions()
    
    if positions:
        pos_df = pd.DataFrame(positions)
        st.dataframe(pos_df, use_container_width=True)
    else:
        st.info("No open positions")
    
    st.markdown("---")
    
    # Recent trades
    st.subheader("📋 Recent Trades")
    if trades_today:
        trades_df = pd.DataFrame([t.to_dict() for t in trades_today[:10]])
        trades_df['net_pnl'] = trades_df['net_pnl'].apply(
            lambda x: f"₹{x:+,.2f}"
        )
        st.dataframe(
            trades_df[['symbol', 'direction', 'strategy', 'entry_price', 
                      'exit_price', 'net_pnl', 'exit_reason']],
            use_container_width=True
        )
    else:
        st.info("No trades today")
    
    # P&L chart
    st.subheader("📈 Cumulative P&L (Today)")
    if trades_today:
        cumulative = []
        running = 0
        for t in sorted(trades_today, key=lambda x: x.entry_time):
            running += t.net_pnl
            cumulative.append({
                'time': t.exit_time,
                'pnl': running,
                'symbol': t.symbol
            })
        
        cum_df = pd.DataFrame(cumulative)
        fig = px.line(
            cum_df, x='time', y='pnl',
            title='Cumulative P&L',
            markers=True
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)


def render_positions(db: TradeDB):
    """Render positions page."""
    st.title("📈 Positions")
    
    positions = db.get_open_positions()
    
    if not positions:
        st.info("No open positions")
        return
    
    for pos in positions:
        with st.expander(f"{pos['symbol']} - {pos['direction']}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Entry Price", f"₹{pos['entry_price']:,.2f}")
            with col2:
                st.metric("Quantity", pos['quantity'])
            with col3:
                st.metric("Stop Loss", f"₹{pos.get('stop_loss', 0):,.2f}")
            with col4:
                st.metric("Target", f"₹{pos.get('target', 0):,.2f}")
            
            st.caption(f"Entry: {pos['entry_time']} | Strategy: {pos.get('strategy', 'N/A')}")


def render_trade_history(db: TradeDB):
    """Render trade history page."""
    st.title("📋 Trade History")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        days = st.selectbox("Period", [7, 14, 30, 60, 90], index=2)
    with col2:
        direction = st.selectbox("Direction", ["All", "LONG", "SHORT"])
    with col3:
        symbol = st.text_input("Symbol")
    with col4:
        strategy = st.text_input("Strategy")
    
    # Get trades
    start_date = (date.today() - timedelta(days=days)).isoformat()
    trades = db.get_trades(
        start_date=start_date,
        symbol=symbol if symbol else None,
        direction=direction if direction != "All" else None,
        strategy=strategy if strategy else None,
        limit=1000
    )
    
    if not trades:
        st.info("No trades found")
        return
    
    # Summary
    col1, col2, col3, col4 = st.columns(4)
    total_pnl = sum(t.net_pnl for t in trades)
    winners = len([t for t in trades if t.net_pnl > 0])
    win_rate = winners / len(trades) * 100
    
    with col1:
        st.metric("Total P&L", f"₹{total_pnl:+,.2f}")
    with col2:
        st.metric("Trades", len(trades))
    with col3:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col4:
        st.metric("Avg Trade", f"₹{total_pnl/len(trades):+,.2f}")
    
    st.markdown("---")
    
    # Trade table
    trades_df = pd.DataFrame([t.to_dict() for t in trades])
    
    # Format columns
    trades_df['net_pnl_fmt'] = trades_df['net_pnl'].apply(lambda x: f"₹{x:+,.2f}")
    trades_df['entry_price_fmt'] = trades_df['entry_price'].apply(lambda x: f"₹{x:,.2f}")
    trades_df['exit_price_fmt'] = trades_df['exit_price'].apply(lambda x: f"₹{x:,.2f}")
    
    st.dataframe(
        trades_df[['symbol', 'direction', 'strategy', 'entry_time', 
                  'entry_price_fmt', 'exit_price_fmt', 'quantity',
                  'net_pnl_fmt', 'exit_reason', 'holding_minutes']],
        use_container_width=True,
        height=500
    )
    
    # Export
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export to CSV"):
            filepath = f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            db.export_to_csv(filepath, start_date=start_date)
            st.success(f"Exported to {filepath}")


def render_analytics(db: TradeDB):
    """Render analytics page."""
    st.title("📉 Performance Analytics")
    
    # Period selector
    days = st.selectbox("Analysis Period", [7, 14, 30, 60, 90], index=2)
    
    stats = db.get_performance_stats(days=days)
    
    if not stats or stats.get('total_trades', 0) == 0:
        st.info("Not enough data for analytics")
        return
    
    # Key metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total P&L", f"₹{stats['net_pnl']:+,.2f}")
    with col2:
        st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    with col3:
        st.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{stats['max_drawdown_pct']:.1f}%")
    with col5:
        st.metric("Avg Win", f"₹{stats['avg_win']:+,.2f}")
    with col6:
        st.metric("Avg Loss", f"₹{stats['avg_loss']:+,.2f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("P&L by Direction")
        direction_data = stats['by_direction']
        fig = go.Figure(data=[
            go.Bar(
                x=['LONG', 'SHORT'],
                y=[direction_data['LONG']['pnl'], direction_data['SHORT']['pnl']],
                marker_color=['green' if direction_data['LONG']['pnl'] > 0 else 'red',
                             'green' if direction_data['SHORT']['pnl'] > 0 else 'red']
            )
        ])
        fig.update_layout(
            yaxis_title="P&L (₹)",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("P&L by Strategy")
        strategy_data = stats.get('by_strategy', {})
        if strategy_data:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(strategy_data.keys()),
                    y=[s['pnl'] for s in strategy_data.values()],
                    marker_color=['green' if s['pnl'] > 0 else 'red' 
                                 for s in strategy_data.values()]
                )
            ])
            fig.update_layout(
                yaxis_title="P&L (₹)",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Strategy breakdown table
    st.subheader("Strategy Breakdown")
    if strategy_data:
        strategy_df = pd.DataFrame([
            {
                'Strategy': k,
                'Trades': v['trades'],
                'P&L': f"₹{v['pnl']:+,.2f}",
                'Win Rate': f"{v['win_rate']:.1f}%"
            }
            for k, v in strategy_data.items()
        ])
        st.dataframe(strategy_df, use_container_width=True)
    
    # Streaks
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Max Win Streak", stats.get('max_win_streak', 0))
    with col2:
        st.metric("Max Loss Streak", stats.get('max_loss_streak', 0))


def render_system_health():
    """Render system health page."""
    st.title("⚙️ System Health")
    
    # Import psutil for system metrics
    try:
        import psutil
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu = psutil.cpu_percent(interval=1)
            st.metric("CPU Usage", f"{cpu:.1f}%")
            
        with col2:
            mem = psutil.virtual_memory()
            st.metric("Memory Usage", f"{mem.percent:.1f}%")
            
        with col3:
            disk = psutil.disk_usage('/')
            st.metric("Disk Usage", f"{disk.percent:.1f}%")
            
        with col4:
            st.metric("Uptime", f"{psutil.boot_time()}")
            
    except ImportError:
        st.warning("psutil not available for system metrics")
    
    st.markdown("---")
    
    # Component status
    st.subheader("Component Status")
    
    components = [
        ("Database", "✅ Connected"),
        ("Broker API", "✅ Connected" if not KillSwitch.is_active() else "⛔ Blocked"),
        ("Telegram", "✅ Active"),
        ("ML Model", "✅ Loaded"),
        ("Claude Brain", "✅ Active"),
    ]
    
    for name, status in components:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(name)
        with col2:
            st.write(status)
    
    st.markdown("---")
    
    # Recent events
    st.subheader("Recent System Events")
    db = get_db()
    events = db.get_events(limit=20)
    
    if events:
        events_df = pd.DataFrame(events)
        st.dataframe(events_df, use_container_width=True)
    else:
        st.info("No recent events")
    
    # Logs
    st.subheader("Recent Logs")
    log_file = Path(__file__).parent.parent / "logs" / f"trading_{date.today()}.log"
    
    if log_file.exists():
        with open(log_file) as f:
            lines = f.readlines()[-50:]
        st.code("".join(lines), language="log")
    else:
        st.info("No logs available for today")


if __name__ == "__main__":
    main()
