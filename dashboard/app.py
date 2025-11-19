"""
Main Streamlit Dashboard Application

Interactive dashboard for:
- Universe construction visualization
- Basket formation and correlation analysis
- Cointegration testing results
- Signal generation and live monitoring
- Backtesting with detailed analytics
- Risk management monitoring
- Parameter tweaking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.ibkr_connector import IBKRConnector
from src.data.universe import UniverseBuilder
from src.strategy.basket_formation import BasketFormer
from src.strategy.cointegration import CointegrationTester
from src.strategy.signals import SignalGenerator
from src.strategy.portfolio import PortfolioManager
from src.risk.risk_management import RiskManager
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import PerformanceMetrics

# Page configuration
st.set_page_config(
    page_title="Dividend Cointegration Strategy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ibkr_connected' not in st.session_state:
    st.session_state.ibkr_connected = False
if 'universe_df' not in st.session_state:
    st.session_state.universe_df = None
if 'baskets' not in st.session_state:
    st.session_state.baskets = None
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'price_data' not in st.session_state:
    st.session_state.price_data = {}


def main():
    """Main dashboard application."""

    # Header
    st.markdown('<h1 class="main-header">📊 Dividend Cointegration Strategy Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # IBKR Connection Status
        st.subheader("IBKR Connection")
        if st.session_state.ibkr_connected:
            st.success("✅ Connected")
            if st.button("Disconnect"):
                st.session_state.ibkr_connected = False
                st.rerun()
        else:
            st.warning("⚠️ Not Connected")
            if st.button("Connect to IBKR"):
                try:
                    # In production, this would actually connect
                    st.info("Note: Using simulated connection. Configure IBKR credentials in config/ibkr_config.yaml")
                    st.session_state.ibkr_connected = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {e}")

        st.divider()

        # Strategy Parameters
        st.subheader("📋 Strategy Parameters")

        with st.expander("Universe Filters"):
            market_cap_min = st.number_input(
                "Min Market Cap ($B)",
                min_value=0.1,
                value=1.0,
                step=0.1
            )

            min_div_years = st.slider(
                "Min Dividend Years",
                min_value=3,
                max_value=10,
                value=5
            )

            avg_volume_min = st.number_input(
                "Min Avg Volume ($M)",
                min_value=1.0,
                value=10.0,
                step=1.0
            )

        with st.expander("Basket Formation"):
            corr_threshold = st.slider(
                "Correlation Threshold",
                min_value=0.5,
                max_value=0.95,
                value=0.85,
                step=0.05
            )

            cluster_size_min = st.slider(
                "Min Cluster Size",
                min_value=3,
                max_value=10,
                value=6
            )

            cluster_size_max = st.slider(
                "Max Cluster Size",
                min_value=5,
                max_value=15,
                value=10
            )

        with st.expander("Cointegration Testing"):
            johansen_pvalue = st.slider(
                "Johansen p-value threshold",
                min_value=0.01,
                max_value=0.10,
                value=0.01,
                step=0.01
            )

            half_life_max = st.slider(
                "Max Half-Life (days)",
                min_value=10,
                max_value=50,
                value=35
            )

            hurst_max = st.slider(
                "Max Hurst Exponent",
                min_value=0.3,
                max_value=0.5,
                value=0.45,
                step=0.05
            )

        with st.expander("Signal Generation"):
            entry_zscore = st.slider(
                "Entry Z-Score",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.1
            )

            exit_zscore = st.slider(
                "Exit Z-Score",
                min_value=0.0,
                max_value=1.5,
                value=0.5,
                step=0.1
            )

        with st.expander("Risk Management"):
            max_position_pct = st.slider(
                "Max Position (%)",
                min_value=1,
                max_value=20,
                value=8
            )

            max_drawdown_pct = st.slider(
                "Max Drawdown (%)",
                min_value=5,
                max_value=20,
                value=15
            )

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Universe & Data",
        "🔗 Basket Formation",
        "🔬 Cointegration Analysis",
        "📡 Live Signals",
        "⚡ Backtesting",
        "🛡️ Risk Monitor"
    ])

    with tab1:
        render_universe_tab()

    with tab2:
        render_basket_tab()

    with tab3:
        render_cointegration_tab()

    with tab4:
        render_signals_tab()

    with tab5:
        render_backtest_tab()

    with tab6:
        render_risk_tab()


def render_universe_tab():
    """Render universe construction tab."""
    st.header("Universe Construction")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Actions")

        if st.button("🔄 Build Universe", type="primary"):
            with st.spinner("Building universe..."):
                # Placeholder for actual universe building
                st.info("Universe building requires IBKR connection with market data subscription.")
                st.info("For demonstration, you can load a pre-built universe CSV.")

        uploaded_file = st.file_uploader("Or upload universe CSV", type=['csv'])

        if uploaded_file is not None:
            st.session_state.universe_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(st.session_state.universe_df)} stocks from CSV")

    with col2:
        if st.session_state.universe_df is not None:
            st.metric("Total Stocks", len(st.session_state.universe_df))

            if 'market_cap' in st.session_state.universe_df.columns:
                avg_mcap = st.session_state.universe_df['market_cap'].mean()
                st.metric("Avg Market Cap", f"${avg_mcap/1e9:.2f}B")

    # Display universe
    if st.session_state.universe_df is not None:
        st.subheader("Universe Stocks")

        st.dataframe(
            st.session_state.universe_df,
            use_container_width=True,
            height=400
        )

        # Download button
        csv = st.session_state.universe_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Universe CSV",
            data=csv,
            file_name=f"universe_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


def render_basket_tab():
    """Render basket formation tab."""
    st.header("Basket Formation & Correlation Analysis")

    if st.session_state.universe_df is None:
        st.warning("⚠️ Please build or load universe first (Universe & Data tab)")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🔗 Form Baskets", type="primary"):
            with st.spinner("Forming baskets..."):
                st.info("Basket formation requires historical price data from IBKR.")
                st.info("For demonstration, you can load pre-formed baskets CSV.")

        uploaded_file = st.file_uploader("Or upload baskets CSV", type=['csv'])

        if uploaded_file is not None:
            baskets_df = pd.read_csv(uploaded_file)

            # Convert to baskets format
            st.session_state.baskets = []
            for _, row in baskets_df.iterrows():
                st.session_state.baskets.append({
                    'sector': row['sector'],
                    'symbols': row['symbols'].split(','),
                    'size': row['size'],
                    'avg_correlation': row['avg_correlation']
                })

            st.success(f"Loaded {len(st.session_state.baskets)} baskets")

    with col2:
        if st.session_state.baskets is not None:
            st.metric("Total Baskets", len(st.session_state.baskets))

            avg_size = np.mean([b['size'] for b in st.session_state.baskets])
            st.metric("Avg Basket Size", f"{avg_size:.1f}")

            avg_corr = np.mean([b['avg_correlation'] for b in st.session_state.baskets])
            st.metric("Avg Correlation", f"{avg_corr:.3f}")

    # Display baskets
    if st.session_state.baskets is not None:
        st.subheader("Baskets by Sector")

        # Convert to DataFrame for display
        baskets_display = pd.DataFrame([
            {
                'Basket ID': i,
                'Sector': b['sector'],
                'Size': b['size'],
                'Avg Correlation': f"{b['avg_correlation']:.3f}",
                'Symbols': ', '.join(b['symbols'][:3]) + ('...' if len(b['symbols']) > 3 else '')
            }
            for i, b in enumerate(st.session_state.baskets)
        ])

        st.dataframe(baskets_display, use_container_width=True)

        # Basket details
        selected_basket = st.selectbox(
            "Select basket to view details",
            range(len(st.session_state.baskets)),
            format_func=lambda x: f"Basket {x}: {st.session_state.baskets[x]['sector']}"
        )

        if selected_basket is not None:
            basket = st.session_state.baskets[selected_basket]

            st.subheader(f"Basket {selected_basket} Details")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Symbols:**")
                st.write(", ".join(basket['symbols']))

            with col2:
                st.write(f"**Sector:** {basket['sector']}")
                st.write(f"**Size:** {basket['size']}")
                st.write(f"**Avg Correlation:** {basket['avg_correlation']:.3f}")

            # Placeholder for correlation matrix visualization
            st.subheader("Correlation Matrix")
            st.info("Correlation matrix visualization requires price data. Connect to IBKR or upload price data.")


def render_cointegration_tab():
    """Render cointegration analysis tab."""
    st.header("Cointegration Testing & Analysis")

    if st.session_state.baskets is None:
        st.warning("⚠️ Please form baskets first (Basket Formation tab)")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🔬 Test Cointegration", type="primary"):
            with st.spinner("Testing cointegration..."):
                st.info("Cointegration testing requires price data from IBKR.")
                st.info("For demonstration, you can load pre-tested vectors CSV.")

        uploaded_file = st.file_uploader("Or upload vectors CSV", type=['csv'])

        if uploaded_file is not None:
            vectors_df = pd.read_csv(uploaded_file)

            # Convert to vectors format
            st.session_state.vectors = []
            for _, row in vectors_df.iterrows():
                st.session_state.vectors.append({
                    'symbols': row['symbols'].split(','),
                    'weights': np.array([float(w) for w in row['weights'].split(',')]),
                    'eigenvalue': row['eigenvalue'],
                    'half_life': row['half_life'],
                    'hurst': row['hurst'],
                    'basket_sector': row['sector']
                })

            st.success(f"Loaded {len(st.session_state.vectors)} cointegration vectors")

    with col2:
        if st.session_state.vectors is not None:
            st.metric("Valid Vectors", len(st.session_state.vectors))

            avg_hl = np.mean([v['half_life'] for v in st.session_state.vectors])
            st.metric("Avg Half-Life", f"{avg_hl:.1f} days")

            avg_hurst = np.mean([v['hurst'] for v in st.session_state.vectors])
            st.metric("Avg Hurst", f"{avg_hurst:.3f}")

    # Display vectors
    if st.session_state.vectors is not None:
        st.subheader("Cointegration Vectors")

        vectors_display = pd.DataFrame([
            {
                'Vector ID': i,
                'Sector': v['basket_sector'],
                'Stocks': len(v['symbols']),
                'Half-Life': f"{v['half_life']:.2f}",
                'Hurst': f"{v['hurst']:.3f}",
                'Eigenvalue': f"{v['eigenvalue']:.4f}",
                'Symbols': ', '.join(v['symbols'][:3]) + ('...' if len(v['symbols']) > 3 else '')
            }
            for i, v in enumerate(st.session_state.vectors)
        ])

        st.dataframe(vectors_display, use_container_width=True)

        # Vector details
        selected_vector = st.selectbox(
            "Select vector to analyze",
            range(len(st.session_state.vectors)),
            format_func=lambda x: f"Vector {x}: {', '.join(st.session_state.vectors[x]['symbols'][:2])}..."
        )

        if selected_vector is not None:
            vector = st.session_state.vectors[selected_vector]

            st.subheader(f"Vector {selected_vector} Analysis")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Half-Life", f"{vector['half_life']:.2f} days")

            with col2:
                st.metric("Hurst Exponent", f"{vector['hurst']:.3f}")

            with col3:
                st.metric("Eigenvalue", f"{vector['eigenvalue']:.4f}")

            with col4:
                st.metric("Num Stocks", len(vector['symbols']))

            # Weights
            st.subheader("Cointegration Weights")

            weights_df = pd.DataFrame({
                'Symbol': vector['symbols'],
                'Weight': vector['weights']
            })

            fig = px.bar(
                weights_df,
                x='Symbol',
                y='Weight',
                title='Cointegration Weights by Stock'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Time series plot placeholder
            st.subheader("Price Series & Spread")
            st.info("Price series and spread visualization requires historical data. Connect to IBKR or upload price data.")


def render_signals_tab():
    """Render live signals tab."""
    st.header("Live Signal Generation")

    if st.session_state.vectors is None:
        st.warning("⚠️ Please test cointegration first (Cointegration Analysis tab)")
        return

    st.subheader("Current Signals")

    # Placeholder for live signals
    st.info("Live signal generation requires real-time data from IBKR.")
    st.info("Connect to IBKR and enable market data subscriptions to see live signals.")

    # Example signal table
    st.subheader("Signal Monitor")

    example_signals = pd.DataFrame([
        {
            'Vector ID': 0,
            'Action': 'HOLD',
            'Z-Score': -0.8,
            'Spread': 102.5,
            'Entry Date': '2024-01-15',
            'P&L': '+$1,250'
        },
        {
            'Vector ID': 1,
            'Action': 'ENTER_LONG',
            'Z-Score': -1.6,
            'Spread': 98.2,
            'Entry Date': '-',
            'P&L': '-'
        }
    ])

    st.dataframe(example_signals, use_container_width=True)


def render_backtest_tab():
    """Render backtesting tab."""
    st.header("Strategy Backtesting")

    if st.session_state.vectors is None:
        st.warning("⚠️ Please test cointegration first (Cointegration Analysis tab)")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Backtest Configuration")

        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=10_000,
            value=1_000_000,
            step=10_000
        )

        date_range = st.date_input(
            "Backtest Period",
            value=(datetime.now() - timedelta(days=365), datetime.now())
        )

        if st.button("▶️ Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                st.info("Backtesting requires historical price data from IBKR.")
                st.info("For demonstration, you can upload backtest results CSV.")

    with col2:
        uploaded_file = st.file_uploader("Or upload backtest results", type=['csv'])

        if uploaded_file is not None:
            # Load and process backtest results
            st.success("Backtest results loaded")

    # Display results
    if st.session_state.backtest_results is not None:
        results = st.session_state.backtest_results

        st.subheader("Performance Summary")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        metrics = results.get('metrics', {})

        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0)*100:.2f}%")

        with col2:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")

        with col3:
            st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")

        with col4:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%")

        # Equity curve
        st.subheader("Equity Curve")

        if 'equity_curve' in results:
            equity_df = results['equity_curve']

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=equity_df.index,
                y=equity_df['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ))

            fig.update_layout(
                title='Portfolio Equity Curve',
                xaxis_title='Date',
                yaxis_title='Equity ($)',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Trade table
        st.subheader("Trade History")

        if 'trades' in results and not results['trades'].empty:
            st.dataframe(results['trades'], use_container_width=True)

    else:
        # Placeholder
        st.info("Run backtest or upload results to see performance analysis.")


def render_risk_tab():
    """Render risk monitoring tab."""
    st.header("Risk Management Monitor")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Active Positions", "0")
        st.metric("Trading Status", "🟢 Active")

    with col2:
        st.metric("Portfolio Beta", "0.05")
        st.metric("Current Drawdown", "-2.5%")

    with col3:
        st.metric("Total Exposure", "$0")
        st.metric("VaR (95%)", "$5,250")

    st.subheader("Position Risk Monitor")

    st.info("Position risk monitoring requires active trades. Connect to IBKR and enter positions to see live risk metrics.")

    st.subheader("Risk Stops Triggered")

    st.dataframe(
        pd.DataFrame(columns=['Vector ID', 'Stop Type', 'Timestamp', 'Action Taken']),
        use_container_width=True
    )


if __name__ == "__main__":
    main()
