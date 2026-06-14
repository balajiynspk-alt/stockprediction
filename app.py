import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Local imports
from src.data_collector import StockDataCollector
from src.feature_engineer import FeatureEngineer
from src.model_pipeline import StockPredictionModel
from src.visualizer import StockVisualizer
from src.train import run_pipeline

# Helper to load all NSE symbols from cache or official exchange download
@st.cache_data
def load_nse_symbols():
    csv_path = "data/nse_equities.csv"
    if not os.path.exists(csv_path):
        try:
            import requests
            url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                os.makedirs("data", exist_ok=True)
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.write(r.text)
        except Exception:
            pass
            
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            df = df.dropna(subset=['SYMBOL', 'NAME OF COMPANY'])
            symbols_dict = dict(zip(df['SYMBOL'].str.strip(), df['NAME OF COMPANY'].str.strip()))
            return symbols_dict
        except Exception:
            pass
    return {}

# -------------------------------------------------------------
# 1. STREAMLIT CONFIGURATION & CUSTOM STYLES
# -------------------------------------------------------------
st.set_page_config(
    page_title="Stock Price Prediction & Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern card layout and styling
st.markdown("""
<style>
    /* Main Background & Fonts */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Metrics Card Styling */
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.8rem;
        color: #212529;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: bold;
    }
    .delta-up {
        color: #28a745;
    }
    .delta-down {
        color: #dc3545;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #1e3d59;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'ticker' not in st.session_state:
    st.session_state.ticker = 'AAPL'
if 'years' not in st.session_state:
    st.session_state.years = 5
if 'tune' not in st.session_state:
    st.session_state.tune = False
if 'pipeline_run' not in st.session_state:
    st.session_state.pipeline_run = False
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_features' not in st.session_state:
    st.session_state.df_features = None
if 'df_dataset' not in st.session_state:
    st.session_state.df_dataset = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'engineer' not in st.session_state:
    st.session_state.engineer = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'df_forecast' not in st.session_state:
    st.session_state.df_forecast = None

# Title area
st.title("📈 Stock Price Prediction & Analytics Dashboard")
st.markdown("A production-grade web application predicting stock trends using **Random Forest Regression** and **Technical Analysis**.")

# -------------------------------------------------------------
# 2. SIDEBAR - MODEL SETTINGS
# -------------------------------------------------------------
st.sidebar.header("🛠️ Dashboard Configuration")

# Popular stocks dictionary
POPULAR_STOCKS = {
    "Popular Indian Stocks (NSE)": {
        "RELIANCE": "Reliance Industries Ltd.",
        "TCS": "Tata Consultancy Services Ltd.",
        "INFY": "Infosys Ltd.",
        "HDFCBANK": "HDFC Bank Ltd.",
        "ICICIBANK": "ICICI Bank Ltd.",
        "SBIN": "State Bank of India",
        "BHARTIARTL": "Bharti Airtel Ltd.",
        "TATAMOTORS": "Tata Motors Ltd.",
        "ITC": "ITC Ltd.",
        "LT": "Larsen & Toubro Ltd.",
        "WIPRO": "Wipro Ltd.",
        "AXISBANK": "Axis Bank Ltd.",
        "MARUTI": "Maruti Suzuki India Ltd.",
        "SUNPHARMA": "Sun Pharmaceutical Industries Ltd.",
        "HINDUNILVR": "Hindustan Unilever Ltd.",
        "TATASTEEL": "Tata Steel Ltd.",
        "HCLTECH": "HCL Technologies Ltd.",
        "ONGC": "Oil & Natural Gas Corp. Ltd.",
        "ADANIENT": "Adani Enterprises Ltd.",
        "POWERGRID": "Power Grid Corp. of India Ltd.",
        "NTPC": "NTPC Ltd.",
        "KOTAKBANK": "Kotak Mahindra Bank Ltd.",
        "COALINDIA": "Coal India Ltd.",
        "ULTRACEMCO": "UltraTech Cement Ltd."
    },
    "Popular US Stocks": {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corporation",
        "AMZN": "Amazon.com Inc.",
        "GOOGL": "Alphabet Inc. (Google)",
        "META": "Meta Platforms Inc.",
        "NFLX": "Netflix Inc."
    }
}

category_choice = st.sidebar.selectbox(
    "Stock Category",
    ["Indian Stocks (NSE)", "US Stocks", "Custom Ticker..."]
)

ticker_input = st.session_state.ticker

if category_choice == "Indian Stocks (NSE)":
    nse_symbols = load_nse_symbols()
    if nse_symbols:
        options = [f"{sym} - {nse_symbols[sym]}" for sym in sorted(nse_symbols.keys())]
        default_index = 0
        current_ticker = st.session_state.ticker
        clean_current = current_ticker[:-3] if current_ticker.endswith(".NS") else current_ticker
        if clean_current in nse_symbols:
            default_index = sorted(nse_symbols.keys()).index(clean_current)
            
        selected_stock = st.sidebar.selectbox(
            "Choose Indian Stock",
            options=options,
            index=default_index
        )
        ticker_input = selected_stock.split(" - ")[0]
    else:
        indian_stocks = POPULAR_STOCKS["Popular Indian Stocks (NSE)"]
        selected_stock = st.sidebar.selectbox(
            "Choose Indian Stock (Popular)",
            options=[f"{symbol} ({name})" for symbol, name in indian_stocks.items()]
        )
        ticker_input = selected_stock.split(" ")[0]
elif category_choice == "US Stocks":
    us_stocks = POPULAR_STOCKS["Popular US Stocks"]
    selected_stock = st.sidebar.selectbox(
        "Choose US Stock",
        options=[f"{symbol} ({name})" for symbol, name in us_stocks.items()]
    )
    ticker_input = selected_stock.split(" ")[0]
else:
    ticker_input = st.sidebar.text_input(
        "Enter Custom Ticker",
        value=st.session_state.ticker if st.session_state.ticker not in [s for cat in POPULAR_STOCKS.values() for s in cat] else "AAPL",
        help="Enter any ticker symbol (e.g. RELIANCE, TCS, AAPL, TSLA, etc.)"
    ).upper().strip()

years_input = st.sidebar.slider(
    "Historical Data (Years)",
    min_value=1,
    max_value=10,
    value=st.session_state.years,
    help="Collect at least 5 years of historical data for model stability."
)

tune_input = st.sidebar.checkbox(
    "Optimize Hyperparameters (GridSearch)",
    value=st.session_state.tune,
    help="If checked, runs a time-series cross-validated grid search. Warning: Takes longer to complete."
)

run_button = st.sidebar.button("🚀 Run Prediction Pipeline", type="primary", use_container_width=True)

# -------------------------------------------------------------
# 3. RUN PIPELINE OR LOAD FROM CACHE
# -------------------------------------------------------------
collector = StockDataCollector(cache_dir="data")
engineer = FeatureEngineer()

model_pkl_path = os.path.join("models", f"{ticker_input.lower()}_rf_model.pkl")

# Helper function to trigger pipeline
def trigger_pipeline(ticker, years, tune):
    with st.spinner(f"Running pipeline for {ticker}... (fetching data, engineering features, training Random Forest)"):
        try:
            model, eng, metrics, df_fc = run_pipeline(ticker, years=years, tune=tune)
            # Store in session state
            st.session_state.ticker = ticker
            st.session_state.years = years
            st.session_state.tune = tune
            st.session_state.model = model
            st.session_state.engineer = eng
            st.session_state.metrics = metrics
            st.session_state.df_forecast = df_fc
            st.session_state.pipeline_run = True
            
            # Re-read raw and featured data
            df_raw = collector.fetch_data(ticker, years=years)
            df_clean = eng.preprocess_data(df_raw)
            df_feats = eng.add_technical_indicators(df_clean)
            st.session_state.df_raw = df_raw
            st.session_state.df_features = df_feats
            
            st.success(f"Pipeline executed successfully for {ticker}!")
        except Exception as e:
            st.session_state.pipeline_run = False
            st.session_state.df_features = None
            st.error(f"Error executing pipeline: {e}")

# Handle click or auto-load cache if available
if run_button:
    trigger_pipeline(ticker_input, years_input, tune_input)
elif not st.session_state.pipeline_run:
    # Check if a saved model exists for the requested ticker
    if os.path.exists(model_pkl_path):
        with st.spinner(f"Loading cached model for {ticker_input}..."):
            try:
                model, eng = StockPredictionModel.load_model(model_pkl_path)
                st.session_state.ticker = ticker_input
                st.session_state.model = model
                st.session_state.engineer = eng
                st.session_state.metrics = model.metrics
                st.session_state.pipeline_run = True
                
                df_raw = collector.fetch_data(ticker_input, years=years_input)
                df_clean = eng.preprocess_data(df_raw)
                df_feats = eng.add_technical_indicators(df_clean)
                st.session_state.df_raw = df_raw
                st.session_state.df_features = df_feats
                
                # Re-generate forecast
                df_fc = model.predict_future_days(df_feats, eng, days=7)
                st.session_state.df_forecast = df_fc
                
                st.success(f"Loaded cached model for {ticker_input} successfully!")
            except Exception as e:
                st.warning(f"Could not load cache: {e}. Running full pipeline.")
                trigger_pipeline(ticker_input, years_input, tune_input)
    else:
        # Run pipeline on initial load
        trigger_pipeline(ticker_input, years_input, tune_input)

# -------------------------------------------------------------
# 4. RENDERING MAIN DASHBOARD
# -------------------------------------------------------------
if st.session_state.pipeline_run and st.session_state.df_features is not None:
    df_raw = st.session_state.df_raw
    df_features = st.session_state.df_features
    df_forecast = st.session_state.df_forecast
    metrics = st.session_state.metrics
    model = st.session_state.model
    eng = st.session_state.engineer
    
    # 4.1. TOP KPI METRICS SECTION
    st.markdown("### 📊 Real-time Analysis & Performance Metrics")
    
    last_actual_row = df_features.iloc[-1]
    last_close = last_actual_row['Close']
    next_day_forecast = df_forecast.iloc[0]['Predicted_Close']
    price_diff = next_day_forecast - last_close
    pct_diff = (price_diff / last_close) * 100
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    
    # KPI 1: Last Close
    with col_kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Last Close Price</div>
            <div class="metric-value">${last_close:.2f}</div>
            <div class="metric-delta">As of {df_features.index[-1].strftime('%Y-%m-%d')}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # KPI 2: Next Day Predict
    delta_class = "delta-up" if price_diff >= 0 else "delta-down"
    arrow = "▲" if price_diff >= 0 else "▼"
    with col_kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Next Close</div>
            <div class="metric-value">${next_day_forecast:.2f}</div>
            <div class="metric-delta {delta_class}">{arrow} {price_diff:+.2f} ({pct_diff:+.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)
        
    # KPI 3: R2 Score
    r2_score_val = metrics.get('R2', 0.0)
    with col_kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Test R² Score</div>
            <div class="metric-value">{r2_score_val:.4f}</div>
            <div class="metric-delta">Variance Explained</div>
        </div>
        """, unsafe_allow_html=True)
        
    # KPI 4: MAE
    mae_val = metrics.get('MAE', 0.0)
    with col_kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MAE (Test Set)</div>
            <div class="metric-value">${mae_val:.2f}</div>
            <div class="metric-delta">Mean Absolute Error</div>
        </div>
        """, unsafe_allow_html=True)
        
    # KPI 5: MAPE
    mape_val = metrics.get('MAPE', 0.0)
    with col_kpi5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MAPE</div>
            <div class="metric-value">{mape_val:.2f}%</div>
            <div class="metric-delta">Mean Absolute % Error</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4.2. TABS CREATION
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Technical Indicators Chart", 
        "🔬 Model Evaluation & Residuals", 
        "💡 Explainability & Feature Importance", 
        "🔮 7-Day Future Forecast"
    ])
    
    # -------------------------------------------------------------
    # TAB 1: TECHNICAL INDICATORS CHART
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Interactive Market Chart")
        st.markdown("Use the checkboxes below to toggle technical indicators over the historical close price.")
        
        col_selects1, col_selects2, col_selects3 = st.columns(3)
        with col_selects1:
            show_smas = st.multiselect("Simple Moving Averages (SMA)", ["SMA_10", "SMA_20", "SMA_50"], default=["SMA_20", "SMA_50"])
        with col_selects2:
            show_emas = st.multiselect("Exponential Moving Averages (EMA)", ["EMA_10", "EMA_20"])
        with col_selects3:
            show_bb = st.toggle("Show Bollinger Bands", value=True)
            
        # Draw interactive plotly chart
        # Create subplots: main price chart & indicators (top), volume/RSI/MACD (bottom)
        chart_df = df_features.tail(250) # Show last ~1 year of data for clarity
        
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05, 
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # 1. Main Price Line / Candlesticks
        fig.add_trace(
            go.Candlestick(
                x=chart_df.index,
                open=chart_df['Open'],
                high=chart_df['High'],
                low=chart_df['Low'],
                close=chart_df['Close'],
                name="Stock Price"
            ),
            row=1, col=1
        )
        
        # SMAs
        for sma in show_smas:
            fig.add_trace(
                go.Scatter(x=chart_df.index, y=chart_df[sma], name=sma, line=dict(width=1.5)),
                row=1, col=1
            )
            
        # EMAs
        for ema in show_emas:
            fig.add_trace(
                go.Scatter(x=chart_df.index, y=chart_df[ema], name=ema, line=dict(width=1.5, dash='dash')),
                row=1, col=1
            )
            
        # Bollinger Bands
        if show_bb:
            fig.add_trace(
                go.Scatter(x=chart_df.index, y=chart_df['BB_Upper'], name='BB Upper', line=dict(color='rgba(173, 181, 189, 0.5)', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=chart_df.index, 
                    y=chart_df['BB_Lower'], 
                    name='BB Lower', 
                    line=dict(color='rgba(173, 181, 189, 0.5)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(173, 181, 189, 0.15)'
                ),
                row=1, col=1
            )
            
        # 2. RSI Subplot
        fig.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df['RSI'], name='RSI', line=dict(color='#8e44ad', width=1.5)),
            row=2, col=1
        )
        # Add RSI oversold/overbought markers
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold")
        
        # 3. MACD Subplot
        fig.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df['MACD'], name='MACD', line=dict(color='#2980b9', width=1.5)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df['MACD_Signal'], name='MACD Signal', line=dict(color='#e67e22', width=1.5)),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=chart_df.index, y=chart_df['MACD_Hist'], name='MACD Hist', marker_color='#27ae60'),
            row=3, col=1
        )
        
        # Layout modifications
        fig.update_layout(
            height=800,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10)
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI (14)", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    # -------------------------------------------------------------
    # TAB 2: MODEL EVALUATION & RESIDUALS
    # -------------------------------------------------------------
    with tab2:
        st.subheader("🔬 Model Diagnostic Charts")
        st.markdown("Visualisations showing the prediction accuracy, error properties, and features relationships on the test split.")
        
        # Display saved matplotlib plots to ensure identical quality
        # File paths:
        path_avp = f"images/{st.session_state.ticker}_actual_vs_predicted.png"
        path_resid = f"images/{st.session_state.ticker}_residuals_distribution.png"
        path_heatmap = f"images/{st.session_state.ticker}_correlation_heatmap.png"
        
        col_diag1, col_diag2 = st.columns(2)
        with col_diag1:
            st.markdown("#### Actual vs Predicted Prices")
            if os.path.exists(path_avp):
                st.image(path_avp, use_container_width=True)
            else:
                st.info("Actual vs Predicted plot not generated yet.")
                
        with col_diag2:
            st.markdown("#### Residual Error Distribution")
            if os.path.exists(path_resid):
                st.image(path_resid, use_container_width=True)
            else:
                st.info("Residuals plot not generated yet.")
                
        st.markdown("<hr>", unsafe_allow_html=True)
        
        col_diag3, col_diag4 = st.columns([1.3, 0.7])
        with col_diag3:
            st.markdown("#### Feature Correlation Heatmap")
            if os.path.exists(path_heatmap):
                st.image(path_heatmap, use_container_width=True)
            else:
                st.info("Correlation Heatmap not generated yet.")
        with col_diag4:
            st.markdown("##### Interpretation of Diagnostic Charts:")
            st.markdown("""
            * **Actual vs Predicted**: Checks the alignment in time. A close overlap indicates the model successfully catches trend directions and local peaks.
            * **Residuals Distribution**: Ideally a narrow bell curve (Normal Distribution) centered at zero. Any skewness or multi-modality indicates systematic bias.
            * **Correlation Heatmap**: Visualizes linear dependencies. High correlation (multicollinearity) between features is typical in technical indicators (e.g. SMA/EMA lines) but Random Forest handles this robustly due to bootstrap aggregation.
            """)
            
    # -------------------------------------------------------------
    # TAB 3: EXPLAINABILITY & FEATURE IMPORTANCE
    # -------------------------------------------------------------
    with tab3:
        st.subheader("💡 Explainability Analysis")
        st.markdown("Which technical indicators mattered the most for predicting tomorrow's closing price?")
        
        path_feat = f"images/{st.session_state.ticker}_feature_importance.png"
        df_importance = model.get_feature_importances(eng.feature_cols)
        
        col_exp1, col_exp2 = st.columns([0.6, 0.4])
        with col_exp1:
            if os.path.exists(path_feat):
                st.image(path_feat, use_container_width=True)
            else:
                st.info("Feature importance plot not generated yet.")
        with col_exp2:
            st.markdown("#### Feature Importance Rankings")
            st.dataframe(
                df_importance.style.format({'Importance': '{:.4f}'}).background_gradient(cmap='viridis'),
                use_container_width=True,
                height=450
            )
            
        st.markdown("#### Technical Insights & Analysis")
        # Generate automated summary of importances
        top_features = df_importance['Feature'].head(3).tolist()
        st.write(f"""
        For **{st.session_state.ticker}**, the top three contributing technical indicators are **{', '.join(top_features)}**.
        
        * **Moving Averages (SMA/EMA)**: High importance scores for SMA and EMA lines indicate that the model is heavily anchoring its predictions on recent price trends. This is typical in markets with strong directional momentum.
        * **Oscillators (RSI / MACD)**: If RSI or MACD ranks highly, it implies that trend momentum and overbought/oversold boundaries are critical indicators for next-day reversals.
        * **Volatilities & Volumes**: Bollinger Bands and Volatility features capture the variance of returns. High importance here suggests the model uses price volatility to adjust its expected boundaries.
        """)

    # -------------------------------------------------------------
    # TAB 4: 7-DAY FUTURE FORECAST
    # -------------------------------------------------------------
    with tab4:
        st.subheader("🔮 Recursive 7-Day Trading Forecast")
        st.markdown("Predictions for the next 7 active trading days using the trained Random Forest model in recursive forecasting mode.")
        
        col_fc1, col_fc2 = st.columns([0.6, 0.4])
        
        with col_fc1:
            # Interactive forecast plot: show last 30 days of actual close + 7 days forecast
            hist_subset = df_features.tail(30)
            
            fig_fc = go.Figure()
            
            # Historical actual line
            fig_fc.add_trace(
                go.Scatter(
                    x=hist_subset.index, 
                    y=hist_subset['Close'], 
                    name='Historical Close',
                    line=dict(color='#1f77b4', width=2)
                )
            )
            
            # Connect actual close to prediction
            fc_dates = pd.to_datetime(df_forecast['Date'])
            fc_prices = df_forecast['Predicted_Close']
            
            # Combine last actual point with forecast to draw continuous line
            connect_dates = [hist_subset.index[-1]] + list(fc_dates)
            connect_prices = [hist_subset['Close'].iloc[-1]] + list(fc_prices)
            
            fig_fc.add_trace(
                go.Scatter(
                    x=connect_dates, 
                    y=connect_prices, 
                    name='7-Day Forecast',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                )
            )
            
            fig_fc.update_layout(
                title=f"{st.session_state.ticker} Future Forecast Curve",
                xaxis_title="Date",
                yaxis_title="Stock Close Price",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=450,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            
            st.plotly_chart(fig_fc, use_container_width=True)
            
        with col_fc2:
            st.markdown("#### Forecast Results Table")
            st.dataframe(
                df_forecast.style.format({'Predicted_Close': '${:.2f}'}),
                use_container_width=True,
                hide_index=True
            )
            
            # Download forecast as CSV
            csv_data = df_forecast.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Forecast Data",
                data=csv_data,
                file_name=f"{st.session_state.ticker}_7_day_forecast.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown("""
            > **Disclaimer**: This tool is built purely for educational and analysis purposes using Random Forest Regression. 
            > Stock markets are highly complex and stochastic. **Do not use this forecast as financial advice.**
            """)
else:
    st.info("Please enter a stock ticker in the sidebar and run the prediction pipeline to generate predictions.")
