import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .metric-card h3 { font-size: 0.85rem; opacity: 0.8; margin: 0 0 4px; }
    .metric-card h2 { font-size: 1.6rem; font-weight: 700; margin: 0; }
    .metric-card p  { font-size: 0.8rem; margin: 4px 0 0; opacity: 0.75; }
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e3a5f;
        border-left: 4px solid #2563eb;
        padding-left: 10px;
        margin: 8px 0 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path):
    df = pd.read_excel(path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_data
def run_arima(series, order, steps):
    model = ARIMA(series, order=order)
    fit   = model.fit()
    fc    = fit.forecast(steps=steps)
    return fc, fit.aic

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")

    uploaded = st.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded:
        df = pd.read_excel(uploaded, parse_dates=["Date"])
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        st.info("Using default dataset path.  \nUpload your own `Book1.xlsx` to override.")
        try:
            df = load_data("/kaggle/input/stock-market-dataset/Book1.xlsx")
        except Exception:
            # Fallback: generate synthetic data so the dashboard always renders
            np.random.seed(42)
            dates = pd.date_range("2018-01-01", periods=500, freq="B")
            close = 100 + np.cumsum(np.random.randn(500) * 2)
            vol   = np.random.randint(1_000_000, 5_000_000, 500).astype(float)
            ma50  = pd.Series(close).rolling(50, min_periods=1).mean().values
            ma200 = pd.Series(close).rolling(200, min_periods=1).mean().values
            df = pd.DataFrame({
                "Date": dates, "Closing Volume": close,
                "Volume": vol, "50-Day Moving Average": ma50,
                "200-Day Moving Average": ma200,
            })
            st.warning("Default file not found — showing synthetic demo data.")

    st.markdown("---")
    st.subheader("📅 Date Range")
    min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
    date_range = st.date_input("Select range", value=(min_d, max_d),
                               min_value=min_d, max_value=max_d)
    if len(date_range) == 2:
        mask = (df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])
        dff  = df[mask].copy()
    else:
        dff = df.copy()

    st.markdown("---")
    st.subheader("📊 Chart Options")
    show_ma50  = st.checkbox("50-Day MA", value=True)
    show_ma200 = st.checkbox("200-Day MA", value=True)
    show_vol   = st.checkbox("Volume bars", value=True)

    st.markdown("---")
    st.subheader("🔮 ARIMA Forecast")
    forecast_steps = st.slider("Forecast horizon (days)", 7, 90, 30)
    arima_p = st.slider("p (AR order)", 0, 5, 5)
    arima_d = st.slider("d (Differencing)", 0, 2, 0)
    arima_q = st.slider("q (MA order)", 0, 5, 0)
    run_fc   = st.button("▶ Run Forecast")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Stock Market Interactive Dashboard")
st.caption(f"Showing {len(dff):,} trading days · {dff['Date'].min().date()} → {dff['Date'].max().date()}")

# ── KPI cards ─────────────────────────────────────────────────────────────────
latest = dff.iloc[-1]
prev   = dff.iloc[-2] if len(dff) > 1 else dff.iloc[-1]
pct_chg = (latest["Closing Volume"] - prev["Closing Volume"]) / prev["Closing Volume"] * 100
arrow  = "▲" if pct_chg >= 0 else "▼"
color  = "#22c55e" if pct_chg >= 0 else "#ef4444"

col1, col2, col3, col4, col5 = st.columns(5)
cards = [
    ("Latest Close",  f"{latest['Closing Volume']:,.2f}", f"{arrow} {abs(pct_chg):.2f}% vs prev day"),
    ("Avg Close",     f"{dff['Closing Volume'].mean():,.2f}", f"σ = {dff['Closing Volume'].std():,.2f}"),
    ("Max Close",     f"{dff['Closing Volume'].max():,.2f}", f"on {dff.loc[dff['Closing Volume'].idxmax(),'Date'].date()}"),
    ("Min Close",     f"{dff['Closing Volume'].min():,.2f}", f"on {dff.loc[dff['Closing Volume'].idxmin(),'Date'].date()}"),
    ("Avg Volume",    f"{dff['Volume'].mean()/1e6:.2f}M",   f"Total: {dff['Volume'].sum()/1e9:.1f}B"),
]
for col, (title, value, sub) in zip([col1,col2,col3,col4,col5], cards):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{title}</h3>
            <h2>{value}</h2>
            <p>{sub}</p>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📉 Price & MAs", "📦 Volume", "📊 Statistics", "🔮 Forecast"])

# ── Tab 1 : Price chart ───────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Closing Volume with Moving Averages</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dff["Date"], y=dff["Closing Volume"],
        name="Closing Volume", line=dict(color="#2563eb", width=1.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
    ))
    if show_ma50:
        fig.add_trace(go.Scatter(
            x=dff["Date"], y=dff["50-Day Moving Average"],
            name="50-Day MA", line=dict(color="#f59e0b", width=1.5, dash="dot"),
        ))
    if show_ma200:
        fig.add_trace(go.Scatter(
            x=dff["Date"], y=dff["200-Day Moving Average"],
            name="200-Day MA", line=dict(color="#ef4444", width=1.5, dash="dash"),
        ))
    if show_vol:
        fig.add_trace(go.Bar(
            x=dff["Date"], y=dff["Volume"],
            name="Volume", yaxis="y2",
            marker_color="rgba(100,116,139,0.25)",
        ))

    fig.update_layout(
        height=480,
        yaxis=dict(title="Price / Closing Volume", showgrid=True, gridcolor="#f1f5f9"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        xaxis=dict(showgrid=False, rangeslider=dict(visible=True)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # MA crossover signal table
    st.markdown('<div class="section-header">MA Crossover Signals (last 10)</div>', unsafe_allow_html=True)
    if "50-Day Moving Average" in dff.columns and "200-Day Moving Average" in dff.columns:
        dff["signal"] = np.where(dff["50-Day Moving Average"] > dff["200-Day Moving Average"], 1, -1)
        dff["crossover"] = dff["signal"].diff()
        signals = dff[dff["crossover"] != 0][["Date","Closing Volume","50-Day Moving Average","200-Day Moving Average","crossover"]].tail(10).copy()
        signals["Type"] = signals["crossover"].apply(lambda x: "🟢 Golden Cross" if x > 0 else "🔴 Death Cross")
        st.dataframe(signals.drop(columns="crossover").set_index("Date"), use_container_width=True)

# ── Tab 2 : Volume ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Volume Over Time</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig_vol = px.bar(dff, x="Date", y="Volume",
                         color_discrete_sequence=["#6366f1"])
        fig_vol.update_layout(height=320, plot_bgcolor="white",
                              margin=dict(l=20,r=20,t=20,b=20),
                              xaxis=dict(showgrid=False),
                              yaxis=dict(showgrid=True, gridcolor="#f1f5f9"))
        st.plotly_chart(fig_vol, use_container_width=True)

    with c2:
        fig_hist = px.histogram(dff, x="Volume", nbins=40,
                                color_discrete_sequence=["#6366f1"],
                                labels={"Volume":"Volume"},
                                title="Volume Distribution")
        fig_hist.update_layout(height=320, plot_bgcolor="white",
                               margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    # Monthly aggregation
    st.markdown('<div class="section-header">Monthly Average Closing Volume</div>', unsafe_allow_html=True)
    monthly = dff.set_index("Date").resample("ME")["Closing Volume"].mean().reset_index()
    monthly.columns = ["Month", "Avg Close"]
    fig_m = px.area(monthly, x="Month", y="Avg Close",
                    color_discrete_sequence=["#2563eb"])
    fig_m.update_layout(height=280, plot_bgcolor="white",
                        margin=dict(l=20,r=20,t=20,b=20),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"))
    st.plotly_chart(fig_m, use_container_width=True)

# ── Tab 3 : Statistics ────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
    num_cols = ["Closing Volume","Volume","50-Day Moving Average","200-Day Moving Average"]
    st.dataframe(dff[num_cols].describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = dff[num_cols].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="Blues",
                         aspect="auto", title="")
    fig_corr.update_layout(height=360, margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown('<div class="section-header">Distribution Plots</div>', unsafe_allow_html=True)
    fig_box = go.Figure()
    for col, clr in zip(["Closing Volume","50-Day Moving Average","200-Day Moving Average"],
                        ["#2563eb","#f59e0b","#ef4444"]):
        fig_box.add_trace(go.Box(y=dff[col], name=col, marker_color=clr,
                                 boxmean="sd"))
    fig_box.update_layout(height=340, plot_bgcolor="white",
                          margin=dict(l=20,r=20,t=20,b=20),
                          yaxis=dict(showgrid=True, gridcolor="#f1f5f9"))
    st.plotly_chart(fig_box, use_container_width=True)

# ── Tab 4 : Forecast ──────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">ARIMA Forecast of Closing Volume</div>', unsafe_allow_html=True)
    st.info(f"Model: ARIMA({arima_p},{arima_d},{arima_q}) · Horizon: {forecast_steps} days  \n"
            "Adjust parameters in the sidebar, then click **▶ Run Forecast**.")

    if run_fc:
        with st.spinner("Fitting ARIMA model …"):
            try:
                fc, aic = run_arima(dff["Closing Volume"].values,
                                    (arima_p, arima_d, arima_q),
                                    forecast_steps)
                last_date  = dff["Date"].iloc[-1]
                fc_dates   = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                           periods=forecast_steps, freq="D")
                df_fc = pd.DataFrame({"Date": fc_dates, "Forecast": fc})

                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=dff["Date"], y=dff["Closing Volume"],
                    name="Historical", line=dict(color="#2563eb", width=1.5),
                ))
                fig_fc.add_trace(go.Scatter(
                    x=df_fc["Date"], y=df_fc["Forecast"],
                    name="Forecast", line=dict(color="#f59e0b", width=2, dash="dash"),
                    fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
                ))
                fig_fc.update_layout(
                    height=420, plot_bgcolor="white",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Closing Volume"),
                    legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
                    margin=dict(l=20,r=20,t=40,b=20),
                )
                st.plotly_chart(fig_fc, use_container_width=True)

                kc1, kc2, kc3 = st.columns(3)
                kc1.metric("AIC", f"{aic:.2f}")
                kc2.metric("Forecast Mean", f"{df_fc['Forecast'].mean():,.2f}")
                kc3.metric("Forecast Range",
                           f"{df_fc['Forecast'].min():,.2f} – {df_fc['Forecast'].max():,.2f}")

                st.markdown('<div class="section-header">Forecast Table</div>', unsafe_allow_html=True)
                st.dataframe(df_fc.set_index("Date").style.format({"Forecast":"{:,.2f}"}),
                             use_container_width=True)

            except Exception as e:
                st.error(f"ARIMA fitting failed: {e}. Try different (p,d,q) values.")
    else:
        st.markdown("*Click **▶ Run Forecast** in the sidebar to generate predictions.*")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Stock Market Dashboard · built with Streamlit & Plotly")
