from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from multi_market_engine import run_multimarket_forecast
from pjm_markets import get_market, market_options
from utils import DEFAULT_APP_DIR, load_feature_store, get_available_run_days
from models import evaluate_models_for_single_day

st.set_page_config(page_title="PJM Runtime-Constrained Price Forecasting", page_icon="⚡", layout="wide")

st.markdown(
    """
<style>
.block-container {max-width: 1220px; padding-top: 1.1rem; padding-bottom: 1rem;}
.hero {padding: 1rem 1.2rem 0.5rem 0.2rem;}
.hero h1 {margin: 0; color: #1d2a44; font-size: 3rem; font-weight: 800;}
.card {background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%); border: 1px solid #e7eef9; border-radius: 18px; padding: 1rem 1.1rem; box-shadow: 0 8px 24px rgba(20, 40, 90, 0.07);}
.metric-box {background: linear-gradient(135deg, #edf4ff 0%, #ffffff 100%); border: 1px solid #dbe7ff; border-radius: 16px; padding: 0.9rem 1rem; box-shadow: 0 6px 18px rgba(31, 71, 136, 0.08);}
.metric-label {font-size: 0.9rem; color: #62718a; margin-bottom: 0.2rem;}
.metric-value {font-size: 1.65rem; font-weight: 800; color: #1d2a44;}
div[data-testid="stTabs"] button {font-size: 1rem;}
</style>
""",
    unsafe_allow_html=True,
)

APP_DIR = Path(DEFAULT_APP_DIR)
MODEL_OPTIONS = ["RandomForest", "ExtraTrees", "XGBoost", "Voting", "Stacking"]

DEFAULTS = {
    "selected_models": ["RandomForest", "ExtraTrees", "XGBoost"],
    "train_window": 14,
    "market_results": {},
    "watchlist": {},
    "detail_model": None,
    "spread_model": None,
    "backtest_day": None,
    "bge_eval_result": None,
    "bge_eval_day": None,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def business_layout(fig: go.Figure, title: str, ytitle: str = "Price (USD/MWh)", height: int = 430) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        margin=dict(l=18, r=18, t=55, b=18),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        xaxis_title="Delivery hour",
        yaxis_title=ytitle,
        hovermode="x unified",
    )
    fig.update_xaxes(tickformat="%H:%M")
    return fig


def live_target_plot(forecasts: pd.DataFrame, target: str, title: str) -> go.Figure:
    """BGE-style live plot: hourly model lines only. No uncertainty shading."""
    sub = forecasts[forecasts["Target"] == target].copy().sort_values(["Model", "Timestamp"])
    fig = go.Figure()
    for model_name, part in sub.groupby("Model"):
        fig.add_trace(
            go.Scatter(
                x=part["Timestamp"],
                y=part["Prediction"],
                mode="lines+markers",
                name=model_name,
                line=dict(width=2),
                marker=dict(size=5),
            )
        )
    return business_layout(fig, title)


def backtest_all_models_plot(preds: pd.DataFrame, target: str, eval_day: pd.Timestamp, title: str) -> go.Figure:
    d = preds.copy()
    d["run_day"] = pd.to_datetime(d["run_day"]).dt.normalize()
    sub = d[(d["Target"] == target) & (d["run_day"] == pd.Timestamp(eval_day).normalize())].copy()
    sub = sub.sort_values(["Model", "Timestamp"])
    fig = go.Figure()
    actual = sub[["Timestamp", "Actual"]].drop_duplicates("Timestamp").sort_values("Timestamp")
    if not actual.empty:
        fig.add_trace(go.Scatter(x=actual["Timestamp"], y=actual["Actual"], mode="lines+markers", name="Actual", line=dict(width=3), marker=dict(size=6)))
    for model_name, part in sub.groupby("Model"):
        fig.add_trace(go.Scatter(x=part["Timestamp"], y=part["Prediction"], mode="lines", name=model_name, line=dict(width=2)))
    return business_layout(fig, title)


def backtest_model_plot(preds: pd.DataFrame, target: str, eval_day: pd.Timestamp, model_name: str, title: str) -> go.Figure:
    d = preds.copy()
    d["run_day"] = pd.to_datetime(d["run_day"]).dt.normalize()
    sub = d[(d["Target"] == target) & (d["run_day"] == pd.Timestamp(eval_day).normalize()) & (d["Model"] == model_name)].copy().sort_values("Timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["Timestamp"], y=sub["Actual"], mode="lines+markers", name="Actual", line=dict(width=3), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=sub["Timestamp"], y=sub["Prediction"], mode="lines+markers", name=f"Forecast ({model_name})", line=dict(width=2, dash="dash"), marker=dict(size=5)))
    return business_layout(fig, title)


def live_da_rt_overlay(comp: pd.DataFrame, model_name: str) -> go.Figure:
    d = comp.copy().sort_values("Timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Timestamp"], y=d["Forecast DA"], mode="lines+markers", name="Day-Ahead forecast", line=dict(width=3), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=d["Timestamp"], y=d["Forecast RT"], mode="lines+markers", name="Real-Time forecast", line=dict(width=3, dash="dash"), marker=dict(size=6)))
    return business_layout(fig, f"Next-day DA and RT hourly forecasts • {model_name}")


def live_da_rt_bars(comp: pd.DataFrame) -> go.Figure:
    d = comp.copy().sort_values("Timestamp")
    plot = d[["Timestamp", "Forecast DA", "Forecast RT"]].copy()
    plot["Hour"] = pd.to_datetime(plot["Timestamp"]).dt.strftime("%H:%M")
    fig = px.bar(plot, x="Hour", y=["Forecast DA", "Forecast RT"], barmode="group", height=390, title="DA and RT forecast comparison by hour")
    fig.update_layout(template="plotly_white", margin=dict(l=18, r=18, t=55, b=18), xaxis_title="Delivery hour", yaxis_title="Price (USD/MWh)", legend_title_text="")
    return fig


def live_spread_plot(comp: pd.DataFrame) -> go.Figure:
    d = comp.copy().sort_values("Timestamp")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["Timestamp"], y=d["Forecast Spread (DA - RT)"], name="Forecast spread", opacity=0.72))
    fig.add_trace(go.Scatter(x=d["Timestamp"], y=d["Forecast Spread (DA - RT)"], mode="lines+markers", name="Spread profile", line=dict(width=2.5), marker=dict(size=5)))
    fig.add_hline(y=0, line_dash="dash")
    return business_layout(fig, "Hourly DA–RT forecast spread", ytitle="DA − RT (USD/MWh)")


def backtest_spread_frame(preds: pd.DataFrame, eval_day: pd.Timestamp, model_name: str) -> pd.DataFrame:
    d = preds.copy()
    d["run_day"] = pd.to_datetime(d["run_day"]).dt.normalize()
    d = d[(d["run_day"] == pd.Timestamp(eval_day).normalize()) & (d["Model"] == model_name)].copy()
    da = d[d["Target"] == "DA"][["Timestamp", "Actual", "Prediction"]].rename(columns={"Actual": "DA Actual", "Prediction": "DA Forecast"})
    rt = d[d["Target"] == "RT"][["Timestamp", "Actual", "Prediction"]].rename(columns={"Actual": "RT Actual", "Prediction": "RT Forecast"})
    out = da.merge(rt, on="Timestamp", how="inner").sort_values("Timestamp")
    out["Actual Spread (DA - RT)"] = out["DA Actual"] - out["RT Actual"]
    out["Forecast Spread (DA - RT)"] = out["DA Forecast"] - out["RT Forecast"]
    out["Absolute Spread Error"] = (out["Forecast Spread (DA - RT)"] - out["Actual Spread (DA - RT)"]).abs()
    return out


def backtest_spread_plot(spread: pd.DataFrame, model_name: str) -> go.Figure:
    d = spread.copy().sort_values("Timestamp")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["Timestamp"], y=d["Actual Spread (DA - RT)"], name="Actual spread", opacity=0.55))
    fig.add_trace(go.Scatter(x=d["Timestamp"], y=d["Forecast Spread (DA - RT)"], mode="lines+markers", name="Forecast spread", line=dict(width=3), marker=dict(size=5)))
    fig.add_hline(y=0, line_dash="dash")
    return business_layout(fig, f"DA–RT spread diagnostic • {model_name}", ytitle="Spread (USD/MWh)")


def scorecard(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    piv = metrics.pivot(index="Model", columns="Target", values=["Recent MAE", "Recent RMSE", "Recent Bias"]).copy()
    piv.columns = [f"{target} {metric.replace('Recent ', '')}" for metric, target in piv.columns]
    piv = piv.reset_index()
    ordered = ["Model", "DA MAE", "DA RMSE", "DA Bias", "RT MAE", "RT RMSE", "RT Bias"]
    for col in ordered:
        if col not in piv.columns:
            piv[col] = np.nan
    piv = piv[ordered]
    for col in ordered[1:]:
        piv[col] = pd.to_numeric(piv[col], errors="coerce").round(2)
    return piv


def render_glossary():
    with st.expander("Glossary", expanded=False):
        glossary = pd.DataFrame(
            [
                ["DA", "Day-Ahead locational marginal price for the next operating day."],
                ["RT", "Real-Time locational marginal price reflecting shorter-term balancing conditions."],
                ["Runtime", "The information cutoff used to prepare the next-day forecast, aligned to the 10:00 AM EPT workflow."],
                ["Forecast spread", "The forecast Day-Ahead price minus the forecast Real-Time price for the same delivery hour."],
                ["90% interval", "An empirical uncertainty range estimated from recent sequential backtest residuals. It is reported in tables rather than shaded over the hourly forecast charts."],
                ["Zone", "A PJM transmission/load zone such as BGE, PEPCO, PPL, DOM or COMED."],
                ["Hub", "A published PJM pricing hub such as WESTERN HUB or EASTERN HUB."],
            ],
            columns=["Term", "Meaning"],
        )
        st.dataframe(glossary, use_container_width=True, hide_index=True)


def render_faq():
    with st.expander("FAQ", expanded=False):
        items = [
            (
                "What does this platform cover?",
                "The platform provides runtime-constrained Day-Ahead and Real-Time hourly price forecasts across PJM zones and selected published hubs."
            ),
            (
                "What happens when a market is run?",
                "For the selected market, the app retrieves recent PJM Day-Ahead and Real-Time price history, regional load and historical load-forecast data, and representative weather. It builds the runtime-safe feature history, performs a recent rolling backtest on up to the five latest completed forecast days, and then generates the next-day forecast."
            ),
            (
                "How are markets outside BGE evaluated?",
                "The additional PJM zones and hubs use the same generalized runtime-safe forecasting engine and receive a recent rolling backtest inside the app. These recent backtests are useful performance checks but are not equivalent to a long multi-year validation."
            ),
            (
                "How does BGE differ from the other markets?",
                "BGE is the implementation evaluated over the full 2024–2026 study period in the accompanying research paper. The additional PJM zones and hubs are extensions of the same forecasting framework and are evaluated using the app's recent rolling backtest."
            ),
            (
                "What is the difference between fresh mode and the saved market cache?",
                "Fresh mode retrieves current PJM and weather inputs for the selected market. Saved-cache mode reuses the latest prepared market/run-day data when available. A PJM API key is required when fresh data must be retrieved."
            ),
            (
                "Does the workflow respect the 10:00 AM runtime?",
                "Yes. The engine retains the runtime-safe design and restricts completed realized information to the operational cutoff."
            ),
            (
                "Can the DA–RT comparison be interpreted as a bid instruction?",
                "No. The app shows predicted hourly Day-Ahead and Real-Time prices and their spread as forecasting information."
            ),
        ]
        for question, answer in items:
            st.markdown(f"**{question}**")
            st.write(answer)


def render_footer():
    st.markdown(
        """
        <div style="margin-top:2rem;border-top:1px solid #e7eef9;padding-top:0.95rem;text-align:center;color:#5d6b82;font-size:0.95rem;">
            <b>Created by Chibuike C. Ibebuchi</b>
        </div>
        <div style="text-align:center;color:#7a8699;font-size:0.85rem;margin-top:0.35rem;">
            © 2026 Chibuike C. Ibebuchi. All rights reserved.
        </div>
        <div style="text-align:center;color:#7a8699;font-size:0.82rem;margin:0.35rem auto 0 auto;max-width:900px;line-height:1.45;">
            For educational and research purposes only. Forecasts are experimental and are not intended as the basis for bidding, trading, investment, or other financial decisions.
        </div>
        <div style="text-align:center;color:#7a8699;font-size:0.82rem;margin-top:0.35rem;">
            PJM Runtime Forecast v2.0.0
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Sidebar — intentionally mirrors the compact BGE application workflow.
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Settings")
    geography = st.radio("Pricing geography", ["Zone", "Hub"], horizontal=True)
    options = market_options(geography.lower())
    market_ids = list(options.keys())
    default_market = "BGE" if geography == "Zone" and "BGE" in market_ids else market_ids[0]
    market_id = st.selectbox("PJM market", market_ids, index=market_ids.index(default_market), format_func=lambda x: options[x])
    spec = get_market(market_id)

    st.session_state.train_window = st.selectbox("Training window (run-days)", [7, 14, 28], index=[7, 14, 28].index(st.session_state.train_window) if st.session_state.train_window in [7, 14, 28] else 1)
    st.session_state.selected_models = st.multiselect("Models", MODEL_OPTIONS, default=st.session_state.selected_models)

    st.markdown("### Live data")
    data_source = st.radio("Live data source", ["Use saved latest market cache", "Fetch latest PJM/weather data now"], index=1)
    force_refresh = data_source.startswith("Fetch")
    pjm_key = st.text_input("PJM API key", type="password", help="Used only for the current session. It is not written to disk by the app.")


st.markdown(
    '<div class="hero"><h1>⚡ PJM Runtime-Constrained Price Forecasting</h1></div>',
    unsafe_allow_html=True,
)

render_glossary()
render_faq()

eval_tab, live_tab = st.tabs(["📊 Model Evaluation", "🚀 Live Forecast"])


def run_selected_market() -> dict | None:
    if not st.session_state.selected_models:
        st.warning("Select at least one model.")
        return None
    if force_refresh and not pjm_key.strip():
        st.warning("Enter a PJM Data Miner API key to fetch the latest source data.")
        return None
    run_day = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    try:
        with st.spinner(f"Preparing runtime-safe history and forecasting {spec.label}…"):
            result = run_multimarket_forecast(
                APP_DIR,
                pjm_key.strip(),
                spec,
                run_day,
                st.session_state.selected_models,
                int(st.session_state.train_window),
                force_refresh=force_refresh,
            )
        st.session_state.market_results[spec.market_id] = result
        st.session_state.watchlist[spec.market_id] = result["summary"]
        return result
    except Exception as exc:
        st.error(f"The {spec.label} run could not be completed: {type(exc).__name__}: {exc}")
        return None


with eval_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Evaluation")

    result = st.session_state.market_results.get(spec.market_id)
    if result is None:
        st.info("Run the selected market to prepare recent model evaluation and the next-day forecast.")
        if st.button("Prepare model evaluation", type="primary", key=f"prep_eval_{market_id}", use_container_width=True):
            result = run_selected_market()
            if result is not None:
                st.success(f"Model evaluation prepared for {spec.label}.")

    result = st.session_state.market_results.get(spec.market_id)
    if result is not None:
        metrics = result["metrics"].copy()
        preds = result["backtest_predictions"].copy()
        if metrics.empty or preds.empty:
            st.warning("No recent backtest results were available for this market/run.")
        else:
            sc = scorecard(metrics)
            model_list = [m for m in st.session_state.selected_models if m in sc["Model"].tolist()]
            reference_model = "Stacking" if "Stacking" in model_list else ("ExtraTrees" if "ExtraTrees" in model_list else model_list[0])
            refrow = sc.loc[sc["Model"] == reference_model].iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Reference model", reference_model)
            c2.metric("Recent DA MAE", f"{refrow['DA MAE']:.2f}")
            c3.metric("Recent RT MAE", f"{refrow['RT MAE']:.2f}")

            st.markdown("### Model scorecard")
            st.dataframe(sc, use_container_width=True, hide_index=True)

            days = [pd.Timestamp(x) for x in sorted(pd.to_datetime(preds["run_day"]).dt.normalize().unique())]
            if st.session_state.backtest_day not in days:
                st.session_state.backtest_day = days[-1]
            st.session_state.backtest_day = st.selectbox(
                "Evaluation date",
                options=days,
                index=days.index(st.session_state.backtest_day),
                format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
                key=f"backtest_day_{market_id}",
            )
            eval_day = st.session_state.backtest_day

            t1, t2, t3, t4 = st.tabs(["Day-Ahead • all models", "Real-Time • all models", "DA–RT spread", "Detailed model view"])
            with t1:
                st.plotly_chart(backtest_all_models_plot(preds, "DA", eval_day, f"{spec.label} • Day-Ahead prices • all models"), use_container_width=True)
                st.plotly_chart(backtest_model_plot(preds, "DA", eval_day, reference_model, f"Day-Ahead forecast vs actual • {reference_model}"), use_container_width=True)
            with t2:
                st.plotly_chart(backtest_all_models_plot(preds, "RT", eval_day, f"{spec.label} • Real-Time prices • all models"), use_container_width=True)
                st.plotly_chart(backtest_model_plot(preds, "RT", eval_day, reference_model, f"Real-Time forecast vs actual • {reference_model}"), use_container_width=True)
            with t3:
                if st.session_state.spread_model not in model_list:
                    st.session_state.spread_model = reference_model
                spread_model = st.selectbox("Spread diagnostic model", model_list, index=model_list.index(st.session_state.spread_model), key=f"spread_model_{market_id}")
                st.session_state.spread_model = spread_model
                spread = backtest_spread_frame(preds, eval_day, spread_model)
                st.plotly_chart(backtest_spread_plot(spread, spread_model), use_container_width=True)
                st.dataframe(spread.round(2), use_container_width=True, hide_index=True)
            with t4:
                if st.session_state.detail_model not in model_list:
                    st.session_state.detail_model = reference_model
                detail_model = st.selectbox("Detailed model", model_list, index=model_list.index(st.session_state.detail_model), key=f"detail_model_{market_id}")
                st.session_state.detail_model = detail_model
                d1, d2 = st.columns(2)
                with d1:
                    st.plotly_chart(backtest_model_plot(preds, "DA", eval_day, detail_model, f"Day-Ahead • {detail_model}"), use_container_width=True)
                with d2:
                    st.plotly_chart(backtest_model_plot(preds, "RT", eval_day, detail_model, f"Real-Time • {detail_model}"), use_container_width=True)

            if market_id == "BGE":
                with st.expander("Published BGE paper benchmark", expanded=False):
                    try:
                        da_store = load_feature_store(APP_DIR, "DA_total")
                        rt_store = load_feature_store(APP_DIR, "RT_total")
                        paper_days = get_available_run_days(da_store["data"], rt_store["data"])
                        if paper_days:
                            if st.session_state.bge_eval_day not in paper_days:
                                st.session_state.bge_eval_day = paper_days[-1]
                            bday = st.selectbox("Paper-benchmark date", paper_days, index=paper_days.index(st.session_state.bge_eval_day), format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"), key="paper_bge_date")
                            st.session_state.bge_eval_day = bday
                            if st.button("Run published BGE feature-store evaluation", key="run_paper_bge"):
                                paper_models = [m for m in st.session_state.selected_models if m in MODEL_OPTIONS]
                                st.session_state.bge_eval_result = evaluate_models_for_single_day(da_store, rt_store, pd.Timestamp(bday), paper_models, int(st.session_state.train_window))
                            if st.session_state.bge_eval_result is not None:
                                st.dataframe(st.session_state.bge_eval_result["summary"].round(2), use_container_width=True, hide_index=True)
                    except Exception as exc:
                        st.caption(f"The packaged BGE paper benchmark could not be loaded: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)


with live_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live Forecast")
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    forecast_day = today + pd.Timedelta(days=1)
    l1, l2, l3 = st.columns([1, 1, 1.35])
    l1.markdown(f'<div class="metric-box"><div class="metric-label">Forecast date</div><div class="metric-value">{forecast_day.strftime("%Y-%m-%d")}</div></div>', unsafe_allow_html=True)
    l2.markdown(f'<div class="metric-box"><div class="metric-label">Training window</div><div class="metric-value">{st.session_state.train_window} days</div></div>', unsafe_allow_html=True)
    l3.markdown(f'<div class="metric-box"><div class="metric-label">Selected market</div><div class="metric-value" style="font-size:1.08rem;">{spec.label}</div></div>', unsafe_allow_html=True)
    st.caption(f"Representative weather input: {spec.weather_label}. Load area: {spec.load_area}.")

    if st.button("Run live forecast", type="primary", key=f"run_live_{market_id}", use_container_width=True):
        result = run_selected_market()
        if result is not None:
            st.success(f"Live forecast completed for {spec.label}.")

    result = st.session_state.market_results.get(spec.market_id)
    if result is not None:
        s = result["summary"]
        a, b, c, d = st.columns(4)
        a.metric("Mean DA forecast", f"${s['mean_da']:.2f}/MWh")
        b.metric("Mean RT forecast", f"${s['mean_rt']:.2f}/MWh")
        c.metric("Mean DA−RT spread", f"${s['mean_spread']:.2f}/MWh")
        d.metric("Reference model", s["reference_model"])

        forecasts = result["forecasts"].copy()
        comp = result["comparison"].copy()

        x1, x2, x3, x4 = st.tabs(["Day-Ahead forecast", "Real-Time forecast", "DA–RT comparison", "Uncertainty table"])
        with x1:
            st.plotly_chart(live_target_plot(forecasts, "DA", f"{spec.label} • next-day Day-Ahead forecast • all models"), use_container_width=True)
        with x2:
            st.plotly_chart(live_target_plot(forecasts, "RT", f"{spec.label} • next-day Real-Time forecast • all models"), use_container_width=True)
        with x3:
            st.caption(f"DA–RT comparison shown for the reference model: {s['reference_model']}.")
            st.plotly_chart(live_da_rt_overlay(comp, s["reference_model"]), use_container_width=True)
            st.plotly_chart(live_da_rt_bars(comp), use_container_width=True)
            st.plotly_chart(live_spread_plot(comp), use_container_width=True)
            display_cols = ["Timestamp", "Forecast DA", "Forecast RT", "Forecast Spread (DA - RT)", "Spread Sign"]
            st.dataframe(comp[display_cols].round(2), use_container_width=True, hide_index=True)
        with x4:
            uncertainty = forecasts[["Timestamp", "Target", "Model", "Prediction", "Lower90", "Upper90"]].copy()
            uncertainty[["Prediction", "Lower90", "Upper90"]] = uncertainty[["Prediction", "Lower90", "Upper90"]].round(2)
            st.dataframe(uncertainty, use_container_width=True, hide_index=True)
            st.markdown("#### DA–RT reference-model uncertainty")
            comp_unc = comp[["Timestamp", "Forecast DA", "DA Lower90", "DA Upper90", "Forecast RT", "RT Lower90", "RT Upper90", "Forecast Spread (DA - RT)", "Spread Lower90", "Spread Upper90", "Spread Sign"]].copy()
            numeric_cols = [c for c in comp_unc.columns if c not in ["Timestamp", "Spread Sign"]]
            comp_unc[numeric_cols] = comp_unc[numeric_cols].round(2)
            st.dataframe(comp_unc, use_container_width=True, hide_index=True)

        if st.session_state.watchlist:
            with st.expander("Session PJM market comparison", expanded=False):
                rows = []
                for mid, ws in st.session_state.watchlist.items():
                    rows.append({
                        "Market": mid,
                        "Label": ws["market"],
                        "Forecast day": ws["forecast_day"],
                        "Reference model": ws["reference_model"],
                        "Mean DA": ws["mean_da"],
                        "Mean RT": ws["mean_rt"],
                        "Mean DA−RT spread": ws["mean_spread"],
                    })
                watch = pd.DataFrame(rows)
                for col in ["Mean DA", "Mean RT", "Mean DA−RT spread"]:
                    watch[col] = pd.to_numeric(watch[col], errors="coerce").round(2)
                st.dataframe(watch, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)

render_footer()
