from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models import make_model
from pjm_markets import MarketSpec

BASE_URL = "https://api.pjm.com/api/v1"
TIME_COL = "datetime_beginning_ept"
RUNTIME_HOUR = 10
STRICT_CUTOFF_HOUR = 9
LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLL_WINDOWS = [6, 24, 72, 168]
BASE_C = 18.0
ROW_COUNT = 50000
TIMEOUT = 60

# Compact but still information-rich portable feature set. The validated BGE paper
# benchmark remains available separately in the app with its original 711-column design.
BASE_COLS = [
    "total_da", "congestion_da", "loss_da", "total_rt", "congestion_rt", "loss_rt",
    "load_actual_mw", "temp", "rhum", "wspd", "prcp", "pres", "cdh", "hdh",
]


def _fmt(ts) -> str:
    return pd.Timestamp(ts).strftime("%m/%d/%Y %H:%M")


def _date_chunks(start, end, days=7):
    cur = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while cur <= end:
        nxt = min(cur + pd.Timedelta(days=days) - pd.Timedelta(minutes=1), end)
        yield cur, nxt
        cur = nxt + pd.Timedelta(minutes=1)


def _call_pjm(api_key: str, endpoint: str, params: Dict, max_429: int = 6) -> Dict:
    if not api_key.strip():
        raise RuntimeError("A PJM Data Miner API key is required for PJM-wide live forecasting.")
    headers = {"Ocp-Apim-Subscription-Key": api_key.strip()}
    url = f"{BASE_URL}/{endpoint}"
    attempt_429 = 0
    net_try = 0
    while True:
        try:
            r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
            if r.status_code == 429:
                if attempt_429 >= max_429:
                    r.raise_for_status()
                time.sleep(min(30.0, 1.5 * (2 ** attempt_429)))
                attempt_429 += 1
                continue
            r.raise_for_status()
            return r.json()
        except (ConnectionError, Timeout, ChunkedEncodingError):
            if net_try >= 4:
                raise
            net_try += 1
            time.sleep(2.0 * net_try)


def _fetch_all(api_key: str, endpoint: str, params_base: Dict) -> pd.DataFrame:
    rows: List[dict] = []
    start_row = 1
    while True:
        params = dict(params_base)
        params["startRow"] = start_row
        params["rowCount"] = ROW_COUNT
        payload = _call_pjm(api_key, endpoint, params)
        items = payload.get("items", [])
        total = int(payload.get("totalRows", 0) or 0)
        if not items:
            break
        rows.extend(items)
        start_row += len(items)
        if start_row > total:
            break
        time.sleep(0.08)
    return pd.DataFrame(rows)




_PNODE_RECORD_CACHE: Dict[str, pd.DataFrame] = {}


def _normalize_pnode_name(value: str) -> str:
    """Normalize PJM pricing-node names for robust exact-name matching."""
    return " ".join(str(value).upper().replace("_", " " ).split())


def _hub_pnode_records(api_key: str, spec: MarketSpec) -> pd.DataFrame:
    """Resolve the PJM pricing-node records corresponding to a published hub.

    The DA hourly LMP feed accepts pnode_id but not pnode_name as a search filter.
    Hub names are therefore resolved once through the pnode feed and the resulting
    numerical IDs are used for both DA and RT LMP retrieval.
    """
    if spec.kind != "hub" or not spec.pnode_name:
        raise ValueError("Hub pricing-node resolution requires a hub MarketSpec with pnode_name.")

    cache_key = _normalize_pnode_name(spec.pnode_name)
    if cache_key in _PNODE_RECORD_CACHE:
        return _PNODE_RECORD_CACHE[cache_key].copy()

    # First use the name filter; it is supported by PJM's pnode reference feed.
    d = _fetch_all(api_key, "pnode", {"pnode_name": spec.pnode_name, "pnode_subtype": "HUB"})

    # Defensive fallback: retrieve the small HUB reference set and match locally.
    if d.empty:
        d = _fetch_all(api_key, "pnode", {"pnode_subtype": "HUB"})

    if d.empty or "pnode_id" not in d.columns or "pnode_name" not in d.columns:
        raise RuntimeError(f"PJM pricing-node reference data could not resolve hub {spec.label}.")

    target = _normalize_pnode_name(spec.pnode_name)
    names = d["pnode_name"].astype(str).map(_normalize_pnode_name)
    exact = d[names == target].copy()
    if exact.empty:
        # Retain a conservative punctuation-insensitive fallback for names such as
        # AEP-DAYTON HUB while avoiding accidental matches to a different hub.
        compact_target = target.replace("-", " " )
        compact_names = names.str.replace("-", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
        exact = d[compact_names == compact_target].copy()

    if exact.empty:
        returned = sorted(set(d["pnode_name"].astype(str)))[:20]
        raise RuntimeError(
            f"PJM did not return an exact HUB pricing-node record for {spec.label}. "
            f"Returned hub names included: {returned}"
        )

    if "pnode_subtype" in exact.columns:
        hubs = exact[exact["pnode_subtype"].astype(str).str.upper().eq("HUB")].copy()
        if not hubs.empty:
            exact = hubs

    exact["effective_date"] = pd.to_datetime(exact.get("effective_date"), errors="coerce")
    exact["termination_date"] = pd.to_datetime(exact.get("termination_date"), errors="coerce")
    exact["pnode_id"] = pd.to_numeric(exact["pnode_id"], errors="coerce")
    exact = exact.dropna(subset=["pnode_id"]).copy()
    if exact.empty:
        raise RuntimeError(f"PJM returned hub metadata for {spec.label} but no usable pnode_id.")

    _PNODE_RECORD_CACHE[cache_key] = exact.copy()
    return exact


def _hub_pnode_id_for_time(api_key: str, spec: MarketSpec, when) -> int:
    records = _hub_pnode_records(api_key, spec)
    when = pd.Timestamp(when).normalize()
    eff = records["effective_date"]
    term = records["termination_date"]
    active = records[(eff.isna() | (eff <= when)) & (term.isna() | (term >= when))].copy()
    candidates = active if not active.empty else records.copy()
    candidates = candidates.sort_values("effective_date", ascending=False, na_position="last")
    return int(candidates.iloc[0]["pnode_id"])


def _deduplicate_lmp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL], errors="coerce")
    if "row_is_current" in d.columns:
        d["_current"] = d["row_is_current"].astype(str).str.upper().eq("TRUE")
    else:
        d["_current"] = True
    if "version_nbr" in d.columns:
        d["_version"] = pd.to_numeric(d["version_nbr"], errors="coerce").fillna(0)
    else:
        d["_version"] = 0
    sort_cols = [TIME_COL]
    asc = [True]
    if "pnode_id" in d.columns:
        sort_cols.append("pnode_id"); asc.append(True)
    sort_cols += ["_current", "_version"]; asc += [False, False]
    d = d.sort_values(sort_cols, ascending=asc)
    subset = [TIME_COL] + (["pnode_id"] if "pnode_id" in d.columns else ["pnode_name"])
    return d.drop_duplicates(subset=subset, keep="first")


def fetch_market_lmps(api_key: str, spec: MarketSpec, market: str, start, end) -> pd.DataFrame:
    endpoint = "da_hrl_lmps" if market.lower() == "da" else "rt_hrl_lmps"
    parts = []
    for s, e in _date_chunks(start, end, days=7):
        params = {TIME_COL: f"{_fmt(s)} to {_fmt(e)}"}
        if spec.kind == "zone":
            params["zone"] = spec.pjm_zone
        else:
            # DA hourly LMP search does not accept pnode_name. Resolve the
            # published hub name through PJM's pnode reference feed and use
            # the numerical pnode_id, which is supported by both DA and RT.
            params["pnode_id"] = _hub_pnode_id_for_time(api_key, spec, e)
        part = _fetch_all(api_key, endpoint, params)
        if part.empty:
            continue
        if spec.kind == "zone" and "zone" in part.columns:
            part = part[part["zone"].astype(str).str.upper() == str(spec.pjm_zone).upper()].copy()
        if spec.kind == "hub" and "pnode_name" in part.columns:
            target = str(spec.pnode_name).upper()
            exact = part[part["pnode_name"].astype(str).str.upper() == target]
            if not exact.empty:
                part = exact.copy()
            else:
                part = part[part["pnode_name"].astype(str).str.upper().str.contains(target, regex=False)].copy()
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    raw = _deduplicate_lmp(pd.concat(parts, ignore_index=True))
    if market.lower() == "da":
        cols = ("system_energy_price_da", "congestion_price_da", "marginal_loss_price_da", "total_lmp_da")
        rename = {"energy": "energy_da", "congestion": "congestion_da", "loss": "loss_da", "total": "total_da", "n_nodes": "n_nodes_da"}
    else:
        cols = ("system_energy_price_rt", "congestion_price_rt", "marginal_loss_price_rt", "total_lmp_rt")
        rename = {"energy": "energy_rt", "congestion": "congestion_rt", "loss": "loss_rt", "total": "total_rt", "n_nodes": "n_nodes_rt"}
    energy, congestion, loss, total = cols
    required = [energy, congestion, loss, total]
    if any(c not in raw.columns for c in required):
        raise RuntimeError(f"PJM {endpoint} response did not contain the expected LMP fields for {spec.label}.")
    for c in required:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    if "pnode_id" not in raw.columns:
        raw["pnode_id"] = raw.get("pnode_name", "market")
    out = (
        raw.groupby(TIME_COL, as_index=False)
        .agg(energy=(energy, "mean"), congestion=(congestion, "mean"), loss=(loss, "mean"), total=(total, "mean"), n_nodes=("pnode_id", "nunique"))
        .rename(columns=rename)
        .sort_values(TIME_COL)
    )
    return out


def fetch_actual_load(api_key: str, load_area: str, start, end) -> pd.DataFrame:
    parts = []
    for s, e in _date_chunks(start, end, days=14):
        part = _fetch_all(api_key, "hrl_load_prelim", {TIME_COL: f"{_fmt(s)} to {_fmt(e)}", "load_area": load_area})
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    if "load_area" in d.columns:
        d = d[d["load_area"].astype(str).str.upper() == load_area.upper()].copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL], errors="coerce")
    d["load_actual_mw"] = pd.to_numeric(d["prelim_load_avg_hourly"], errors="coerce")
    return d[[TIME_COL, "load_actual_mw"]].dropna(subset=[TIME_COL]).sort_values(TIME_COL).drop_duplicates(TIME_COL, keep="last")


def fetch_load_forecast_history(api_key: str, forecast_area: str, start, end) -> pd.DataFrame:
    parts = []
    for s, e in _date_chunks(start, end, days=7):
        part = _fetch_all(api_key, "load_frcstd_hist", {"evaluated_at_ept": f"{_fmt(s)} to {_fmt(e)}", "forecast_area": forecast_area})
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    if "forecast_area" in d.columns:
        d = d[d["forecast_area"].astype(str).str.upper() == forecast_area.upper()].copy()
    for c in ["evaluated_at_ept", "forecast_hour_beginning_ept"]:
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d["forecast_load_mw"] = pd.to_numeric(d["forecast_load_mw"], errors="coerce")
    return d.dropna(subset=["evaluated_at_ept", "forecast_hour_beginning_ept"]).sort_values(["forecast_hour_beginning_ept", "evaluated_at_ept"])


def fetch_live_load_forecast(api_key: str, forecast_area: str, run_day) -> pd.DataFrame:
    d = _fetch_all(api_key, "load_frcstd_7_day", {"forecast_area": forecast_area})
    if d.empty:
        return d
    rename = {"evaluated_at_datetime_ept": "evaluated_at_ept", "forecast_datetime_beginning_ept": "forecast_hour_beginning_ept"}
    d = d.rename(columns=rename)
    for c in ["evaluated_at_ept", "forecast_hour_beginning_ept"]:
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d["forecast_load_mw"] = pd.to_numeric(d["forecast_load_mw"], errors="coerce")
    runtime = pd.Timestamp(run_day).normalize() + pd.Timedelta(hours=RUNTIME_HOUR)
    target = pd.date_range(pd.Timestamp(run_day).normalize() + pd.Timedelta(days=1), periods=24, freq="h")
    d = d[(d["evaluated_at_ept"] <= runtime) & (d["forecast_hour_beginning_ept"].isin(target))].copy()
    if d.empty:
        return d
    return d.sort_values(["forecast_hour_beginning_ept", "evaluated_at_ept"]).drop_duplicates("forecast_hour_beginning_ept", keep="last")


def fetch_weather(latitude: float, longitude: float, start, end) -> pd.DataFrame:
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    cols = ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure", "wind_speed_10m"]
    frames = []

    # Archive data are stable for older history.
    archive_end = min(end, pd.Timestamp.now().normalize() - pd.Timedelta(days=5))
    if start <= archive_end:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": archive_end.strftime("%Y-%m-%d"),
                "hourly": ",".join(cols),
                "timezone": "America/New_York",
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        payload = r.json().get("hourly", {})
        if payload.get("time"):
            frames.append(_weather_payload_to_df(payload))

    # Forecast API supplies recent observed/past-day weather and the current day.
    recent_start = max(start, pd.Timestamp.now().normalize() - pd.Timedelta(days=14))
    if recent_start <= end:
        past_days = int(max(0, (pd.Timestamp.now().normalize() - recent_start).days))
        past_days = min(14, past_days)
        forecast_days = max(1, min(3, int((end - pd.Timestamp.now().normalize()).days) + 1))
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": ",".join(cols),
                "timezone": "America/New_York",
                "past_days": past_days,
                "forecast_days": forecast_days,
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        payload = r.json().get("hourly", {})
        if payload.get("time"):
            frames.append(_weather_payload_to_df(payload))

    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True).sort_values(TIME_COL).drop_duplicates(TIME_COL, keep="last")
    start_h = pd.Timestamp(start)
    end_h = pd.Timestamp(end) + pd.Timedelta(hours=23)
    return d[(d[TIME_COL] >= start_h) & (d[TIME_COL] <= end_h)].reset_index(drop=True)


def _weather_payload_to_df(payload: Dict) -> pd.DataFrame:
    d = pd.DataFrame({
        TIME_COL: pd.to_datetime(payload["time"], errors="coerce"),
        "temp": payload.get("temperature_2m"),
        "rhum": payload.get("relative_humidity_2m"),
        "prcp": payload.get("precipitation"),
        "pres": payload.get("surface_pressure"),
        "wspd": payload.get("wind_speed_10m"),
    })
    for c in ["temp", "rhum", "prcp", "pres", "wspd"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["cdh"] = (d["temp"] - BASE_C).clip(lower=0)
    d["hdh"] = (BASE_C - d["temp"]).clip(lower=0)
    return d.dropna(subset=[TIME_COL])


def merge_hourly(parts: Iterable[pd.DataFrame]) -> pd.DataFrame:
    usable = [p.copy() for p in parts if isinstance(p, pd.DataFrame) and not p.empty]
    if not usable:
        return pd.DataFrame(columns=[TIME_COL])
    all_times = pd.concat([p[[TIME_COL]] for p in usable], ignore_index=True).drop_duplicates().sort_values(TIME_COL)
    out = all_times.copy()
    for p in usable:
        p[TIME_COL] = pd.to_datetime(p[TIME_COL], errors="coerce")
        p = p.drop_duplicates(TIME_COL, keep="last")
        out = out.merge(p, on=TIME_COL, how="left")
    return out.sort_values(TIME_COL).reset_index(drop=True)


def _last_row_leq(indexed: pd.DataFrame, t: pd.Timestamp, cols: List[str]) -> pd.Series:
    sub = indexed.loc[:t, cols]
    if sub.empty:
        return pd.Series([np.nan] * len(cols), index=cols)
    return sub.iloc[-1]


def snapshot_features(indexed: pd.DataFrame, cutoff: pd.Timestamp, base_cols: List[str]) -> Dict[str, float]:
    feat: Dict[str, float] = {}
    last = indexed.loc[cutoff, base_cols] if cutoff in indexed.index else _last_row_leq(indexed, cutoff, base_cols)
    for c in base_cols:
        feat[f"{c}_at_runtime"] = float(last[c]) if pd.notna(last[c]) else np.nan
    for lag in LAGS:
        t = cutoff - pd.Timedelta(hours=lag)
        row = indexed.loc[t, base_cols] if t in indexed.index else _last_row_leq(indexed, t, base_cols)
        for c in base_cols:
            v = float(row[c]) if pd.notna(row[c]) else np.nan
            feat[f"{c}_lag{lag}h"] = v
            a = feat[f"{c}_at_runtime"]
            feat[f"{c}_delta{lag}h"] = a - v if np.isfinite(a) and np.isfinite(v) else np.nan
    for w in ROLL_WINDOWS:
        win = indexed.loc[cutoff - pd.Timedelta(hours=w - 1):cutoff, base_cols]
        for c in base_cols:
            s = pd.to_numeric(win[c], errors="coerce")
            feat[f"{c}_rollmean{w}h"] = float(s.mean()) if s.notna().any() else np.nan
            feat[f"{c}_rollstd{w}h"] = float(s.std()) if s.notna().any() else np.nan
            feat[f"{c}_rollmin{w}h"] = float(s.min()) if s.notna().any() else np.nan
            feat[f"{c}_rollmax{w}h"] = float(s.max()) if s.notna().any() else np.nan
    return feat


def _add_time_features(d: pd.DataFrame, col: str) -> pd.DataFrame:
    out = d.copy()
    t = pd.to_datetime(out[col])
    hour = t.dt.hour
    dow = t.dt.dayofweek
    doy = t.dt.dayofyear
    month = t.dt.month
    out["is_weekend"] = (dow >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return out


def _forecast_vector(hist: pd.DataFrame, run_day, target_index: pd.DatetimeIndex) -> np.ndarray:
    if hist is None or hist.empty:
        return np.full(len(target_index), np.nan)
    runtime = pd.Timestamp(run_day).normalize() + pd.Timedelta(hours=RUNTIME_HOUR)
    d = hist[(hist["evaluated_at_ept"] <= runtime) & (hist["forecast_hour_beginning_ept"].isin(target_index))].copy()
    if d.empty:
        return np.full(len(target_index), np.nan)
    d = d.sort_values(["forecast_hour_beginning_ept", "evaluated_at_ept"]).drop_duplicates("forecast_hour_beginning_ept", keep="last")
    return d.set_index("forecast_hour_beginning_ept")["forecast_load_mw"].reindex(target_index).to_numpy(dtype=float)


def build_run_block(merged: pd.DataFrame, forecast_hist: pd.DataFrame, run_day, require_targets: bool) -> pd.DataFrame:
    d = merged.copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL])
    indexed = d.sort_values(TIME_COL).drop_duplicates(TIME_COL, keep="last").set_index(TIME_COL)
    run_day = pd.Timestamp(run_day).normalize()
    runtime = run_day + pd.Timedelta(hours=RUNTIME_HOUR)
    cutoff = run_day + pd.Timedelta(hours=STRICT_CUTOFF_HOUR)
    target_index = pd.date_range(run_day + pd.Timedelta(days=1), periods=24, freq="h")
    base_cols = [c for c in BASE_COLS if c in indexed.columns]
    snap = snapshot_features(indexed, cutoff, base_cols)
    block = pd.DataFrame([snap] * 24)
    block["run_day"] = run_day
    block["run_time"] = runtime
    block["target_timestamp"] = target_index
    block["load_forecast_mw"] = _forecast_vector(forecast_hist, run_day, target_index)
    block["load_forecast_present"] = np.isfinite(block["load_forecast_mw"]).astype(int)
    block = _add_time_features(block, "target_timestamp")
    for target in ["total_da", "total_rt"]:
        if target in indexed.columns:
            block[target] = pd.to_numeric(indexed[target], errors="coerce").reindex(target_index).to_numpy(dtype=float)
        else:
            block[target] = np.nan
    if require_targets:
        block = block[np.isfinite(block["total_da"]) & np.isfinite(block["total_rt"])].copy()
    return block


def build_feature_dataset(merged: pd.DataFrame, forecast_hist: pd.DataFrame, run_days: Iterable[pd.Timestamp]) -> pd.DataFrame:
    blocks = []
    for d in run_days:
        b = build_run_block(merged, forecast_hist, d, require_targets=True)
        if len(b) >= 20:
            blocks.append(b)
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True).sort_values(["run_day", "target_timestamp"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"run_day", "run_time", "target_timestamp", "total_da", "total_rt"}
    return [c for c in df.columns if c not in excluded]


def _rmse(y, p) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def _finite_quantile(residuals: np.ndarray, coverage: float = 0.90) -> float:
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.nan
    level = min(1.0, math.ceil((len(r) + 1) * coverage) / len(r))
    try:
        return float(np.quantile(r, level, method="higher"))
    except TypeError:
        return float(np.quantile(r, level, interpolation="higher"))


def rolling_backtest(dataset: pd.DataFrame, models: List[str], train_window_days: int, backtest_days: int = 5):
    if dataset.empty:
        return pd.DataFrame(), {}, pd.DataFrame()
    days = sorted(pd.to_datetime(dataset["run_day"]).dt.normalize().unique())
    if len(days) < train_window_days + 2:
        raise RuntimeError("Not enough completed run-days were available for the requested training window.")
    bt_days = [pd.Timestamp(x) for x in days[-min(backtest_days, max(2, len(days) - train_window_days)):]]
    feats = feature_columns(dataset)
    pred_rows = []
    for eval_day in bt_days:
        prior_days = [pd.Timestamp(x) for x in days if pd.Timestamp(x) < eval_day]
        use_days = prior_days[-train_window_days:]
        train = dataset[pd.to_datetime(dataset["run_day"]).dt.normalize().isin(use_days)].copy()
        test = dataset[pd.to_datetime(dataset["run_day"]).dt.normalize() == eval_day].copy()
        if train.empty or test.empty:
            continue
        Xtr = train[feats].apply(pd.to_numeric, errors="coerce")
        Xte = test[feats].apply(pd.to_numeric, errors="coerce")
        for target, label in [("total_da", "DA"), ("total_rt", "RT")]:
            ytr = pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
            yte = pd.to_numeric(test[target], errors="coerce").to_numpy(dtype=float)
            for model_name in models:
                model = make_model(model_name)
                model.fit(Xtr, ytr)
                p = model.predict(Xte)
                part = pd.DataFrame({
                    "run_day": eval_day,
                    "Timestamp": pd.to_datetime(test["target_timestamp"]).to_numpy(),
                    "Target": label,
                    "Model": model_name,
                    "Actual": yte,
                    "Prediction": p,
                })
                part["Error"] = part["Prediction"] - part["Actual"]
                part["Absolute Error"] = part["Error"].abs()
                pred_rows.append(part)
    if not pred_rows:
        return pd.DataFrame(), {}, pd.DataFrame()
    preds = pd.concat(pred_rows, ignore_index=True)
    metrics = []
    intervals = {}
    sorted_bt = sorted(preds["run_day"].unique())
    split = max(1, len(sorted_bt) - 2)
    cal_days = set(sorted_bt[:split])
    score_days = set(sorted_bt[split:])
    for (target, model_name), g in preds.groupby(["Target", "Model"]):
        y = g["Actual"].to_numpy(dtype=float)
        p = g["Prediction"].to_numpy(dtype=float)
        cal = g[g["run_day"].isin(cal_days)]["Absolute Error"].to_numpy(dtype=float)
        if len(cal) < 12:
            cal = g["Absolute Error"].to_numpy(dtype=float)
        q = _finite_quantile(cal, 0.90)
        scored = g[g["run_day"].isin(score_days)].copy()
        coverage = np.nan
        if len(scored) and np.isfinite(q):
            coverage = float(np.mean((scored["Actual"] >= scored["Prediction"] - q) & (scored["Actual"] <= scored["Prediction"] + q)))
        metrics.append({
            "Target": target,
            "Model": model_name,
            "Recent MAE": float(mean_absolute_error(y, p)),
            "Recent RMSE": _rmse(y, p),
            "Recent Bias": float(np.mean(p - y)),
            "Recent 90% Coverage": coverage,
            "Interval Half-width": q,
            "Backtest Days": int(g["run_day"].nunique()),
        })
        intervals[(target, model_name)] = q
    return pd.DataFrame(metrics), intervals, preds


def fit_live_forecast(dataset: pd.DataFrame, live_block: pd.DataFrame, models: List[str], train_window_days: int, intervals: Dict[Tuple[str, str], float]):
    days = sorted(pd.to_datetime(dataset["run_day"]).dt.normalize().unique())
    use_days = [pd.Timestamp(x) for x in days[-train_window_days:]]
    train = dataset[pd.to_datetime(dataset["run_day"]).dt.normalize().isin(use_days)].copy()
    feats = feature_columns(dataset)
    Xtr = train[feats].apply(pd.to_numeric, errors="coerce")
    Xlive = live_block.reindex(columns=feats).apply(pd.to_numeric, errors="coerce")
    rows = []
    for target, label in [("total_da", "DA"), ("total_rt", "RT")]:
        ytr = pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
        for model_name in models:
            model = make_model(model_name)
            model.fit(Xtr, ytr)
            p = model.predict(Xlive)
            q = intervals.get((label, model_name), np.nan)
            rows.append(pd.DataFrame({
                "Timestamp": pd.to_datetime(live_block["target_timestamp"]).to_numpy(),
                "Target": label,
                "Model": model_name,
                "Prediction": p,
                "Lower90": p - q if np.isfinite(q) else np.nan,
                "Upper90": p + q if np.isfinite(q) else np.nan,
            }))
    return pd.concat(rows, ignore_index=True)


def prepare_market_inputs(app_dir: Path, api_key: str, spec: MarketSpec, run_day, train_window_days: int, force_refresh: bool = False):
    run_day = pd.Timestamp(run_day).normalize()
    history_days = max(28, int(train_window_days) + 16)
    start = run_day - pd.Timedelta(days=history_days + 8)
    end = run_day + pd.Timedelta(hours=STRICT_CUTOFF_HOUR)
    cache_dir = Path(app_dir) / "10_multimarket_cache" / spec.market_id / run_day.strftime("%Y-%m-%d")
    cache_dir.mkdir(parents=True, exist_ok=True)
    merged_path = cache_dir / "merged.parquet"
    fh_path = cache_dir / "forecast_history.parquet"
    fl_path = cache_dir / "forecast_live.parquet"
    meta_path = cache_dir / "source_meta.json"

    if not force_refresh and merged_path.exists() and fh_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("history_days", 0) >= history_days:
            merged = pd.read_parquet(merged_path)
            fh = pd.read_parquet(fh_path)
            fl = pd.read_parquet(fl_path) if fl_path.exists() else pd.DataFrame()
            return merged, fh, fl, meta

    da = fetch_market_lmps(api_key, spec, "da", start, end)
    rt = fetch_market_lmps(api_key, spec, "rt", start, end)
    if da.empty or rt.empty:
        raise RuntimeError(f"PJM returned insufficient DA/RT hourly price history for {spec.label}.")
    load = fetch_actual_load(api_key, spec.load_area, start, end)
    try:
        wx = fetch_weather(spec.latitude, spec.longitude, start, run_day)
    except Exception:
        wx = pd.DataFrame()
    fh = fetch_load_forecast_history(api_key, spec.forecast_area_hist, start, run_day + pd.Timedelta(hours=RUNTIME_HOUR))
    try:
        fl = fetch_live_load_forecast(api_key, spec.forecast_area_live, run_day)
    except Exception:
        fl = pd.DataFrame()
    merged = merge_hourly([da, rt, load, wx])
    merged.to_parquet(merged_path, index=False)
    fh.to_parquet(fh_path, index=False)
    if not fl.empty:
        fl.to_parquet(fl_path, index=False)
    elif fl_path.exists():
        fl_path.unlink()
    meta = {
        "market_id": spec.market_id,
        "market_label": spec.label,
        "kind": spec.kind,
        "run_day": run_day.strftime("%Y-%m-%d"),
        "history_days": history_days,
        "rows": {"da": len(da), "rt": len(rt), "load": len(load), "weather": len(wx), "forecast_history": len(fh), "forecast_live": len(fl), "merged": len(merged)},
        "weather_reference": spec.weather_label,
        "load_area": spec.load_area,
        "forecast_area_hist": spec.forecast_area_hist,
        "forecast_area_live": spec.forecast_area_live,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return merged, fh, fl, meta


def run_multimarket_forecast(app_dir: Path, api_key: str, spec: MarketSpec, run_day, models: List[str], train_window_days: int = 14, force_refresh: bool = False):
    merged, fh, fl, source_meta = prepare_market_inputs(app_dir, api_key, spec, run_day, train_window_days, force_refresh)
    run_day = pd.Timestamp(run_day).normalize()
    # Completed historical run-days must have their full next-day DA and RT targets observed.
    candidate_days = pd.date_range(run_day - pd.Timedelta(days=train_window_days + 10), run_day - pd.Timedelta(days=2), freq="D")
    dataset = build_feature_dataset(merged, fh, candidate_days)
    if dataset.empty:
        raise RuntimeError("The on-demand feature builder could not construct enough completed historical run-days for this market.")

    # For the live block, replace historical forecast data with the runtime-safe current 7-day feed rows.
    live_hist = fl.rename(columns={}) if isinstance(fl, pd.DataFrame) else pd.DataFrame()
    if not live_hist.empty:
        live_hist = live_hist[["evaluated_at_ept", "forecast_hour_beginning_ept", "forecast_load_mw"]].copy()
    live_block = build_run_block(merged, live_hist, run_day, require_targets=False)

    metrics, intervals, backtest_preds = rolling_backtest(dataset, models, train_window_days=train_window_days, backtest_days=5)
    forecasts = fit_live_forecast(dataset, live_block, models, train_window_days=train_window_days, intervals=intervals)

    ref = "ExtraTrees" if "ExtraTrees" in models else models[0]
    da = forecasts[(forecasts["Target"] == "DA") & (forecasts["Model"] == ref)].copy()
    rt = forecasts[(forecasts["Target"] == "RT") & (forecasts["Model"] == ref)].copy()
    comp = da[["Timestamp", "Prediction", "Lower90", "Upper90"]].rename(columns={"Prediction": "Forecast DA", "Lower90": "DA Lower90", "Upper90": "DA Upper90"}).merge(
        rt[["Timestamp", "Prediction", "Lower90", "Upper90"]].rename(columns={"Prediction": "Forecast RT", "Lower90": "RT Lower90", "Upper90": "RT Upper90"}),
        on="Timestamp", how="inner"
    )
    comp["Forecast Spread (DA - RT)"] = comp["Forecast DA"] - comp["Forecast RT"]
    comp["Spread Lower90"] = comp["DA Lower90"] - comp["RT Upper90"]
    comp["Spread Upper90"] = comp["DA Upper90"] - comp["RT Lower90"]
    comp["Spread Sign"] = np.select(
        [comp["Spread Lower90"] > 0, comp["Spread Upper90"] < 0],
        ["Positive range", "Negative range"],
        default="Uncertain sign",
    )
    summary = {
        "market": spec.label,
        "reference_model": ref,
        "forecast_day": (run_day + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "mean_da": float(comp["Forecast DA"].mean()),
        "mean_rt": float(comp["Forecast RT"].mean()),
        "mean_spread": float(comp["Forecast Spread (DA - RT)"].mean()),
        "positive_range_hours": int((comp["Spread Sign"] == "Positive range").sum()),
        "negative_range_hours": int((comp["Spread Sign"] == "Negative range").sum()),
        "uncertain_hours": int((comp["Spread Sign"] == "Uncertain sign").sum()),
    }
    return {
        "source_meta": source_meta,
        "feature_count": len(feature_columns(dataset)),
        "completed_run_days": int(pd.to_datetime(dataset["run_day"]).dt.normalize().nunique()),
        "metrics": metrics,
        "backtest_predictions": backtest_preds,
        "forecasts": forecasts,
        "comparison": comp,
        "summary": summary,
    }
