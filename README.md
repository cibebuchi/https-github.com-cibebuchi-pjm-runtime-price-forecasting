# PJM Runtime Forecast v2

**Runtime-constrained Day-Ahead and Real-Time electricity price forecasting across PJM zones and selected hubs.**

Created by **Chibuike C. Ibebuchi**.

## What v2 adds

Version `2.0.0` generalizes the original BGE research application into a market-parameterized PJM platform. It currently supports the PJM transmission zones defined in `pjm_markets.py` plus selected published PJM hubs. The user selects a market, enters a PJM Data Miner API key, and the app retrieves recent market history, regional load, historical load forecasts, and representative local weather before building a runtime-safe rolling training set.

The workflow keeps a **10:00 AM EPT runtime** and restricts realized hourly inputs to completed observations through **09:00 EPT**. It then:

1. downloads or reuses cached source data for the selected PJM market;
2. creates prespecified lag, difference, rolling-window, calendar, weather, load, and load-forecast predictors;
3. performs recent sequential historical backtests;
4. estimates an empirical 90% uncertainty interval from prior backtest residuals;
5. retrains the selected models on the most recent completed run-days; and
6. produces next-day DA, RT, and DA-minus-RT spread forecasts.

The **BGE Paper Benchmark** is kept as a separate tab and uses the packaged BGE feature stores from the research application. This separation is intentional: BGE is the paper-validated implementation, while the PJM-wide engine is an engineering generalization that should be prospectively validated market by market before performance claims are made.

## Supported market types

- PJM transmission zones such as BGE, PEPCO, PPL, PSEG, DOM, AEP, COMED, ATSI, DAY, DPL, DUQ and others listed in `pjm_markets.py`.
- Selected PJM hubs including WESTERN HUB, EASTERN HUB, DOMINION HUB, CHICAGO HUB, OHIO HUB, NEW JERSEY HUB, AEP GEN HUB, AEP-DAYTON HUB and ATSI GEN HUB.

PJM identifiers are parameterized in `pjm_markets.py`, so additional zones, hubs, or pricing nodes can be added without rewriting the forecasting engine.

## Run locally

Python 3.12 is recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The PJM-wide forecast requires a PJM Data Miner API key entered in the Streamlit sidebar. The key is passed only to the current run and is not written to application files.

## Streamlit Community Cloud

Use:

- Main file: `app.py`
- Python: `3.12`
- Requirements: `requirements.txt`

No API key needs to be stored as a Streamlit secret because users can supply their own PJM key in the password field.

## Project structure

```text
app.py                     # PJM-wide Streamlit interface
multi_market_engine.py     # market-general PJM fetch, feature, backtest, interval, forecast engine
pjm_markets.py             # zone/hub metadata and load/weather mappings
models.py                  # model factory + packaged BGE evaluation functions
utils.py                   # packaged BGE research utilities
legacy_bge_app.py          # previous BGE-only interface retained for reference
01_raw_parquet/ ...        # packaged BGE research data
02_curated/ ...
03_features/ ...
05_eval/ ...
06_live/ ...
07_proxy_mae/ ...
08_node_raw/ ...
09_node_features/ ...
10_multimarket_cache/      # created at runtime; ignored by Git
```

## Interpretation

The DA-minus-RT view is a **forecast spread diagnostic**. The app reports the predicted spread together with a conservative uncertainty range. It does not convert the forecast into a buy/sell or bidding instruction.

Representative weather coordinates are predictor locations and do not define PJM transmission-zone boundaries. Zone LMPs in the generalized engine are unweighted composites of the hourly pricing-node rows returned by PJM for the selected transmission-zone filter. Hub forecasts use the named PJM hub pricing node.

## Research status

The original BGE implementation accompanies the manuscript *Runtime-Constrained Day-Ahead and Real-Time Electricity Price Forecasting in PJM's BGE Zone: Stacked Ensembles and Residual Diffusion*. The multi-market v2 engine is deliberately labeled exploratory until its fixed pipeline is evaluated prospectively across the additional PJM markets.


## Version 2.0.0

Version 2.0.0 is the first consolidated PJM-wide release. It preserves the clean BGE-style hourly Day-Ahead and Real-Time visualization, supports PJM zones and selected hubs, resolves hub pricing through PJM pnode identifiers, keeps numerical uncertainty in dedicated tables, and moves methodological/validation detail into the in-app FAQ so the main forecasting interface remains compact.

### Final interface polish

The final v2.0.0 release keeps the main interface focused on forecasting. Detailed methodological and validation explanations remain in the FAQ, while the footer states the educational/research-use limitation and clarifies that forecasts are not intended as the basis for bidding, trading, investment, or other financial decisions.
