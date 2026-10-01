from __future__ import annotations

"""Metadata for the PJM-wide experimental forecasting engine.

Coordinates are representative points used only to obtain local weather inputs. They are
not intended to define transmission-zone boundaries. PJM market identifiers and load-
forecast mappings are kept separate so the application can evolve without changing the
forecasting code.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass(frozen=True)
class MarketSpec:
    market_id: str
    label: str
    kind: str  # "zone" or "hub"
    pjm_zone: str | None
    pnode_name: str | None
    latitude: float
    longitude: float
    weather_label: str
    load_area: str
    forecast_area_hist: str
    forecast_area_live: str
    region: str

    def to_dict(self) -> dict:
        return asdict(self)


# Historical load_frcstd_hist exposes a smaller set of regional forecast areas than
# load_frcstd_7_day. Mid-Atlantic zones therefore share MIDATL historical load forecasts,
# while the live feed uses the matching MID_ATLANTIC_REGION aggregate so the scale is
# consistent between training and live prediction.
_MARKETS: List[MarketSpec] = [
    MarketSpec("AECO", "AECO — Atlantic City Electric", "zone", "AECO", None, 39.36, -74.42, "Atlantic City, NJ", "MIDATL", "MIDATL", "MID_ATLANTIC_REGION", "Mid-Atlantic"),
    MarketSpec("AEP", "AEP — American Electric Power", "zone", "AEP", None, 39.96, -83.00, "Columbus, OH", "AEP", "AEP", "AEP", "Western"),
    MarketSpec("APS", "APS — Allegheny Power", "zone", "APS", None, 39.63, -79.96, "Morgantown, WV", "AP", "APS", "AP", "Western"),
    MarketSpec("ATSI", "ATSI — American Transmission Systems", "zone", "ATSI", None, 41.08, -81.52, "Akron, OH", "ATSI", "ATSI", "ATSI", "Western"),
    MarketSpec("BGE", "BGE — Baltimore Gas and Electric", "zone", "BGE", None, 39.29, -76.61, "Baltimore, MD", "MIDATL", "MIDATL", "MID_ATLANTIC_REGION", "Mid-Atlantic"),
    MarketSpec("COMED", "COMED — Commonwealth Edison", "zone", "COMED", None, 41.88, -87.63, "Chicago, IL", "NI", "COMED", "COMED", "Western"),
    MarketSpec("DAY", "DAY — Dayton", "zone", "DAY", None, 39.76, -84.19, "Dayton, OH", "DAY", "DAY", "DAYTON", "Western"),
    MarketSpec("DEOK", "DEOK — Duke Energy Ohio/Kentucky", "zone", "DEOK", None, 39.10, -84.51, "Cincinnati, OH", "DEOK", "RTO", "RTO_COMBINED", "Western"),
    MarketSpec("DOM", "DOM — Dominion", "zone", "DOM", None, 37.54, -77.44, "Richmond, VA", "DOM", "DOM", "DOMINION", "Southern"),
    MarketSpec("DPL", "DPL — Delmarva Power & Light", "zone", "DPL", None, 39.74, -75.55, "Wilmington, DE", "MIDATL", "MIDATL", "MID_ATLANTIC_REGION", "Mid-Atlantic"),
    MarketSpec("DUQ", "DUQ — Duquesne Light", "zone", "DUQ", None, 40.44, -79.99, "Pittsburgh, PA", "DUQ", "DUQ", "DUQUESNE", "Western"),
    MarketSpec("EKPC", "EKPC — East Kentucky Power Cooperative", "zone", "EKPC", None, 38.04, -84.50, "Lexington, KY", "EKPC", "RTO", "RTO_COMBINED", "Western"),
    MarketSpec("JCPL", "JCPL — Jersey Central Power & Light", "zone", "JCPL", None, 40.80, -74.48, "Morristown, NJ", "MIDATL", "MIDATL", "MID_ATLANTIC_REGION", "Mid-Atlantic"),
    MarketSpec("METED", "METED — Metropolitan Edison", "zone", "METED", None, 40.34, -75.93, "Reading, PA", "MIDATL", "MIDATL", "MID_ATLANTIC_REGION", "Mid-Atlantic"),
    MarketSpec("PECO", "PECO — PECO Energy", "zone", "PECO", None, 39.95, -75.16, "Philadelphia, PA", "MIDATL", "MIDATL", "MID_ATLANTIC_REGION", "Mid-Atlantic"),
    MarketSpec("PENELEC", "PENELEC — Pennsylvania Electric", "zone", "PENELEC", None, 40.52, -78.39, "Altoona, PA", "MIDATL", "MIDATL", "MID_ATLANTIC_REGION", "Mid-Atlantic"),
    MarketSpec("PEPCO", "PEPCO — Potomac Electric Power", "zone", "PEPCO", None, 38.91, -77.04, "Washington, DC", "MIDATL", "MIDATL", "MID_ATLANTIC_REGION", "Mid-Atlantic"),
    MarketSpec("PPL", "PPL — PPL Electric Utilities", "zone", "PPL", None, 40.60, -75.49, "Allentown, PA", "MIDATL", "MIDATL", "MID_ATLANTIC_REGION", "Mid-Atlantic"),
    MarketSpec("PSEG", "PSEG — Public Service Electric & Gas", "zone", "PSEG", None, 40.74, -74.17, "Newark, NJ", "MIDATL", "MIDATL", "MID_ATLANTIC_REGION", "Mid-Atlantic"),
    MarketSpec("RECO", "RECO — Rockland Electric", "zone", "RECO", None, 41.15, -74.03, "Rockland County, NY", "MIDATL", "MIDATL", "MID_ATLANTIC_REGION", "Mid-Atlantic"),

    # Selected published PJM hubs. pnode_name is used as the Data Miner filter.
    MarketSpec("AEP_GEN_HUB", "AEP GEN HUB", "hub", None, "AEP GEN HUB", 39.96, -83.00, "Columbus, OH", "AEP", "AEP", "AEP", "Western"),
    MarketSpec("AEP_DAYTON_HUB", "AEP-DAYTON HUB", "hub", None, "AEP-DAYTON HUB", 39.76, -84.19, "Dayton, OH", "DAY", "DAY", "DAYTON", "Western"),
    MarketSpec("ATSI_GEN_HUB", "ATSI GEN HUB", "hub", None, "ATSI GEN HUB", 41.50, -81.69, "Cleveland, OH", "ATSI", "ATSI", "ATSI", "Western"),
    MarketSpec("CHICAGO_HUB", "CHICAGO HUB", "hub", None, "CHICAGO HUB", 41.88, -87.63, "Chicago, IL", "NI", "COMED", "COMED", "Western"),
    MarketSpec("DOMINION_HUB", "DOMINION HUB", "hub", None, "DOMINION HUB", 37.54, -77.44, "Richmond, VA", "DOM", "DOM", "DOMINION", "Southern"),
    MarketSpec("EASTERN_HUB", "EASTERN HUB", "hub", None, "EASTERN HUB", 39.95, -75.16, "Philadelphia, PA", "MIDATL", "MIDATL", "MID_ATLANTIC_REGION", "Mid-Atlantic"),
    MarketSpec("NEW_JERSEY_HUB", "NEW JERSEY HUB", "hub", None, "NEW JERSEY HUB", 40.74, -74.17, "Newark, NJ", "MIDATL", "MIDATL", "MID_ATLANTIC_REGION", "Mid-Atlantic"),
    MarketSpec("OHIO_HUB", "OHIO HUB", "hub", None, "OHIO HUB", 39.96, -83.00, "Columbus, OH", "AEP", "AEP", "AEP", "Western"),
    MarketSpec("WESTERN_HUB", "WESTERN HUB", "hub", None, "WESTERN HUB", 40.44, -79.99, "Pittsburgh, PA", "DUQ", "DUQ", "DUQUESNE", "Western"),
]

MARKETS: Dict[str, MarketSpec] = {m.market_id: m for m in _MARKETS}


def get_market(market_id: str) -> MarketSpec:
    try:
        return MARKETS[market_id]
    except KeyError as exc:
        raise KeyError(f"Unknown PJM market identifier: {market_id}") from exc


def markets_by_kind(kind: str) -> List[MarketSpec]:
    return [m for m in _MARKETS if m.kind == kind]


def market_options(kind: str | None = None) -> Dict[str, str]:
    values = _MARKETS if kind is None else markets_by_kind(kind)
    return {m.market_id: m.label for m in values}
