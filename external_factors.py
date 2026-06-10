"""Squad quality, injuries, manager effect, and travel/climate factors for 2026."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from predict_worldcup import GROUPS_2026, DATA_DIR, normalize_team

REFERENCE_DATE = datetime(2026, 6, 11)
INJURY_ELO_PER_MEUR = 0.12
SQUAD_VALUE_SCALE = 45.0
TRAVEL_KM_PENALTY = 0.015
HOST_BONUS = 35.0
HOME_COUNTRY_MATCH_BONUS = 20.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _load_squad_values() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "squad_values_2026.csv")
    df["team"] = df["team"].map(normalize_team)
    return df


def _load_external() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "external_factors_2026.csv", parse_dates=["manager_appointed"])
    df["team"] = df["team"].map(normalize_team)
    return df


def _load_venues() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "group_venues_2026.csv")


def squad_value_bonus(team: str, squad_df: pd.DataFrame) -> float:
    team = normalize_team(team)
    row = squad_df[squad_df["team"] == team]
    if row.empty:
        return 0.0
    values = np.log1p(squad_df["market_value_meur"].values)
    z = (np.log1p(row["market_value_meur"].iloc[0]) - values.mean()) / values.std()
    return float(z * SQUAD_VALUE_SCALE)


def injury_penalty(team: str, ext_df: pd.DataFrame) -> float:
    team = normalize_team(team)
    row = ext_df[ext_df["team"] == team]
    if row.empty:
        return 0.0
    loss = row["injury_loss_meur"].iloc[0]
    return -float(loss * INJURY_ELO_PER_MEUR)


def manager_bonus(team: str, ext_df: pd.DataFrame) -> float:
    team = normalize_team(team)
    row = ext_df[ext_df["team"] == team]
    if row.empty:
        return 0.0
    months = (REFERENCE_DATE - row["manager_appointed"].iloc[0].to_pydatetime()).days / 30.44

    # New manager honeymoon (Ancelotti Brazil, Tuchel England, Marsch Canada, etc.)
    if months <= 6:
        return 28.0
    if months <= 18:
        return 18.0
    # Long successful tenures at a tournament
    if months >= 48 and team in {"Argentina", "Croatia", "France", "Morocco", "Japan", "Iran"}:
        return 12.0
    if months >= 36:
        return 6.0
    return 0.0


def travel_climate_bonus(team: str, ext_df: pd.DataFrame, venues_df: pd.DataFrame) -> float:
    team = normalize_team(team)
    row = ext_df[ext_df["team"] == team]
    if row.empty:
        return 0.0

    camp_lat = row["base_camp_lat"].iloc[0]
    camp_lon = row["base_camp_lon"].iloc[0]
    camp_country = row["base_camp_country"].iloc[0]

    # Host nations
    hosts = {"Mexico": "MEX", "Canada": "CAN", "United States": "USA"}
    if team in hosts and camp_country == hosts[team]:
        return HOST_BONUS

    group = next(g for g, teams in GROUPS_2026.items() if team in teams)
    group_venues = venues_df[venues_df["group"] == group]

    distances = [
        haversine_km(camp_lat, camp_lon, r.lat, r.lon)
        for r in group_venues.itertuples()
    ]
    avg_km = float(np.mean(distances))
    travel_penalty = -avg_km * TRAVEL_KM_PENALTY

    venue_countries = group_venues["country"].value_counts()
    majority_country = venue_countries.idxmax()
    if camp_country == majority_country:
        travel_penalty += HOME_COUNTRY_MATCH_BONUS

    # Mexico-based teams with groups partly in Mexico
    if camp_country == "MEX" and "MEX" in venue_countries.index:
        travel_penalty += 10.0

    return travel_penalty


@dataclass
class ExternalFactorBundle:
    squad_bonus: dict[str, float]
    injury_penalty: dict[str, float]
    manager_bonus: dict[str, float]
    travel_bonus: dict[str, float]

    def total(self, team: str) -> float:
        team = normalize_team(team)
        return (
            self.squad_bonus.get(team, 0)
            + self.injury_penalty.get(team, 0)
            + self.manager_bonus.get(team, 0)
            + self.travel_bonus.get(team, 0)
        )


def build_external_factors() -> ExternalFactorBundle:
    squad_df = _load_squad_values()
    ext_df = _load_external()
    venues_df = _load_venues()
    teams = [t for grp in GROUPS_2026.values() for t in grp]

    return ExternalFactorBundle(
        squad_bonus={t: squad_value_bonus(t, squad_df) for t in teams},
        injury_penalty={t: injury_penalty(t, ext_df) for t in teams},
        manager_bonus={t: manager_bonus(t, ext_df) for t in teams},
        travel_bonus={t: travel_climate_bonus(t, ext_df, venues_df) for t in teams},
    )


def external_factors_table() -> pd.DataFrame:
    bundle = build_external_factors()
    squad_df = _load_squad_values()
    ext_df = _load_external()
    teams = [t for grp in GROUPS_2026.values() for t in grp]
    rows = []
    for t in teams:
        t = normalize_team(t)
        mv = squad_df.loc[squad_df["team"] == t, "market_value_meur"]
        rows.append({
            "team": t,
            "market_value_meur": int(mv.iloc[0]) if len(mv) else 0,
            "squad_bonus": round(bundle.squad_bonus.get(t, 0), 1),
            "injury_penalty": round(bundle.injury_penalty.get(t, 0), 1),
            "manager_bonus": round(bundle.manager_bonus.get(t, 0), 1),
            "travel_bonus": round(bundle.travel_bonus.get(t, 0), 1),
            "external_total": round(bundle.total(t), 1),
        })
    return pd.DataFrame(rows).sort_values("external_total", ascending=False)
