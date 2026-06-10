"""
Machine learning layer: train on all past World Cups to predict tournament finish.
Ridge regression (numpy-only) — no sklearn dependency at runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from model_enhancements import EnhancedRatings
from predict_worldcup import DATA_DIR, EloSystem, GROUPS_2026, normalize_team

OUTPUT_DIR = Path(__file__).parent / "output"

CONFED_MAP = {
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    "France": "UEFA", "Germany": "UEFA", "Spain": "UEFA", "England": "UEFA",
    "Netherlands": "UEFA", "Portugal": "UEFA", "Belgium": "UEFA",
    "Croatia": "UEFA", "Switzerland": "UEFA", "Austria": "UEFA",
    "Scotland": "UEFA", "Norway": "UEFA", "Sweden": "UEFA", "Turkey": "UEFA",
    "Czechia": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Australia": "AFC",
    "Saudi Arabia": "AFC", "Qatar": "AFC", "Iraq": "AFC", "Jordan": "AFC",
    "Uzbekistan": "AFC",
    "Morocco": "CAF", "Senegal": "CAF", "Ghana": "CAF", "Egypt": "CAF",
    "Ivory Coast": "CAF", "Algeria": "CAF", "Tunisia": "CAF", "South Africa": "CAF",
    "DR Congo": "CAF", "Cape Verde": "CAF",
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Panama": "CONCACAF", "Haiti": "CONCACAF", "Curaçao": "CONCACAF",
    "New Zealand": "OFC",
}

FEATURE_COLS = [
    "pre_elo", "elo_rank", "elo_rank_pct", "is_host",
    "confed_uefa", "confed_conmebol", "confed_caf",
    "prior_best_finish", "prior_avg_finish", "prior_appearances",
    "squad_value_log", "injury_loss", "manager_months", "travel_km",
]


def _confed(team: str) -> str:
    return CONFED_MAP.get(normalize_team(team), "UEFA")


def _fit_elo_pre(results: pd.DataFrame, cutoff: pd.Timestamp) -> EloSystem:
    pre = results[(results["date"] < cutoff)].dropna(subset=["home_score", "away_score"])
    elo = EloSystem()
    elo.fit(pre)
    return elo


def build_training_data() -> pd.DataFrame:
    results = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    standings = pd.read_csv(DATA_DIR / "fjelstul_tournament_standings.csv")
    hosts = pd.read_csv(DATA_DIR / "fjelstul_host_countries.csv")

    host_lookup: dict[int, set[str]] = {}
    for row in hosts.itertuples():
        year = int(row.tournament_name[:4])
        host_lookup.setdefault(year, set()).add(normalize_team(row.team_name))

    records = []
    for tourn in sorted(standings["tournament_name"].unique()):
        year = int(tourn[:4])
        cutoff = pd.Timestamp(f"{year}-06-01")
        t_standings = standings[standings["tournament_name"] == tourn]
        teams = [normalize_team(t) for t in t_standings["team_name"]]
        n_teams = len(teams)

        elo = _fit_elo_pre(results, cutoff)
        elos = {t: elo.ratings[t] for t in teams}
        sorted_teams = sorted(elos, key=elos.get, reverse=True)
        elo_rank = {t: i + 1 for i, t in enumerate(sorted_teams)}

        prior = standings[standings["tournament_name"].str[:4].astype(int) < year]
        prior_best: dict[str, list[int]] = {}
        for row in prior.itertuples():
            tm = normalize_team(row.team_name)
            prior_best.setdefault(tm, []).append(row.position)

        for row in t_standings.itertuples():
            team = normalize_team(row.team_name)
            pb = prior_best.get(team, [n_teams])
            records.append({
                "year": year,
                "team": team,
                "finish_rank": row.position,
                "n_teams": n_teams,
                "pre_elo": elos[team],
                "elo_rank": elo_rank[team],
                "elo_rank_pct": elo_rank[team] / n_teams,
                "is_host": int(team in host_lookup.get(year, set())),
                "confed_uefa": int(_confed(team) == "UEFA"),
                "confed_conmebol": int(_confed(team) == "CONMEBOL"),
                "confed_caf": int(_confed(team) == "CAF"),
                "prior_best_finish": min(pb),
                "prior_avg_finish": float(np.mean(pb)),
                "prior_appearances": len(pb),
            })

    return pd.DataFrame(records)


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return (X - mu) / sigma, mu, sigma


class WorldCupMLModel:
    """Ridge regression predicting tournament finish rank (lower = better)."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.weights: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.cv_mae_: float | None = None

    def _build_X(self, df: pd.DataFrame, historical: bool = True) -> np.ndarray:
        X = df[self._feature_cols_available(df, historical)].copy()
        if historical:
            X["squad_value_log"] = np.log1p(df["pre_elo"] / 10)
            X["injury_loss"] = 0.0
            X["manager_months"] = 36.0
            X["travel_km"] = 1500.0
        return X.values.astype(float)

    def _feature_cols_available(self, df: pd.DataFrame, historical: bool) -> list[str]:
        return FEATURE_COLS if not historical else FEATURE_COLS[:10]

    def _fit_weights(self, df: pd.DataFrame) -> None:
        X_raw = self._build_X(df, historical=True)
        X, self.mu, self.sigma = _standardize(X_raw)
        X = np.column_stack([np.ones(len(X)), X])
        y = df["finish_rank"].values.astype(float)

        n_feat = X.shape[1]
        reg = self.alpha * np.eye(n_feat)
        reg[0, 0] = 0
        self.weights = np.linalg.lstsq(X.T @ X + reg, X.T @ y, rcond=None)[0]

    def train(self, df: pd.DataFrame) -> None:
        self._fit_weights(df)

        errors = []
        for year in df["year"].unique():
            train = df[df["year"] != year]
            test = df[df["year"] == year]
            if len(train) < 20 or len(test) < 4:
                continue
            tmp = WorldCupMLModel(alpha=self.alpha)
            tmp._fit_weights(train)
            pred = tmp.predict_df(test)
            errors.extend(np.abs(pred - test["finish_rank"].values))
        self.cv_mae_ = float(np.mean(errors)) if errors else None

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        X_raw = self._build_X(df, historical=True)
        X = (X_raw - self.mu) / self.sigma
        X = np.column_stack([np.ones(len(X)), X])
        return X @ self.weights

    def predict_finish_2026(self, ratings: EnhancedRatings, ext_bundle) -> dict[str, float]:
        from external_factors import haversine_km

        squad_df = pd.read_csv(DATA_DIR / "squad_values_2026.csv")
        squad_df["team"] = squad_df["team"].map(normalize_team)
        ext_df = pd.read_csv(DATA_DIR / "external_factors_2026.csv", parse_dates=["manager_appointed"])
        ext_df["team"] = ext_df["team"].map(normalize_team)
        venues_df = pd.read_csv(DATA_DIR / "group_venues_2026.csv")
        standings = pd.read_csv(DATA_DIR / "fjelstul_tournament_standings.csv")

        teams = [t for grp in GROUPS_2026.values() for t in grp]
        elos = {t: ratings.blended_base(t) for t in teams}
        sorted_teams = sorted(elos, key=elos.get, reverse=True)

        rows = []
        for team in teams:
            team = normalize_team(team)
            elo_rank = sorted_teams.index(team) + 1
            prior = standings[standings["team_name"].map(normalize_team) == team]
            pb = prior["position"].tolist() if len(prior) else [48]

            squad_row = squad_df[squad_df["team"] == team]
            ext_row = ext_df[ext_df["team"] == team]
            mv = float(squad_row["market_value_meur"].iloc[0]) if len(squad_row) else 50.0
            injury = float(ext_row["injury_loss_meur"].iloc[0]) if len(ext_row) else 0.0
            months = 24.0
            travel_km = 2000.0
            if len(ext_row):
                months = (pd.Timestamp("2026-06-11") - ext_row["manager_appointed"].iloc[0]).days / 30.44
                group = next(g for g, ts in GROUPS_2026.items() if team in ts)
                venues = venues_df[venues_df["group"] == group]
                camp_lat = ext_row["base_camp_lat"].iloc[0]
                camp_lon = ext_row["base_camp_lon"].iloc[0]
                travel_km = float(np.mean([
                    haversine_km(camp_lat, camp_lon, r.lat, r.lon) for r in venues.itertuples()
                ]))

            rows.append({
                "pre_elo": ratings.blended_base(team),
                "elo_rank": elo_rank,
                "elo_rank_pct": elo_rank / 48,
                "is_host": int(team in {"Mexico", "Canada", "United States"}),
                "confed_uefa": int(_confed(team) == "UEFA"),
                "confed_conmebol": int(_confed(team) == "CONMEBOL"),
                "confed_caf": int(_confed(team) == "CAF"),
                "prior_best_finish": min(pb),
                "prior_avg_finish": float(np.mean(pb)),
                "prior_appearances": len(prior),
                "squad_value_log": np.log1p(mv),
                "injury_loss": injury,
                "manager_months": months,
                "travel_km": travel_km,
            })

        df = pd.DataFrame(rows)
        X_raw = df[FEATURE_COLS].values.astype(float)
        X = (X_raw - self.mu) / self.sigma
        X = np.column_stack([np.ones(len(X)), X])
        preds = np.clip(X @ self.weights, 1, 48)
        return {normalize_team(t): float(p) for t, p in zip(teams, preds)}

    def finish_to_elo_bonus(self, predicted_finish: dict[str, float]) -> dict[str, float]:
        mean_f = float(np.mean(list(predicted_finish.values())))
        return {team: (mean_f - finish) * 3.5 for team, finish in predicted_finish.items()}


def train_and_save() -> WorldCupMLModel:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = build_training_data()
    ml = WorldCupMLModel(alpha=2.0)
    ml.train(df)
    return ml
