"""
Portfolio-grade World Cup model enhancements.

Design choice: keep only features that are explainable and defensible:
  1) Career Elo from all historical international matches
  2) Recent-form Elo, initialized from career Elo so it cannot collapse elite teams
  3) 2026 external factors: squad value, injury loss, manager stability, host/travel

Removed on purpose because backtesting showed they added noise:
  - hand-made World Cup history bonus
  - qualifying bonus
  - giant-killing bonus
  - ML bonus trained on a tiny historical WC sample
  - draw-pot bonus
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from predict_worldcup import DATA_DIR, EloSystem, GROUPS_2026, normalize_team

FORM_MONTHS = 30
CAREER_WEIGHT = 0.80
FORM_WEIGHT = 0.20
FORM_CAP = 45.0
FORM_K_MULTIPLIER = 1.8  # recent matches get extra weight versus career Elo

# Knockout matches are higher variance than group matches.
KNOCKOUT_SHRINK = 0.78
KNOCKOUT_NOISE_SD = 35

# External features are useful, but should not overpower actual match results.
EXTERNAL_WEIGHT = 0.45
EXTERNAL_CAP = 55.0


@dataclass
class EnhancedRatings:
    career_elo: EloSystem
    form_elo: EloSystem
    external_bonus: dict[str, float] = field(default_factory=dict)
    reference_date: pd.Timestamp = pd.Timestamp("today")

    def form_adjustment(self, team: str) -> float:
        team = normalize_team(team)
        diff = self.form_elo.ratings[team] - self.career_elo.ratings[team]
        return float(np.clip(diff * FORM_WEIGHT, -FORM_CAP, FORM_CAP))

    def blended_base(self, team: str) -> float:
        team = normalize_team(team)
        return float(self.career_elo.ratings[team] + self.form_adjustment(team))

    def static_bonus(self, team: str) -> float:
        team = normalize_team(team)
        return float(self.external_bonus.get(team, 0.0))

    def match_elos(
        self,
        team_a: str,
        team_b: str,
        knockout: bool = False,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, float]:
        elo_a = self.blended_base(team_a) + self.static_bonus(team_a)
        elo_b = self.blended_base(team_b) + self.static_bonus(team_b)

        if knockout:
            # In one-game knockouts, favorites are still favorites, but less dominant.
            mean = (elo_a + elo_b) / 2.0
            diff = (elo_a - elo_b) * KNOCKOUT_SHRINK
            elo_a = mean + diff / 2.0
            elo_b = mean - diff / 2.0
            if rng is not None:
                elo_a += float(rng.normal(0, KNOCKOUT_NOISE_SD))
                elo_b += float(rng.normal(0, KNOCKOUT_NOISE_SD))

        return elo_a, elo_b


def _fit_elo_on(matches: pd.DataFrame) -> EloSystem:
    elo = EloSystem()
    elo.fit(matches)
    return elo


def _fit_form_elo(results: pd.DataFrame, cutoff: pd.Timestamp) -> EloSystem:
    """
    Correct form Elo:
    Build the rating state up to the form window, then update only with recent games.
    This prevents the old bug where form Elo restarted everyone around 1500.
    """
    form_start = cutoff - pd.DateOffset(months=FORM_MONTHS)
    before_form = results[results["date"] < form_start]
    recent = results[(results["date"] >= form_start) & (results["date"] < cutoff)]

    elo = _fit_elo_on(before_form)
    # Make recent form an actual signal: career Elo already includes these matches once,
    # so this form model replays the recent window with a higher K-factor. The final
    # form adjustment is still capped, so short-term noise cannot dominate the model.
    elo.base_k *= FORM_K_MULTIPLIER
    elo.fit(recent)
    return elo


def _build_external_bonus(cutoff: pd.Timestamp) -> dict[str, float]:
    # External factors are only available for the 2026 forecast, not historical backtests.
    if cutoff <= pd.Timestamp("2026-01-01"):
        return {}

    try:
        from external_factors import build_external_factors
    except Exception:
        return {}

    bundle = build_external_factors()
    bonuses: dict[str, float] = {}
    for team in [t for teams in GROUPS_2026.values() for t in teams]:
        raw = bundle.total(team)
        bonuses[normalize_team(team)] = float(np.clip(raw * EXTERNAL_WEIGHT, -EXTERNAL_CAP, EXTERNAL_CAP))
    return bonuses


def build_enhanced_ratings(cutoff: pd.Timestamp | None = None) -> EnhancedRatings:
    results = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    results = results.dropna(subset=["home_score", "away_score"])

    if cutoff is None:
        cutoff = results["date"].max() + pd.Timedelta(days=1)

    career = results[results["date"] < cutoff]
    career_elo = _fit_elo_on(career)
    form_elo = _fit_form_elo(results, cutoff)
    external_bonus = _build_external_bonus(cutoff)

    return EnhancedRatings(
        career_elo=career_elo,
        form_elo=form_elo,
        external_bonus=external_bonus,
        reference_date=cutoff,
    )


def top_adjustments(ratings: EnhancedRatings, teams: list[str], n: int = 10) -> pd.DataFrame:
    rows = []
    for t in teams:
        t = normalize_team(t)
        rows.append({
            "team": t,
            "career_elo": round(ratings.career_elo.ratings[t], 1),
            "form_elo": round(ratings.form_elo.ratings[t], 1),
            "form_adj": round(ratings.form_adjustment(t), 1),
            "external": round(ratings.external_bonus.get(t, 0.0), 1),
            "total": round(ratings.blended_base(t) + ratings.static_bonus(t), 1),
        })
    return pd.DataFrame(rows).sort_values("total", ascending=False).head(n)
