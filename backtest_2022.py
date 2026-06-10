#!/usr/bin/env python3
"""Backtest baseline vs enhanced model on the 2022 FIFA World Cup."""

from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd

import predict_worldcup as pw

_ORIGINAL_EFFECTIVE_ELO = pw.effective_elo

from predict_worldcup import (
    DATA_DIR,
    EloSystem,
    GroupStanding,
    N_SIMULATIONS,
    OUTPUT_DIR,
    RANDOM_SEED,
    clear_enhanced_model,
    init_enhanced_model,
    normalize_team,
    simulate_knockout_match,
    simulate_group,
)

OUTPUT_DIR.mkdir(exist_ok=True)
WC_CUTOFF = pd.Timestamp("2022-11-20")

GROUPS_2022 = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}

R16_FIXTURES_2022 = [
    ("1A", "2B"), ("1C", "2D"), ("1B", "2A"), ("1D", "2C"),
    ("1E", "2F"), ("1G", "2H"), ("1F", "2E"), ("1H", "2G"),
]

ACTUAL_RANKINGS_2022 = {
    "Argentina": 1, "France": 2, "Croatia": 3, "Morocco": 4,
    "Brazil": 5, "Netherlands": 6, "Portugal": 7, "England": 8,
    "Japan": 9, "Senegal": 10, "Australia": 11, "Switzerland": 12,
    "Spain": 13, "United States": 14, "South Korea": 15, "Poland": 16,
    "Germany": 17, "Uruguay": 18, "Tunisia": 19, "Mexico": 20,
    "Ghana": 21, "Cameroon": 22, "Serbia": 23, "Belgium": 24,
    "Iran": 25, "Wales": 26, "Canada": 27, "Costa Rica": 28,
    "Denmark": 29, "Saudi Arabia": 30, "Ecuador": 31, "Qatar": 32,
}


def fit_pre_tournament_elo() -> EloSystem:
    results = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    results = results.dropna(subset=["home_score", "away_score"])
    pre = results[results["date"] < WC_CUTOFF]

    elo = EloSystem()
    elo.fit(pre)
    return elo


def raw_effective(elo: EloSystem, team: str) -> float:
    return elo.ratings[normalize_team(team)]


def resolve_slot_2022(slot: str, group_results: dict[str, list[GroupStanding]]) -> str:
    return group_results[slot[1]][int(slot[0]) - 1].team


def simulate_tournament_2022(elo: EloSystem, rng: np.random.Generator) -> dict[str, int]:
    group_results = {
        grp: simulate_group(elo, teams, rng)
        for grp, teams in GROUPS_2022.items()
    }

    r16 = [
        (resolve_slot_2022(a, group_results), resolve_slot_2022(b, group_results))
        for a, b in R16_FIXTURES_2022
    ]

    def play_round(matches: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
        winners, losers = [], []
        for a, b in matches:
            winner, loser = simulate_knockout_match(elo, a, b, rng)
            winners.append(winner)
            losers.append(loser)
        return winners, losers

    r16_winners, r16_losers = play_round(r16)

    qf_pairs = [
        (r16_winners[i], r16_winners[i + 1])
        for i in range(0, 8, 2)
    ]
    qf_winners, qf_losers = play_round(qf_pairs)

    sf_pairs = [
        (qf_winners[i], qf_winners[i + 1])
        for i in range(0, 4, 2)
    ]
    sf_winners, sf_losers = play_round(sf_pairs)

    final_w, final_l = simulate_knockout_match(
        elo, sf_winners[0], sf_winners[1], rng
    )
    bronze_w, bronze_l = simulate_knockout_match(
        elo, sf_losers[0], sf_losers[1], rng
    )

    # Round-based placements.
    # This avoids fake ranking differences caused only by bracket order.
    placements = {
        final_w: 1,
        final_l: 2,
        bronze_w: 3,
        bronze_l: 4,
    }

    for team in qf_losers:
        placements.setdefault(team, 5)

    for team in r16_losers:
        placements.setdefault(team, 9)

    for ranked in group_results.values():
        for standing in ranked[2:]:
            placements.setdefault(standing.team, 17)

    return placements


def run_backtest(elo: EloSystem, label: str) -> dict:
    rng = np.random.default_rng(RANDOM_SEED)

    all_teams = [
        team
        for teams in GROUPS_2022.values()
        for team in teams
    ]

    finish_counts = {team: [] for team in all_teams}
    champion_counts: dict[str, int] = defaultdict(int)

    for _ in range(N_SIMULATIONS):
        sim_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        placements = simulate_tournament_2022(elo, sim_rng)

        for team, place in placements.items():
            finish_counts[team].append(place)

        champion = min(placements, key=placements.get)
        champion_counts[champion] += 1

    rows = []
    for team in all_teams:
        places = finish_counts[team]
        rows.append({
            "team": team,
            "avg_finish": float(np.mean(places)),
            "champion_pct": 100 * champion_counts[team] / N_SIMULATIONS,
            "top4_pct": 100 * sum(1 for place in places if place <= 4) / N_SIMULATIONS,
            "top8_pct": 100 * sum(1 for place in places if place <= 8) / N_SIMULATIONS,
            "actual_rank": ACTUAL_RANKINGS_2022[team],
        })

    df = pd.DataFrame(rows).sort_values(
        ["champion_pct", "top4_pct", "top8_pct", "avg_finish"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    df["predicted_rank"] = range(1, len(df) + 1)
    df["rank_error"] = (df["predicted_rank"] - df["actual_rank"]).abs()

    champion_pick = df.loc[df["champion_pct"].idxmax(), "team"]

    top4_pred = set(df.head(4)["team"])
    top4_actual = {
        team
        for team, rank in ACTUAL_RANKINGS_2022.items()
        if rank <= 4
    }

    return {
        "label": label,
        "champion_pick": champion_pick,
        "argentina_win_pct": df.loc[
            df["team"] == "Argentina", "champion_pct"
        ].iloc[0],
        "morocco_pred_rank": int(
            df.loc[df["team"] == "Morocco", "predicted_rank"].iloc[0]
        ),
        "croatia_pred_rank": int(
            df.loc[df["team"] == "Croatia", "predicted_rank"].iloc[0]
        ),
        "mae": df["rank_error"].mean(),
        "top4_overlap": len(top4_pred & top4_actual),
        "df": df,
    }


def main() -> None:
    print("=" * 72)
    print("2022 WORLD CUP BACKTEST: BASELINE vs ENHANCED MODEL")
    print("=" * 72)

    elo = fit_pre_tournament_elo()
    print(f"Training data: matches before {WC_CUTOFF.date()}\n")

    # Baseline: raw career Elo only.
    clear_enhanced_model()
    pw.effective_elo = raw_effective
    baseline = run_backtest(elo, "Baseline (career Elo)")

    # Enhanced: portfolio model.
    pw.effective_elo = _ORIGINAL_EFFECTIVE_ELO
    clear_enhanced_model()
    init_enhanced_model(WC_CUTOFF)
    enhanced = run_backtest(elo, "Enhanced")

    clear_enhanced_model()
    pw.effective_elo = _ORIGINAL_EFFECTIVE_ELO

    print(f"{'Metric':<35} {'Baseline':>14} {'Enhanced':>14}")
    print("-" * 65)
    print(f"{'Champion pick':<35} {baseline['champion_pick']:>14} {enhanced['champion_pick']:>14}")
    print(f"{'Argentina champion %':<35} {baseline['argentina_win_pct']:>13.1f}% {enhanced['argentina_win_pct']:>13.1f}%")
    print(f"{'Morocco predicted rank':<35} {baseline['morocco_pred_rank']:>14} {enhanced['morocco_pred_rank']:>14}")
    print(f"{'Croatia predicted rank':<35} {baseline['croatia_pred_rank']:>14} {enhanced['croatia_pred_rank']:>14}")
    print(f"{'Top 4 overlap':<35} {baseline['top4_overlap']:>11}/4 {enhanced['top4_overlap']:>11}/4")
    print(f"{'Mean absolute rank error':<35} {baseline['mae']:>14.1f} {enhanced['mae']:>14.1f}")

    print("\nActual 2022: Argentina champion | Top 4: Argentina, France, Croatia, Morocco")

    print("\nEnhanced model rating breakdown:")
    from model_enhancements import build_enhanced_ratings, top_adjustments

    ratings = build_enhanced_ratings(WC_CUTOFF)
    key_teams = ["Morocco", "Croatia", "Japan", "Argentina", "Belgium", "Brazil"]
    adj = top_adjustments(ratings, key_teams, len(key_teams))

    cols = [
        col for col in [
            "team",
            "career_elo",
            "form_elo",
            "blended",
            "external",
            "total",
        ]
        if col in adj.columns
    ]

    print(adj[cols].to_string(index=False))

    enhanced["df"].to_csv(OUTPUT_DIR / "backtest_2022_enhanced.csv", index=False)
    baseline["df"].to_csv(OUTPUT_DIR / "backtest_2022_baseline.csv", index=False)

    print(f"\nSaved comparison to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()