#!/usr/bin/env python3
"""
2026 FIFA World Cup prediction model.

- Elo ratings from all men's internationals since 1872 (martj42/international_results)
- Historical World Cup group patterns (Fjelstul World Cup Database)
- Monte Carlo simulation of 2026 tournament (48 teams, 12 groups of 4)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
N_SIMULATIONS = 100_000
RANDOM_SEED = 42

# Map 2026 World Cup team names -> international_results.csv names
TEAM_NAME_MAP = {
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "United States": "United States",
    "USA": "United States",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Iran": "Iran",
    "IR Iran": "Iran",
    "Czech Republic": "Czechia",
    "Czechia": "Czechia",
}

# 2026 World Cup draw (all 48 teams confirmed as of June 2026)
GROUPS_2026 = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# FIFA draw pots (proxy for qualifying strength / pre-tournament ranking)
POT_1 = {
    "Mexico", "Canada", "United States", "Spain", "Argentina", "France",
    "England", "Brazil", "Portugal", "Netherlands", "Belgium", "Germany",
}
POT_2 = {
    "Croatia", "Morocco", "Colombia", "Uruguay", "Switzerland", "Japan",
    "Senegal", "Iran", "South Korea", "Ecuador", "Austria", "Australia",
}
POT_3 = {
    "Norway", "Panama", "Egypt", "Algeria", "Scotland", "Paraguay",
    "Tunisia", "Ivory Coast", "Uzbekistan", "Qatar", "Saudi Arabia",
    "South Africa", "Jordan", "Cape Verde", "Ghana", "Curaçao",
}
POT_4 = {
    "New Zealand", "Haiti", "Sweden", "DR Congo", "Iraq", "Bosnia and Herzegovina",
    "Turkey", "Czechia",
}

# Qualifying pathway adjustments (points added to Elo for simulation)
# Based on how teams reached the tournament (group winners vs playoffs)
QUALIFIER_ADJUSTMENTS = {
    "Bosnia and Herzegovina": 15,   # UEFA playoff winner (beat Italy)
    "Turkey": 10,                   # UEFA playoff
    "Czechia": 10,                  # UEFA playoff
    "Sweden": 5,                    # UEFA playoff
    "Iraq": 20,                     # AFC playoff
    "DR Congo": 15,                 # CAF playoff
    "New Zealand": -25,             # Lowest-ranked qualifier
    "Curaçao": -15,                 # Debutant, Concacaf
    "Haiti": -10,
    "Jordan": -10,                  # Debutant
    "Uzbekistan": -5,               # Debutant but strong Asian side
    "Cape Verde": -5,               # Debutant
    "Mexico": 5,                    # Host
    "Canada": 5,
    "United States": 5,
}

# Round of 32 fixtures: (slot_a, slot_b) where slot is like "1A", "2B", "3ABCD"
R32_FIXTURES = [
    ("2A", "2B"),           # 73
    ("1E", "3ABCDF"),       # 74
    ("1F", "2C"),           # 75
    ("1C", "2F"),           # 76
    ("1I", "3CDFGH"),       # 77
    ("2E", "2I"),           # 78
    ("1A", "3CEFHI"),       # 79
    ("1L", "3EHIJK"),       # 80
    ("1D", "3BEFIJ"),       # 81
    ("1G", "3AEHIJ"),       # 82
    ("2K", "2L"),           # 83
    ("1H", "2J"),           # 84
    ("1B", "3EFGIJ"),       # 85
    ("1J", "2H"),           # 86
    ("1K", "3DEIJL"),       # 87
    ("2D", "2G"),           # 88
]

# Third-place slot eligibility: which groups can fill each 3rd-place slot
THIRD_SLOT_GROUPS = {
    "3ABCDF": list("ABCDF"),
    "3CDFGH": list("CDFGH"),
    "3CEFHI": list("CEFHI"),
    "3EHIJK": list("EHIJK"),
    "3BEFIJ": list("BEFIJ"),
    "3AEHIJ": list("AEHIJ"),
    "3EFGIJ": list("EFGIJ"),
    "3DEIJL": list("DEIJL"),
}


def normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def pot_bonus(team: str) -> float:
    # No draw-pot bonus in the portfolio model.
    # The draw pot mostly duplicates team strength already captured by Elo.
    return 0.0


@dataclass
class EloSystem:
    ratings: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 1500.0))
    home_advantage: float = 100.0
    base_k: float = 20.0

    def expected_score(self, rating_a: float, rating_b: float, home_adv: float = 0.0) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a - home_adv) / 400.0))

    def k_factor(self, tournament: str) -> float:
        t = tournament.lower()
        if "world cup" in t:
            if "qualification" in t or "qualifying" in t:
                return self.base_k * 1.25
            return self.base_k * 1.5
        if any(x in t for x in ("euro", "copa america", "nations league", "continental")):
            return self.base_k * 1.2
        if "friendly" in t:
            return self.base_k * 0.7
        return self.base_k

    def update_match(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        neutral: bool,
        tournament: str,
    ) -> None:
        home = normalize_team(home)
        away = normalize_team(away)
        k = self.k_factor(tournament)

        if home_score > away_score:
            score_home, score_away = 1.0, 0.0
        elif home_score < away_score:
            score_home, score_away = 0.0, 1.0
        else:
            score_home, score_away = 0.5, 0.5

        ha = 0.0 if neutral or str(neutral).upper() == "TRUE" else self.home_advantage
        exp_home = self.expected_score(self.ratings[home], self.ratings[away], ha)
        exp_away = 1.0 - exp_home

        margin = abs(home_score - away_score)
        goal_mult = math.log(max(margin, 1) + 1) * (2.2 / (0.001 * abs(self.ratings[home] - self.ratings[away]) + 2.2))

        self.ratings[home] += k * goal_mult * (score_home - exp_home)
        self.ratings[away] += k * goal_mult * (score_away - exp_away)

    def fit(self, matches: pd.DataFrame) -> None:
        matches = matches.sort_values("date")
        for row in matches.itertuples():
            self.update_match(
                row.home_team,
                row.away_team,
                int(row.home_score),
                int(row.away_score),
                row.neutral,
                row.tournament,
            )


def load_and_fit_elo() -> tuple[EloSystem, pd.DataFrame]:
    results = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    results = results.dropna(subset=["home_score", "away_score"])
    print(f"Loaded {len(results):,} international matches since {results['date'].min().date()}")

    elo = EloSystem()
    elo.fit(results)
    return elo, results


def analyze_fjelstul_history(elo: EloSystem) -> pd.DataFrame:
    """Analyze historical WC group-stage outcomes vs pre-tournament Elo."""
    standings = pd.read_csv(DATA_DIR / "fjelstul_group_standings.csv")
    wc_matches = pd.read_csv(DATA_DIR / "fjelstul_matches.csv")

    # Rebuild Elo at start of each WC tournament using only pre-tournament matches
    results = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    tournaments = standings["tournament_name"].unique()

    records = []
    for tourn in sorted(tournaments):
        year = int(tourn[:4])
        cutoff = pd.Timestamp(f"{year}-06-01")
        pre = results[results["date"] < cutoff]
        temp_elo = EloSystem()
        temp_elo.fit(pre)

        t_standings = standings[standings["tournament_name"] == tourn]
        for row in t_standings.itertuples():
            team = normalize_team(row.team_name)
            records.append({
                "tournament": tourn,
                "year": year,
                "team": team,
                "group_position": row.position,
                "advanced": bool(row.advanced),
                "pre_elo": temp_elo.ratings[team],
            })

    hist = pd.DataFrame(records)
    hist["elo_rank_in_tournament"] = hist.groupby("tournament")["pre_elo"].rank(ascending=False, method="min")

    # Summary stats
    summary = hist.groupby("group_position").agg(
        avg_elo_rank=("elo_rank_in_tournament", "mean"),
        advance_rate=("advanced", "mean"),
        count=("team", "count"),
    ).reset_index()

    print("\n--- Fjelstul historical group-stage insights ---")
    print("Average pre-tournament Elo rank by final group position:")
    for _, r in summary.iterrows():
        print(f"  Position {int(r['group_position'])}: avg Elo rank {r['avg_elo_rank']:.1f}, "
              f"advance rate {r['advance_rate']*100:.0f}% ({int(r['count'])} teams)")

    return hist


@dataclass
class GroupStanding:
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def points(self) -> int:
        return 3 * self.wins + self.draws

    @property
    def gd(self) -> int:
        return self.gf - self.ga


# Set by init_enhanced_model() — adds form, WC history, qualifying momentum, knockout variance
_RATINGS_CONTEXT = None


def init_enhanced_model(cutoff=None) -> None:
    """Enable enhanced ratings (form + WC history + qualifying + knockout upsets)."""
    global _RATINGS_CONTEXT
    from model_enhancements import build_enhanced_ratings
    _RATINGS_CONTEXT = build_enhanced_ratings(cutoff)


def clear_enhanced_model() -> None:
    global _RATINGS_CONTEXT
    _RATINGS_CONTEXT = None


def effective_elo(elo: EloSystem, team: str) -> float:
    team = normalize_team(team)
    if _RATINGS_CONTEXT is not None:
        return _RATINGS_CONTEXT.blended_base(team) + _RATINGS_CONTEXT.static_bonus(team)
    return elo.ratings[team]


def simulate_score(elo_a: float, elo_b: float, rng: np.random.Generator) -> tuple[int, int]:
    """Poisson goals model from Elo difference (neutral venue)."""
    exp_a = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
    base_goals = 1.35
    lambda_a = base_goals * (0.5 + exp_a)
    lambda_b = base_goals * (1.5 - exp_a)
    return int(rng.poisson(lambda_a)), int(rng.poisson(lambda_b))


def play_match(
    elo: EloSystem,
    team_a: str,
    team_b: str,
    rng: np.random.Generator,
    knockout: bool = False,
) -> tuple[int, int]:
    if _RATINGS_CONTEXT is not None:
        ea, eb = _RATINGS_CONTEXT.match_elos(team_a, team_b, knockout=knockout, rng=rng)
    else:
        ea, eb = effective_elo(elo, team_a), effective_elo(elo, team_b)
    return simulate_score(ea, eb, rng)


def sort_standings(standings: list[GroupStanding]) -> list[GroupStanding]:
    return sorted(standings, key=lambda s: (s.points, s.gd, s.gf, effective_elo(EloSystem(), s.team)), reverse=True)


def simulate_group(elo: EloSystem, teams: list[str], rng: np.random.Generator) -> list[GroupStanding]:
    stats = {t: GroupStanding(team=t) for t in teams}
    fixtures = [(teams[i], teams[j]) for i in range(4) for j in range(i + 1, 4)]

    for home, away in fixtures:
        gh, ga = play_match(elo, home, away, rng)
        for team, gf, ga_against, won, drew in [
            (home, gh, ga, gh > ga, gh == ga),
            (away, ga, gh, ga > gh, gh == ga),
        ]:
            s = stats[team]
            s.played += 1
            s.gf += gf
            s.ga += ga_against
            if won:
                s.wins += 1
            elif drew:
                s.draws += 1
            else:
                s.losses += 1

    ranked = sorted(
        stats.values(),
        key=lambda s: (s.points, s.gd, s.gf, effective_elo(elo, s.team)),
        reverse=True,
    )
    return ranked


def rank_third_place(third_placers: list[tuple[str, str, GroupStanding]]) -> list[tuple[str, str, GroupStanding]]:
    """Rank third-place teams for best-8 selection (FIFA tiebreakers simplified)."""
    return sorted(
        third_placers,
        key=lambda x: (x[2].points, x[2].gd, x[2].gf, effective_elo(EloSystem(), x[1])),
        reverse=True,
    )


def resolve_slot(
    slot: str,
    group_results: dict[str, list[GroupStanding]],
    third_assignments: dict[str, str],
) -> str:
    if slot.startswith("3"):
        return third_assignments[slot]

    pos = int(slot[0])
    grp = slot[1]
    return group_results[grp][pos - 1].team


def assign_third_place_teams(
    advancing_thirds: dict[str, GroupStanding],
    elo: EloSystem,
) -> dict[str, str]:
    """Assign each advancing third-place team to a compatible R32 third-place slot."""
    assignments: dict[str, str] = {}
    unassigned_groups = set(advancing_thirds.keys())
    third_slots = list(THIRD_SLOT_GROUPS.keys())

    # Greedy: fill slots in order of fewest eligible remaining groups
    while unassigned_groups:
        best_slot = None
        best_candidates = []
        for slot in third_slots:
            if slot in assignments:
                continue
            eligible = [g for g in THIRD_SLOT_GROUPS[slot] if g in unassigned_groups]
            if not eligible:
                continue
            if best_slot is None or len(eligible) < len(best_candidates):
                best_slot = slot
                best_candidates = eligible

        if best_slot is None:
            # Fallback: any remaining slot/group pairing
            g = max(unassigned_groups, key=lambda x: (
                advancing_thirds[x].points, advancing_thirds[x].gd, effective_elo(elo, advancing_thirds[x].team)
            ))
            remaining_slots = [s for s in third_slots if s not in assignments]
            assignments[remaining_slots[0]] = advancing_thirds[g].team
            unassigned_groups.remove(g)
            continue

        chosen_g = max(
            best_candidates,
            key=lambda g: (
                advancing_thirds[g].points,
                advancing_thirds[g].gd,
                advancing_thirds[g].gf,
                effective_elo(elo, advancing_thirds[g].team),
            ),
        )
        assignments[best_slot] = advancing_thirds[chosen_g].team
        unassigned_groups.remove(chosen_g)

    return assignments


def simulate_knockout_match(elo: EloSystem, a: str, b: str, rng: np.random.Generator) -> tuple[str, str]:
    """Return (winner, loser)."""
    gh, ga = play_match(elo, a, b, rng, knockout=True)
    if gh != ga:
        return (a, b) if gh > ga else (b, a)
    for _ in range(2):
        gh, ga = play_match(elo, a, b, rng, knockout=True)
        if gh != ga:
            return (a, b) if gh > ga else (b, a)
    if _RATINGS_CONTEXT is not None:
        ea, eb = _RATINGS_CONTEXT.match_elos(a, b, knockout=True, rng=rng)
    else:
        ea, eb = effective_elo(elo, a), effective_elo(elo, b)
    p = 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))
    return (a, b) if rng.random() < p else (b, a)


def simulate_tournament(elo: EloSystem, rng: np.random.Generator) -> dict[str, int]:
    """Return final placement (1=best) for each team."""
    group_results: dict[str, list[GroupStanding]] = {}
    third_placers: list[tuple[str, str, GroupStanding]] = []

    for grp, teams in GROUPS_2026.items():
        ranked = simulate_group(elo, teams, rng)
        group_results[grp] = ranked
        third_placers.append((grp, ranked[2].team, ranked[2]))

    ranked_thirds = rank_third_place(third_placers)
    advancing_third_groups = {g for g, _, _ in ranked_thirds[:8]}
    advancing_thirds = {g: s for g, _, s in third_placers if g in advancing_third_groups}

    third_assignments = assign_third_place_teams(advancing_thirds, elo)
    r32 = [
        (
            resolve_slot(slot_a, group_results, third_assignments),
            resolve_slot(slot_b, group_results, third_assignments),
        )
        for slot_a, slot_b in R32_FIXTURES
    ]

    def play_round(matches: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
        winners, losers = [], []
        for a, b in matches:
            w, l = simulate_knockout_match(elo, a, b, rng)
            winners.append(w)
            losers.append(l)
        return winners, losers

    r32_winners, r32_losers = play_round(r32)
    r16_pairs = [(r32_winners[i], r32_winners[i + 1]) for i in range(0, 16, 2)]
    r16_winners, r16_losers = play_round(r16_pairs)
    qf_pairs = [(r16_winners[i], r16_winners[i + 1]) for i in range(0, 8, 2)]
    qf_winners, qf_losers = play_round(qf_pairs)
    sf_pairs = [(qf_winners[i], qf_winners[i + 1]) for i in range(0, 4, 2)]
    sf_winners, sf_losers = play_round(sf_pairs)

    final_w, final_l = simulate_knockout_match(elo, sf_winners[0], sf_winners[1], rng)
    bronze_w, bronze_l = simulate_knockout_match(elo, sf_losers[0], sf_losers[1], rng)

    # Use round-based placements instead of arbitrary bracket-order ranks.
    # Old bug: QF loser #1 became rank 9, QF loser #4 became rank 12 just
    # because of bracket order. That distorted avg_finish badly.
    placements: dict[str, int] = {
        final_w: 1,
        final_l: 2,
        bronze_w: 3,
        bronze_l: 4,
    }

    for team in sf_losers:
        placements.setdefault(team, 4)     # reached semifinal / top 4
    for team in qf_losers:
        placements.setdefault(team, 5)     # reached quarterfinal / top 8
    for team in r16_losers:
        placements.setdefault(team, 9)     # reached round of 16 / top 16
    for team in r32_losers:
        placements.setdefault(team, 17)    # reached round of 32 / top 32

    # Group-stage eliminations. Keep one consistent round value so average finish
    # measures tournament progress instead of fake precision.
    group_eliminated = [t for g, t, _ in ranked_thirds[8:]]
    for grp, ranked in group_results.items():
        group_eliminated.append(ranked[3].team)
    for team in group_eliminated:
        placements.setdefault(team, 33)

    return placements


def run_simulations(elo: EloSystem) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    all_teams = [t for teams in GROUPS_2026.values() for t in teams]

    finish_counts: dict[str, list[int]] = {t: [] for t in all_teams}
    win_counts: dict[str, int] = defaultdict(int)

    for _ in range(N_SIMULATIONS):
        sim_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        placements = simulate_tournament(elo, sim_rng)
        for team, place in placements.items():
            finish_counts[team].append(place)
        winner = min(placements, key=placements.get)
        win_counts[winner] += 1

    rows = []
    for team in all_teams:
        places = finish_counts[team]
        rows.append({
            "team": team,
            "elo": round(elo.ratings[normalize_team(team)], 1),
            "adjusted_elo": round(effective_elo(elo, team), 1),
            "avg_finish": round(np.mean(places), 2),
            "median_finish": int(np.median(places)),
            "champion_pct": round(100 * win_counts[team] / N_SIMULATIONS, 2),
            "final_pct": round(100 * sum(1 for p in places if p <= 2) / N_SIMULATIONS, 2),
            "sf_pct": round(100 * sum(1 for p in places if p <= 4) / N_SIMULATIONS, 2),
            "qf_pct": round(100 * sum(1 for p in places if p <= 5) / N_SIMULATIONS, 2),
            "r32_pct": round(100 * sum(1 for p in places if p <= 17) / N_SIMULATIONS, 2),
        })

    df = pd.DataFrame(rows)
    # Rank by title probability first, then deep-run odds. This keeps the printed
    # ranking aligned with the “most likely champion” line.
    df = df.sort_values(
        ["champion_pct", "final_pct", "sf_pct", "qf_pct", "avg_finish", "adjusted_elo"],
        ascending=[False, False, False, False, True, False],
    ).reset_index(drop=True)
    df["predicted_rank"] = range(1, len(df) + 1)
    return df


def print_group_predictions(elo: EloSystem) -> None:
    print("\n--- 2026 Group stage predictions (by adjusted Elo) ---")
    for grp, teams in GROUPS_2026.items():
        ranked = sorted(teams, key=lambda t: effective_elo(elo, t), reverse=True)
        elos = [f"{t} ({effective_elo(elo, t):.0f})" for t in ranked]
        print(f"  Group {grp}: 1. {elos[0]} | 2. {elos[1]} | 3. {elos[2]} | 4. {elos[3]}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("2026 FIFA WORLD CUP PREDICTION MODEL (FULL)")
    print("Elo + recent form + squad value + injuries + host/travel factors")
    print(f"Monte Carlo simulations: {N_SIMULATIONS:,}")
    print("=" * 70)

    elo, _ = load_and_fit_elo()
    init_enhanced_model()
    analyze_fjelstul_history(elo)

    from external_factors import external_factors_table

    print("\n--- External factors (Transfermarkt, injuries, managers, travel) ---")
    print(external_factors_table().head(12).to_string(index=False))
    print("\n--- Model design ---")
    print("Career Elo + corrected recent-form Elo + capped external factors. No noisy hand-made WC/qualifying/ML bonuses.")

    from model_enhancements import top_adjustments
    wc_teams_preview = [t for teams in GROUPS_2026.values() for t in teams]
    if _RATINGS_CONTEXT is not None:
        print("\n--- Rating breakdown (top 12) ---")
        print(top_adjustments(_RATINGS_CONTEXT, wc_teams_preview, 12).to_string(index=False))

    # Top Elo ratings for WC teams
    wc_teams = [t for teams in GROUPS_2026.values() for t in teams]
    elo_table = sorted(
        [(t, elo.ratings[normalize_team(t)], effective_elo(elo, t)) for t in wc_teams],
        key=lambda x: x[2], reverse=True,
    )

    print("\n--- Current Elo strength (2026 World Cup teams) ---")
    for i, (team, raw, adj) in enumerate(elo_table[:15], 1):
        print(f"  {i:2d}. {team:25s}  Elo: {raw:7.1f}  Adjusted: {adj:7.1f}")

    print_group_predictions(elo)

    print(f"\nRunning {N_SIMULATIONS:,} tournament simulations...")
    results = run_simulations(elo)

    print("\n" + "=" * 70)
    print("PREDICTED FINAL RANKINGS — 2026 FIFA WORLD CUP (all 48 teams)")
    print("=" * 70)
    print(f"{'Rank':<6}{'Team':<28}{'Elo':>7}{'Adj':>7}{'AvgFin':>8}{'Champ%':>8}{'Final%':>8}{'SF%':>7}{'QF%':>7}{'R32%':>7}")
    print("-" * 70)
    for _, r in results.iterrows():
        print(
            f"{int(r['predicted_rank']):<6}{r['team']:<28}"
            f"{r['elo']:>7.0f}{r['adjusted_elo']:>7.0f}"
            f"{r['avg_finish']:>8.1f}{r['champion_pct']:>7.1f}%"
            f"{r['final_pct']:>7.1f}%{r['sf_pct']:>6.1f}%{r['qf_pct']:>6.1f}%{r['r32_pct']:>6.1f}%"
        )

    champion = results.loc[results["champion_pct"].idxmax()]
    print("\n" + "=" * 70)
    print(f"MOST LIKELY CHAMPION: {champion['team']}")
    print(f"  Win probability: {champion['champion_pct']:.1f}%")
    print(f"  Final probability: {champion['final_pct']:.1f}%")
    print(f"  Semifinal probability: {champion['sf_pct']:.1f}%")
    print(f"  Quarterfinal probability: {champion['qf_pct']:.1f}%")
    print(f"  Adjusted Elo: {champion['adjusted_elo']:.0f}")
    print(f"  (Overall rank by average finish: #{int(results.loc[results['team']==champion['team'], 'predicted_rank'].iloc[0])})")
    print("=" * 70)

    results.to_csv(OUTPUT_DIR / "predicted_rankings_2026.csv", index=False)
    with open(OUTPUT_DIR / "elo_ratings.json", "w") as f:
        json.dump({normalize_team(t): round(elo.ratings[normalize_team(t)], 1) for t in wc_teams}, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
