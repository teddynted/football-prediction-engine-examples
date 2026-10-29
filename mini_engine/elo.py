"""Elo ratings with the same constants as Tips8ng's feature code.

Constants mirror services/ml-training/training/features/engineer.py (ELO_BASE,
ELO_K, ELO_HOME_ADV, ELO_SCALE, ELO_SEASON_REGRESSION). This is a re-implementation
for teaching, not an import of the production code.
"""
from __future__ import annotations

from collections import defaultdict

ELO_BASE = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 60.0
ELO_SCALE = 400.0
ELO_SEASON_REGRESSION = 0.25


def expected_home(r_home: float, r_away: float) -> float:
    """Expected home score (win = 1, draw = 0.5) before the match."""
    diff = r_home + ELO_HOME_ADV - r_away
    return 1.0 / (1.0 + 10 ** (-diff / ELO_SCALE))


def pre_match_ratings(matches) -> list[tuple[float, float]]:
    """Replay matches in order; return each match's PRE-match (home, away) ratings.

    The result of match i updates the ratings only after its pre-match ratings are
    recorded, so the feature for match i never contains match i's own result.
    """
    ratings: dict[str, float] = defaultdict(lambda: ELO_BASE)
    out, prev_season = [], None
    for m in matches:
        if prev_season is not None and m.season != prev_season:
            for t in list(ratings):   # pull every rating 25% of the way back to 1500
                ratings[t] = ELO_BASE + (1 - ELO_SEASON_REGRESSION) * (
                    ratings[t] - ELO_BASE)
        prev_season = m.season
        rh, ra = ratings[m.home], ratings[m.away]
        out.append((rh, ra))
        score = (1.0, 0.5, 0.0)[m.outcome]
        delta = ELO_K * (score - expected_home(rh, ra))
        ratings[m.home] += delta
        ratings[m.away] -= delta
    return out
