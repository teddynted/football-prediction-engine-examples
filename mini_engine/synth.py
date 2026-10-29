"""SYNTHETIC league generator. Every number it produces is made up.

Teams have hidden attack and defence strengths. Goals are Poisson draws, so the
true outcome probabilities of every match are known, which a real league never
offers. That lets the evaluation compare each model with the best possible
forecaster ("oracle").
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

HOME_GOALS_BASE = 1.45   # synthetic league: mean home goals for average teams
AWAY_GOALS_BASE = 1.15   # synthetic league: mean away goals for average teams
STRENGTH_SD = 0.25       # spread of log attack / log defence across teams
DRIFT_SD = 0.08          # change in each team's strength between seasons


@dataclass(frozen=True)
class Match:
    season: int
    round: int
    home: str
    away: str
    home_goals: int
    away_goals: int
    lam_home: float      # true expected goals (hidden from the models)
    lam_away: float

    @property
    def outcome(self) -> int:
        """0 = home win, 1 = draw, 2 = away win (Tips8ng's HOME, DRAW, AWAY order)."""
        if self.home_goals > self.away_goals:
            return 0
        return 1 if self.home_goals == self.away_goals else 2


def poisson(rng: random.Random, lam: float) -> int:
    """Knuth's method; fine for the small means of football goals."""
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    """Double round robin by the circle method: 2 * (n - 1) rounds."""
    n, ts = len(teams), list(teams)
    first = []
    for r in range(n - 1):
        pairs = [(ts[i], ts[n - 1 - i]) for i in range(n // 2)]
        first.append([(a, b) if r % 2 == 0 else (b, a) for a, b in pairs])
        ts = [ts[0], ts[-1]] + ts[1:-1]
    return first + [[(b, a) for a, b in rnd] for rnd in first]


def generate(seed: int = 7, n_teams: int = 20, n_seasons: int = 4) -> list[Match]:
    rng = random.Random(seed)
    teams = [f"Team {chr(65 + i)}" for i in range(n_teams)]
    att = {t: rng.gauss(0.0, STRENGTH_SD) for t in teams}
    dfn = {t: rng.gauss(0.0, STRENGTH_SD) for t in teams}
    matches: list[Match] = []
    for season in range(1, n_seasons + 1):
        if season > 1:
            for t in teams:
                att[t] += rng.gauss(0.0, DRIFT_SD)
                dfn[t] += rng.gauss(0.0, DRIFT_SD)
        order = teams[:]
        rng.shuffle(order)
        for rnd, fixtures in enumerate(round_robin(order), start=1):
            for h, a in fixtures:
                lh = HOME_GOALS_BASE * math.exp(att[h] - dfn[a])
                la = AWAY_GOALS_BASE * math.exp(att[a] - dfn[h])
                matches.append(Match(season, rnd, h, a,
                                     poisson(rng, lh), poisson(rng, la), lh, la))
    return matches
