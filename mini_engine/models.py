"""Three forecasters plus the oracle. Each returns [P(home), P(draw), P(away)]."""
from __future__ import annotations

import math

MAX_GOALS = 8   # Tips8ng's Poisson model also sums a 0..8 score grid


def grid_outcome(lam_h: float, lam_a: float) -> list[float]:
    """1X2 probabilities from two independent Poisson goal counts on a 0..8 grid."""
    ph = [math.exp(-lam_h) * lam_h ** k / math.factorial(k)
          for k in range(MAX_GOALS + 1)]
    pa = [math.exp(-lam_a) * lam_a ** k / math.factorial(k)
          for k in range(MAX_GOALS + 1)]
    home = sum(ph[i] * pa[j] for i in range(MAX_GOALS + 1) for j in range(i))
    draw = sum(ph[i] * pa[i] for i in range(MAX_GOALS + 1))
    away = sum(ph[i] * pa[j] for i in range(MAX_GOALS + 1)
               for j in range(i + 1, MAX_GOALS + 1))
    total = home + draw + away                   # renormalise the truncated grid
    return [home / total, draw / total, away / total]


class Prior:
    """Base rate: the training window's home/draw/away frequencies, for every match."""

    def fit(self, train):
        counts = [1.0, 1.0, 1.0]                  # add-one smoothing
        for m in train:
            counts[m.outcome] += 1
        s = sum(counts)
        self.p = [c / s for c in counts]
        return self

    def predict(self, m, elo=None):
        return list(self.p)


class EloLogit:
    """Multinomial logistic regression on one feature, the pre-match Elo difference.

    Fitted by plain gradient descent on mean log loss. The feature is
    standardised, as in the Tips8ng research 'Elo-only' model.
    """

    def fit(self, train, elos, iters: int = 300, lr: float = 0.5):
        xs = [rh - ra for rh, ra in elos]
        self.mu = sum(xs) / len(xs)
        self.sd = (sum((x - self.mu) ** 2 for x in xs) / len(xs)) ** 0.5 or 1.0
        z = [(x - self.mu) / self.sd for x in xs]
        ys = [m.outcome for m in train]
        self.a, self.b = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        n = len(z)
        for _ in range(iters):
            ga, gb = [0.0] * 3, [0.0] * 3
            for zi, yi in zip(z, ys):
                p = self._softmax(zi)
                for k in range(3):
                    err = p[k] - (1.0 if k == yi else 0.0)
                    ga[k] += err
                    gb[k] += err * zi
            for k in range(3):
                self.a[k] -= lr * ga[k] / n
                self.b[k] -= lr * gb[k] / n
        return self

    def _softmax(self, zi: float) -> list[float]:
        s = [self.a[k] + self.b[k] * zi for k in range(3)]
        mx = max(s)
        e = [math.exp(v - mx) for v in s]
        t = sum(e)
        return [v / t for v in e]

    def predict(self, m, elo):
        return self._softmax((elo[0] - elo[1] - self.mu) / self.sd)


class Poisson:
    """Maher-style multiplicative model: lam_home = g * att[h] * dfn[a],
    lam_away = att[a] * dfn[h]. Fitted by iterative proportional fitting, with
    pseudo-counts so a team with few matches stays near average.
    """

    PSEUDO = 2.0   # pseudo-matches of 'average' evidence per team

    def fit(self, train, iters: int = 40):
        teams = sorted({m.home for m in train} | {m.away for m in train})
        att = {t: 1.0 for t in teams}
        dfn = {t: 1.0 for t in teams}
        g = 1.0
        mean_goals = sum(m.home_goals + m.away_goals for m in train) / (2 * len(train))
        for _ in range(iters):
            num = {t: self.PSEUDO * mean_goals for t in teams}
            den = {t: self.PSEUDO * mean_goals for t in teams}
            for m in train:
                num[m.home] += m.home_goals
                den[m.home] += g * dfn[m.away]
                num[m.away] += m.away_goals
                den[m.away] += dfn[m.home]
            att = {t: num[t] / den[t] for t in teams}
            num = {t: self.PSEUDO * mean_goals for t in teams}
            den = {t: self.PSEUDO * mean_goals for t in teams}
            for m in train:
                num[m.away] += m.home_goals
                den[m.away] += g * att[m.home]
                num[m.home] += m.away_goals
                den[m.home] += att[m.away]
            dfn = {t: num[t] / den[t] for t in teams}
            g = (sum(m.home_goals for m in train)
                 / sum(att[m.home] * dfn[m.away] for m in train))
        self.att, self.dfn, self.g = att, dfn, g
        return self

    def predict(self, m, elo=None):
        lh = self.g * self.att.get(m.home, 1.0) * self.dfn.get(m.away, 1.0)
        la = self.att.get(m.away, 1.0) * self.dfn.get(m.home, 1.0)
        return grid_outcome(lh, la)


class Oracle:
    """Knows the TRUE goal means. Only possible because the league is synthetic."""

    def fit(self, train, elos=None):
        return self

    def predict(self, m, elo=None):
        return grid_outcome(m.lam_home, m.lam_away)
