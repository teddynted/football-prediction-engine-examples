"""Walk-forward evaluation of the SYNTHETIC mini engine.

    python3 evaluate.py            # prints the report (matches expected_output.txt)

Season 1 is history only. From season 2 on, models are refitted every
REFIT_EVERY rounds on every match played before that round, then predict the
next REFIT_EVERY rounds. No model ever sees a result from the round it predicts.
"""
from __future__ import annotations

import math

import elo
import metrics as M
from models import EloLogit, Oracle, Poisson, Prior
from synth import generate

REFIT_EVERY = 4
MODELS = {"prior": Prior, "elo-logit": EloLogit, "poisson": Poisson,
          "oracle": Oracle}


def walk_forward(matches):
    ratings = elo.pre_match_ratings(matches)          # pre-match, leakage-free
    keyed = list(zip(matches, ratings))
    preds = {name: [] for name in MODELS}
    ys = []
    blocks = sorted({(m.season, (m.round - 1) // REFIT_EVERY)
                     for m in matches if m.season > 1})
    for season, block in blocks:
        start = block * REFIT_EVERY + 1
        train = [(m, r) for m, r in keyed if (m.season, m.round) < (season, start)]
        test = [(m, r) for m, r in keyed
                if m.season == season and start <= m.round < start + REFIT_EVERY]
        tm, tr = [m for m, _ in train], [r for _, r in train]
        fitted = {}
        for name, cls in MODELS.items():
            fitted[name] = (cls().fit(tm, tr) if name == "elo-logit"
                            else cls().fit(tm))
        for m, r in test:
            ys.append(m.outcome)
            for name, model in fitted.items():
                preds[name].append(model.predict(m, r))
    return preds, ys


def report(preds, ys) -> str:
    lines = ["SYNTHETIC DATA - not Tips8ng results.",
             f"Out-of-sample matches: {len(ys)}  (seasons 2-4, walk-forward,"
             f" refit every {REFIT_EVERY} rounds)",
             "Outcome mix: home {:.3f}  draw {:.3f}  away {:.3f}".format(
                 *[ys.count(k) / len(ys) for k in range(3)]),
             "",
             f"{'model':<10} {'log loss':>9} {'Brier':>7} {'RPS':>7}"
             f" {'accuracy':>9} {'ECE':>6}",
             f"{'(better)':<10} {'lower':>9} {'lower':>7} {'lower':>7}"
             f" {'higher':>9} {'lower':>6}"]
    losses = {}
    for name, ps in preds.items():
        losses[name] = [M.log_loss(p, y) for p, y in zip(ps, ys)]
        lines.append(
            f"{name:<10} {M.mean(losses[name]):>9.4f}"
            f" {M.mean([M.brier(p, y) for p, y in zip(ps, ys)]):>7.4f}"
            f" {M.mean([M.rps(p, y) for p, y in zip(ps, ys)]):>7.4f}"
            f" {M.mean([M.correct(p, y) for p, y in zip(ps, ys)]):>9.4f}"
            f" {M.ece(ps, ys):>6.4f}")
    lines.append(f"{'uniform':<10} {math.log(3):>9.4f} {2 / 3:>7.4f}")
    lines += ["", "Paired bootstrap, log-loss difference vs prior,"
              " 1000 resamples (negative = better):"]
    for name in ("elo-logit", "poisson", "oracle"):
        d, (lo, hi) = M.paired_bootstrap(losses["prior"], losses[name])
        lines.append(f"  {name:<10} {d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    d, (lo, hi) = M.paired_bootstrap(losses["elo-logit"], losses["poisson"])
    lines.append(f"  poisson vs elo-logit {d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    lines.append("  (last line: negative = poisson better)")
    return "\n".join(lines)


def main() -> str:
    preds, ys = walk_forward(generate())
    return report(preds, ys)


if __name__ == "__main__":
    print(main())
