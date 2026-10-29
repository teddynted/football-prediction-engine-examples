"""Re-run the SYNTHETIC evaluation on several seeds (about 2.5 minutes).

One synthetic league is one random draw. This shows how much the log losses,
and the Poisson-vs-Elo ranking, move from one draw to the next.
"""
import metrics as M
from evaluate import walk_forward
from synth import generate

print("SYNTHETIC DATA - not Tips8ng results.")
print(f"{'seed':>4} {'prior':>7} {'elo':>7} {'poisson':>7} {'oracle':>7}"
      f" {'poisson-elo':>12}  95% CI")
for seed in (1, 2, 3, 4, 5, 6):
    preds, ys = walk_forward(generate(seed=seed))
    ll = {k: [M.log_loss(p, y) for p, y in zip(v, ys)] for k, v in preds.items()}
    d, (lo, hi) = M.paired_bootstrap(ll["elo-logit"], ll["poisson"])
    print(f"{seed:>4} {M.mean(ll['prior']):>7.4f} {M.mean(ll['elo-logit']):>7.4f}"
          f" {M.mean(ll['poisson']):>7.4f} {M.mean(ll['oracle']):>7.4f}"
          f" {d:>+12.4f}  [{lo:+.4f}, {hi:+.4f}]")
