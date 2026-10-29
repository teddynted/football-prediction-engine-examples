# Mini prediction engine (SYNTHETIC)

Companion code for Appendix B of *Inside the Football Prediction Engine*. Every number this code
produces comes from a made-up league. None of it is a Tips8ng result.

Python 3.10+, standard library only. No network access, no credentials.

```bash
python3 evaluate.py      # walk-forward report, about 25 s; equals expected_output.txt
python3 -m unittest      # 9 tests, 25-55 s
python3 seed_sweep.py    # six more synthetic leagues, about 2.5 min -> seed_sweep_output.txt
```

| File | Role |
| --- | --- |
| `synth.py` | Seeded synthetic league (20 teams, 4 seasons, Poisson goals, known true means) |
| `elo.py` | Pre-match Elo with Tips8ng's constants (1500 / K 20 / home 60 / scale 400 / 25% season regression) |
| `models.py` | Prior, Elo-logit, Poisson (Maher-style) and the oracle |
| `metrics.py` | Log loss, Brier (3-class sum), RPS, accuracy, ECE (10 bins), paired bootstrap |
| `evaluate.py` | Walk-forward evaluation, refit every 4 rounds, and the report |

The generator is itself a Poisson model, so the Poisson forecaster is the right model family *for
this data*. That says nothing about real football. See Appendix B.6.
