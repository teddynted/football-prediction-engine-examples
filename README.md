# Football prediction engine: examples

Companion code for the book *Inside the Football Prediction Engine: A Practical Machine-Learning Case Study Using
Tips8ng*, by Teddy Kekana.

| Folder | What it is | Book section |
| --- | --- | --- |
| [`mini_engine/`](mini_engine/) | A runnable mini prediction engine on a **synthetic** league: Elo, Poisson, a prior and an oracle, walk-forward evaluation, log loss / Brier / RPS / ECE, and paired bootstrap intervals | Appendix B |

Everything here runs on Python 3.10+ with the standard library only. No data downloads, no API keys, no network.
All numbers come from a made-up league; none is a Tips8ng result.

```bash
cd mini_engine
python3 evaluate.py      # matches expected_output.txt
python3 -m unittest      # 9 tests
```

Licence: MIT (see `LICENSE`).
