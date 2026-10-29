"""python3 -m unittest   (about 30 s: the last test re-runs the full evaluation)"""
import math
import pathlib
import unittest

import elo
import metrics as M
from models import grid_outcome
from synth import Match, generate

U = [1 / 3, 1 / 3, 1 / 3]


class Metrics(unittest.TestCase):
    def test_uniform_reference_values(self):
        for y in range(3):
            self.assertAlmostEqual(M.log_loss(U, y), math.log(3))
            self.assertAlmostEqual(M.brier(U, y), 2 / 3)
        self.assertAlmostEqual(M.rps(U, 0), 5 / 18)   # home win
        self.assertAlmostEqual(M.rps(U, 1), 1 / 9)    # draw
        self.assertAlmostEqual(M.rps(U, 2), 5 / 18)   # away win

    def test_perfect_forecast_scores_zero(self):
        self.assertAlmostEqual(M.log_loss([1, 0, 0], 0), 0.0)
        self.assertAlmostEqual(M.brier([0, 1, 0], 1), 0.0)
        self.assertAlmostEqual(M.rps([0, 0, 1], 2), 0.0)

    def test_rps_punishes_the_far_side_more(self):
        p = [0.8, 0.1, 0.1]
        self.assertGreater(M.rps(p, 2), M.rps(p, 1))


class Elo(unittest.TestCase):
    def test_pre_match_rating_excludes_own_result(self):
        m = Match(1, 1, "A", "B", 5, 0, 1.0, 1.0)
        self.assertEqual(elo.pre_match_ratings([m]), [(1500.0, 1500.0)])

    def test_update_is_zero_sum_and_uses_home_advantage(self):
        ms = [Match(1, 1, "A", "B", 1, 1, 1.0, 1.0), Match(1, 2, "A", "B", 0, 0, 1, 1)]
        (h, a) = elo.pre_match_ratings(ms)[1]
        self.assertAlmostEqual(h + a, 3000.0)
        self.assertLess(h, 1500.0)   # a home draw is below the home expectation

    def test_season_regression(self):
        ms = [Match(1, 1, "A", "B", 3, 0, 1, 1), Match(2, 1, "A", "B", 0, 0, 1, 1)]
        r1 = 1500 + 20 * (1 - elo.expected_home(1500, 1500))
        self.assertAlmostEqual(elo.pre_match_ratings(ms)[1][0],
                               1500 + 0.75 * (r1 - 1500))


class Models(unittest.TestCase):
    def test_grid_is_a_distribution(self):
        p = grid_outcome(1.5, 1.1)
        self.assertAlmostEqual(sum(p), 1.0)
        self.assertGreater(p[0], p[2])

    def test_generator_is_deterministic(self):
        self.assertEqual(generate()[:5], generate()[:5])
        self.assertEqual(len(generate()), 4 * 380)


class Golden(unittest.TestCase):
    def test_report_matches_expected_output(self):
        import evaluate
        here = pathlib.Path(__file__).parent
        expected = (here / "expected_output.txt").read_text().rstrip("\n")
        self.assertEqual(evaluate.main(), expected)


if __name__ == "__main__":
    unittest.main()
