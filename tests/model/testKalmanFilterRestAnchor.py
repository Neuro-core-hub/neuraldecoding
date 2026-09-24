import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pickle
import tempfile
import unittest
import warnings
from types import SimpleNamespace

import numpy as np

from neuraldecoding.model.linear_models.KF import KalmanFilter


def one_sided_data(n=4000, seed=0):
    """Two joints moving between random targets; 6 EMG channels that are L-shaped around rest (0.5) in position,
    plus a floor and noise -- the case where a least-squares intercept misses the relaxed EMG."""
    rng = np.random.default_rng(seed)
    p = np.full((n, 2), 0.5); cur = np.array([0.5, 0.5]); tgt = cur.copy()
    for t in range(n):
        if t % 60 == 0:
            j = rng.integers(2); tgt[j] = rng.uniform(0.1, 0.9)
        cur = cur + 0.12 * (tgt - cur); p[t] = cur
    v = np.vstack([np.zeros((1, 2)), np.diff(p, axis=0)])
    up, dn = np.maximum(p - 0.5, 0), np.maximum(0.5 - p, 0)
    x = np.column_stack([10 + 40 * up[:, 0], 10 + 40 * dn[:, 0], 10 + 40 * up[:, 1], 10 + 40 * dn[:, 1],
                         10 + 400 * np.abs(v[:, 0]), 10 + 400 * np.abs(v[:, 1])])
    x = x + rng.normal(0, 1.0, x.shape)
    x_rest = 10 + rng.normal(0, 1.0, (200, 6))
    return x, np.hstack([p, v]), x_rest


def rest_prediction(kf, rest):
    n = kf.output_size // 2
    return kf.C @ np.r_[np.full(n, rest), np.zeros(kf.output_size - n), 1.0]


def drift_at_rest(kf, x_rest, rest=0.5, steps=200):
    """The rig's velocity loop: integrate the decoded velocity, clip to 0-1, feed the position back."""
    kf.running_online = True; kf.reinitialize(); pos = np.full(kf.output_size // 2, rest); kf.set_position(pos)
    for t in range(steps):
        y = kf.forward(x_rest[t % len(x_rest)])
        pos = np.clip(pos + y[kf.output_size // 2:], 0, 1); kf.set_position(pos)
    return np.abs(pos - rest)


class testKalmanFilterRestAnchor(unittest.TestCase):
    def setUp(self):
        self.x, self.y, self.x_rest = one_sided_data()
        self.params = {"append_ones_y": True, "start_y": [0.5, 0.5, 0, 0], "zero_position_uncertainty": True,
                       "rest_anchor": True, "rest_position": [0.5, 0.5]}

    def trained(self, **over):
        kf = KalmanFilter({**self.params, **over}); kf.train_step((self.x, self.y)); return kf

    def test_default_is_off_and_unchanged(self):
        base = KalmanFilter({"append_ones_y": True}); base.train_step((self.x, self.y))
        self.assertFalse(base.rest_anchor)
        self.assertIsNone(base.x_rest_mean)
        other = self.trained()                      # rest_anchor only asks the TRAINER to anchor; train_step alone does not
        np.testing.assert_array_equal(base.C, other.C)

    def test_anchor_makes_rest_predict_relaxed_emg(self):
        kf = self.trained(); C0 = kf.C.copy(); A0, W0, Q0 = kf.A.copy(), kf.W.copy(), kf.Q.copy()
        self.assertGreater(np.abs(rest_prediction(kf, 0.5) - self.x_rest.mean(0)).max(), 0.5)   # the gap exists before
        kf.anchor_rest(self.x_rest)
        np.testing.assert_allclose(rest_prediction(kf, 0.5), self.x_rest.mean(0), atol=1e-9)
        np.testing.assert_array_equal(kf.C[:, :-1], C0[:, :-1])                  # only the intercept moved
        for a, b in ((kf.A, A0), (kf.W, W0), (kf.Q, Q0)):
            np.testing.assert_array_equal(a, b)

    def test_rest_position_scalar_and_per_position(self):
        kf = self.trained(rest_position=0.3); kf.anchor_rest(self.x_rest)
        np.testing.assert_allclose(rest_prediction(kf, 0.3), self.x_rest.mean(0), atol=1e-9)
        kf = self.trained(rest_position=[0.6364, 0.2202]); kf.anchor_rest(self.x_rest)
        n = 2
        z = np.r_[0.6364, 0.2202, np.zeros(n), 1.0]
        np.testing.assert_allclose(kf.C @ z, self.x_rest.mean(0), atol=1e-9)
        with self.assertRaises(ValueError):
            self.trained(rest_position=[0.5, 0.5, 0.5]).anchor_rest(self.x_rest)
        with self.assertRaises(ValueError):
            self.trained(rest_position=None).anchor_rest(self.x_rest)
        with self.assertRaises(ValueError):
            self.trained().anchor_rest(self.x_rest[:, :3])

    def test_anchor_removes_drift_at_rest(self):
        plain = self.trained(); anchored = self.trained(); anchored.anchor_rest(self.x_rest)
        before, after = drift_at_rest(plain, self.x_rest), drift_at_rest(anchored, self.x_rest)
        self.assertGreater(before.max(), 0.05)
        self.assertLess(after.max(), before.max() / 5)

    def test_save_load_and_refit_keep_the_anchor(self):
        kf = self.trained(); kf.anchor_rest(self.x_rest)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "kf.pkl"); kf.save_model(path)
            loaded = KalmanFilter({"append_ones_y": True}); loaded.load_model(path)
            np.testing.assert_allclose(loaded.x_rest_mean, kf.x_rest_mean)
            self.assertEqual(loaded.rest_position, [0.5, 0.5])
            np.testing.assert_allclose(loaded.C, kf.C)
            refit = KalmanFilter({"append_ones_y": True, "is_refit": True}); refit.load_model(path)
            x2, y2, _ = one_sided_data(seed=1)
            refit.train_step((x2, y2))
            self.assertFalse(np.allclose(refit.C[:, :-1], kf.C[:, :-1]))                  # C was refit ...
            np.testing.assert_allclose(rest_prediction(refit, 0.5), self.x_rest.mean(0), atol=1e-9)   # ... and re-anchored

    def test_old_model_files_still_load(self):
        kf = self.trained()
        old = {k: getattr(kf, k) for k in ("A", "C", "W", "Q", "input_size", "output_size")}
        old.update(start_y=np.zeros(4), Model="KF")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "old.pkl")
            with open(path, "wb") as f:
                pickle.dump(old, f)
            loaded = KalmanFilter({"append_ones_y": True}); loaded.load_model(path)
            self.assertIsNone(loaded.x_rest_mean)
            np.testing.assert_allclose(loaded.C, kf.C)


class testRestPeriodBlock(unittest.TestCase):
    def interpipe(self, flags):
        ts = np.arange(1000) * 50.0 + 25.0                                     # 50 ms bins
        return {"trial_rest_period": flags, "neural_ts": ts, "save_keys_ram": [],
                "bin_trial_start_idx": np.array([10, 300, 500]), "bin_trial_end_idx": np.array([210, 400, 600])}

    def test_collects_the_rest_trial_minus_the_settle_time(self):
        from neuraldecoding.preprocessing.blocks import RestPeriodBlock
        X = np.arange(1000 * 3, dtype=float).reshape(1000, 3)
        data, ip = RestPeriodBlock(settle_ms=1000).transform({"neural": X}, self.interpipe(np.array([True, False, False])))
        np.testing.assert_array_equal(ip["rest_neural"], X[30:210])            # 1 s = 20 bins dropped
        self.assertIn("rest_neural", ip["save_keys_ram"])
        np.testing.assert_array_equal(data["neural"], X)                        # training data untouched

    def test_no_rest_period_passes_through(self):
        from neuraldecoding.preprocessing.blocks import RestPeriodBlock
        X = np.zeros((1000, 3))
        for flags in (None, np.array([False, False, False])):
            _, ip = RestPeriodBlock().transform({"neural": X}, self.interpipe(flags))
            self.assertNotIn("rest_neural", ip)


class testLinearTrainerAnchor(unittest.TestCase):
    def setUp(self):
        self.x, self.y, self.x_rest = one_sided_data()

    def model(self, anchor):
        kf = KalmanFilter({"append_ones_y": True, "rest_anchor": anchor, "rest_position": 0.5}); kf.train_step((self.x, self.y)); return kf

    def test_on_off_and_missing_rest(self):
        from neuraldecoding.trainer.LinearTrainer import LinearTrainer
        with_rest = SimpleNamespace(saved_data={"rest_neural": self.x_rest})
        on = self.model(True); LinearTrainer.anchor_rest(SimpleNamespace(model=on, preprocessor=with_rest))
        np.testing.assert_allclose(rest_prediction(on, 0.5), self.x_rest.mean(0), atol=1e-9)
        off = self.model(False); C0 = off.C.copy(); LinearTrainer.anchor_rest(SimpleNamespace(model=off, preprocessor=with_rest))
        np.testing.assert_array_equal(off.C, C0)
        with self.assertRaises(ValueError):
            LinearTrainer.anchor_rest(SimpleNamespace(model=self.model(True), preprocessor=SimpleNamespace(saved_data={})))


if __name__ == "__main__":
    unittest.main()
