import os
import sys
import tempfile
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np

from neuraldecoding.model.linear_models import KalmanFilter, ModifiedKalmanFilter


def make_data(seed=0, T=3000, n_dof=2, n_ch=12):
    """Smooth 2-DOF kinematics [pos, pos, vel, vel] and linear-Gaussian observations."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((T, n_dof))
    k = np.ones(25) / 25
    v = np.column_stack([np.convolve(v[:, i], k, mode="same") for i in range(n_dof)])
    v *= 0.05 / np.abs(v).max()  # full-scale velocity ~0.05 / bin, like the rig
    p = 0.5 + np.cumsum(v, axis=0)
    y = np.hstack([p, v])
    C = rng.standard_normal((n_ch, 2 * n_dof + 1))
    x = np.hstack([y, np.ones((T, 1))]) @ C.T + 0.3 * rng.standard_normal((T, n_ch))
    return x, y


class testModifiedKF(unittest.TestCase):
    def setUp(self):
        self.x, self.y = make_data()
        self.x_train, self.y_train = self.x[:2500], self.y[:2500]
        self.x_test = self.x[2500:]
        self.n_out = self.y.shape[1]

    def _train(self, cls, **params):
        model = cls({"zero_position_uncertainty": False, **params})
        model.train_step((self.x_train, self.y_train))
        return model

    # ------------------------------------------------------------------ identity / A-B control

    def test_none_matches_plain_kf(self):
        kf = self._train(KalmanFilter)
        mkf = self._train(ModifiedKalmanFilter, modify_states="none", threshold=0.2, gain=3.0)
        np.testing.assert_allclose(mkf(self.x_test), kf(self.x_test))
        # stepwise (online path) as well
        kf.reinitialize(); mkf.reinitialize()
        for t in range(50):
            np.testing.assert_allclose(mkf(self.x_test[t]), kf(self.x_test[t]))

    def test_threshold_zero_gain_one_is_identity_on_output(self):
        mkf = self._train(ModifiedKalmanFilter, threshold=0.0, gain=1.0)
        row = np.array([0.4, 0.6, 0.01, -0.03, 1.0])
        np.testing.assert_allclose(mkf._modify_output(row), row)

    # ------------------------------------------------------------------ the formula

    def test_static_formula_and_untouched_states(self):
        mkf = self._train(ModifiedKalmanFilter, threshold=0.2, gain=1.0)
        idx = mkf.modified_indices()
        np.testing.assert_array_equal(idx, [2, 3])  # velocity half of [p, p, v, v]
        r = mkf._range
        self.assertEqual(r.shape, (2,))
        self.assertTrue(np.all(r > 0))
        # below threshold -> 0; above -> (|u| - thr) / (1 - thr), sign preserved
        row = np.array([0.4, 0.6, 0.1 * r[0], -0.5 * r[1], 1.0])
        out = mkf._modify_output(row)
        self.assertEqual(out[2], 0.0)
        self.assertAlmostEqual(out[3], -(0.5 - 0.2) / 0.8 * r[1])
        # positions and bias untouched
        np.testing.assert_allclose(out[[0, 1, 4]], row[[0, 1, 4]])
        # raw state recorded
        np.testing.assert_allclose(mkf.last_raw_yhat, row)
        # exactly at threshold -> 0 (continuous), full scale -> full scale
        row = np.array([0.0, 0.0, 0.2 * r[0], 1.0 * r[1], 1.0])
        out = mkf._modify_output(row)
        self.assertAlmostEqual(out[2], 0.0)
        self.assertAlmostEqual(out[3], r[1])

    def test_gain_applied_before_threshold(self):
        r_fixed = [0.05, 0.05]
        low = self._train(ModifiedKalmanFilter, threshold=0.2, gain=1.0, output_range=r_fixed)
        high = self._train(ModifiedKalmanFilter, threshold=0.2, gain=2.0, output_range=r_fixed)
        row = np.array([0.0, 0.0, 0.15 * 0.05, 0.0, 1.0])  # 0.15 of full scale, below 0.2
        self.assertEqual(low._modify_output(row)[2], 0.0)
        # gain 2 -> 0.30 of full scale -> (0.30 - 0.2) / 0.8 = 0.125 of full scale
        self.assertAlmostEqual(high._modify_output(row)[2], 0.125 * 0.05)

    def test_recursive_state_bounded_by_inverse_gain(self):
        r_fixed = [0.05, 0.05]
        mkf = self._train(ModifiedKalmanFilter, threshold=0.2, gain=4.0, output_range=r_fixed)
        row = np.array([0.4, 0.6, 0.04, -0.04, 1.0])  # |v| = 0.8 fs > 1/gain = 0.25 fs
        state = mkf._bound_recursive_state(row)
        np.testing.assert_allclose(state[2:4], [0.05 / 4, -0.05 / 4])
        np.testing.assert_allclose(state[[0, 1, 4]], row[[0, 1, 4]])
        self.assertIsNot(state, row)
        # and through a real predict on strong input the recursion never exceeds the bound
        mkf.reinitialize()
        mkf.predict(self.x_test * 20)
        self.assertTrue(np.all(np.abs(mkf.last_yhat[2:4]) <= 0.05 / 4 + 1e-12))

    def test_dead_zone_zeroes_small_velocities(self):
        thr = self._train(ModifiedKalmanFilter, threshold=0.2, gain=1.0)
        nothr = self._train(ModifiedKalmanFilter, threshold=0.0, gain=1.0)
        out_thr = thr(self.x_test)
        out_nothr = nothr(self.x_test)
        self.assertGreater(np.mean(out_thr[:, 2:] == 0.0), 0.05)
        self.assertEqual(np.mean(out_nothr[:, 2:] == 0.0), 0.0)
        # positions are never modified (velocity mode): identical up to the recursion bound
        self.assertEqual(out_thr.shape, out_nothr.shape)

    # ------------------------------------------------------------------ online == batch

    def test_online_stepwise_equals_batch(self):
        mkf = self._train(ModifiedKalmanFilter, threshold=0.2, gain=1.5)
        batch = mkf(self.x_test[:200])
        mkf.reinitialize()
        step = np.stack([mkf(self.x_test[t]) for t in range(200)])
        np.testing.assert_allclose(step, batch)

    # ------------------------------------------------------------------ io

    def test_save_load_roundtrip_and_cross_class(self):
        mkf = self._train(ModifiedKalmanFilter, threshold=0.2, gain=1.0)
        expected = mkf(self.x_test)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mkf.pkl")
            mkf.save_model(path)
            loaded = ModifiedKalmanFilter({"threshold": 0.2, "gain": 1.0})
            loaded.load_model(path)
            np.testing.assert_allclose(loaded._trained_output_range, mkf._trained_output_range)
            np.testing.assert_allclose(loaded(self.x_test), expected)
            # same pickle loads into the plain KF (Model tag stays "KF")
            plain = KalmanFilter({})
            plain.load_model(path)
            self.assertEqual(plain(self.x_test).shape, expected.shape)

    def test_plain_kf_pickle_needs_output_range(self):
        kf = self._train(KalmanFilter)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "kf.pkl")
            kf.save_model(path)
            mkf = ModifiedKalmanFilter({"threshold": 0.2})
            with self.assertRaises(ValueError):
                mkf.load_model(path)  # fails at load, not on the first online bin
            mkf = ModifiedKalmanFilter({"threshold": 0.2, "output_range": [0.05, 0.05]})
            mkf.load_model(path)
            self.assertEqual(mkf(self.x_test[:5]).shape, (5, self.n_out))

    def test_config_output_range_overrides_pickle(self):
        mkf = self._train(ModifiedKalmanFilter, output_range=[0.01, 0.02])
        np.testing.assert_allclose(mkf._range, [0.01, 0.02])
        self.assertIsNotNone(mkf._trained_output_range)

    # ------------------------------------------------------------------ parameter handling

    def test_per_dof_vectors_and_validation(self):
        mkf = self._train(ModifiedKalmanFilter, threshold=[0.2, 0.4], gain=[1.0, 2.0])
        np.testing.assert_allclose(mkf._thr, [0.2, 0.4])
        np.testing.assert_allclose(mkf._gain, [1.0, 2.0])
        with self.assertRaises(ValueError):
            self._train(ModifiedKalmanFilter, threshold=[0.2, 0.2, 0.2])(self.x_test[:2])
        with self.assertRaises(ValueError):
            self._train(ModifiedKalmanFilter, threshold=1.0)(self.x_test[:2])
        with self.assertRaises(ValueError):
            self._train(ModifiedKalmanFilter, gain=0.0)(self.x_test[:2])
        with self.assertRaises(ValueError):
            ModifiedKalmanFilter({"modify_states": "bogus"})

    def test_modify_states_selects_indices(self):
        self.assertListEqual(self._train(ModifiedKalmanFilter, modify_states="position").modified_indices().tolist(), [0, 1])
        self.assertListEqual(self._train(ModifiedKalmanFilter, modify_states="all").modified_indices().tolist(), [0, 1, 2, 3])
        self.assertListEqual(self._train(ModifiedKalmanFilter, modify_states="none").modified_indices().tolist(), [])

    def test_learned_range_is_percentile_of_labels(self):
        mkf = self._train(ModifiedKalmanFilter, range_percentile=99)
        expected = np.percentile(np.abs(self.y_train), 99, axis=0)
        np.testing.assert_allclose(mkf._trained_output_range, expected)
        np.testing.assert_allclose(mkf._range, expected[2:])


if __name__ == "__main__":
    unittest.main()
