import logging
import pickle

import numpy as np

from neuraldecoding.model.linear_models.KF import KalmanFilter


class ModifiedKalmanFilter(KalmanFilter):
    """
    Kalman filter with the ad-hoc output modification of George et al. 2020
    (J Neurosci Methods 330:108462, sec. 2.8), as used in Page et al. 2018 and
    Thomson et al. 2024 (J Neural Eng).

    Per modified state, in normalized units u = y / output_range (1 = full scale, 0 = rest):

        g        = gain * u
        modified = sign(g) * (|g| - threshold) / (1 - threshold)   if |g| >= threshold
                 = 0                                               otherwise

    The gain is applied before the threshold so that it eases movement initiation; the
    rescale by (1 - threshold) keeps the output continuous at the threshold and lets it still
    reach full scale. The modified output is what forward() / predict() return (mapped back
    to raw units). It is NOT fed back into the filter: the recursion runs on the raw Kalman
    state, clipped per state to +-output_range / gain (the paper's "bounded by the inverse of
    the gain"), so the gained state can never exceed full scale and the filter cannot wind up.

    The state layout is the parent's: [pos_0..pos_n-1, vel_0..vel_n-1, (bias)]. The paper's
    formulas assume 0 = rest, which is true of this rig's velocity states but not of its
    positions (0..1, rest 0.5), hence modify_states defaults to "velocity".

    model_params, in addition to KalmanFilter's:
        threshold (float | list): dead zone in normalized units. Default 0.2 (paper default).
            One value, or one per modified state.
        gain (float | list): multiplicative gain, applied before the threshold. Default 1.0
            (paper default). One value, or one per modified state.
        modify_states (str): which states get the modification and the recursion bound:
            "velocity" (default), "position", "all" or "none". "none" reproduces KalmanFilter
            exactly and is the A/B control.
        output_range (float | list | None): raw value that maps to 1.0 in normalized units,
            one per modified state. None (default) uses the value learned in train_step (the
            range_percentile-th percentile of |label| for every state, saved in the pickle).
            A value given here overrides the pickle, which also lets a plain KalmanFilter
            pickle (no output_range inside) run modified.
        range_percentile (float): percentile of |label| used when output_range is None.
            Default 99.

    Gain and threshold are read from the config only, never from the pickle, so they can be
    retuned (thresholds first, then gains, per the paper) without retraining.
    """

    _MODIFY_STATES = ("velocity", "position", "all", "none")

    def __init__(self, model_params):
        super().__init__(model_params)
        self.threshold = model_params.get("threshold", 0.2)
        self.gain = model_params.get("gain", 1.0)
        self.modify_states = str(model_params.get("modify_states", "velocity")).lower()
        if self.modify_states not in self._MODIFY_STATES:
            raise ValueError(
                f"modify_states must be one of {self._MODIFY_STATES}, got {self.modify_states!r}"
            )
        self.output_range = model_params.get("output_range", None)
        self.range_percentile = float(model_params.get("range_percentile", 99))
        if not (0 < self.range_percentile <= 100):
            raise ValueError(f"range_percentile must be in (0, 100], got {self.range_percentile}")

        # Learned in train_step / read from the pickle: one value per state (all states, so
        # modify_states can be changed later without retraining).
        self._trained_output_range = None
        # Resolved per-state vectors, built lazily once output_size is known.
        self._resolved = False
        self._idx = None
        self._thr = None
        self._gain = None
        self._range = None
        # Raw (unmodified) state of the most recent update, for logging / inspection.
        self.last_raw_yhat = None

    # ------------------------------------------------------------------ training / io

    def train_step(self, input_data):
        x, y = input_data
        super().train_step(input_data)
        if self.is_refit and self._trained_output_range is not None:
            # ReFIT re-estimates C and Q only; keep the full-scale of the model being refit
            # rather than re-deriving it from the rotated (synthetic) intent labels.
            pass
        else:
            self._trained_output_range = np.percentile(
                np.abs(np.asarray(y, dtype=float)), self.range_percentile, axis=0
            )
        self._resolved = False
        self._resolve()

    def save_model(self, fpath):
        # Parent writes the standard KF dict (Model: "KF", so the pickle stays loadable by the
        # plain KalmanFilter). Add our learned full-scale to it.
        super().save_model(fpath)
        with open(fpath, "rb") as file:
            model_dict = pickle.load(file)
        model_dict["output_range"] = (
            None if self._trained_output_range is None
            else np.asarray(self._trained_output_range, dtype=float)
        )
        with open(fpath, "wb") as file:
            pickle.dump(model_dict, file)

    def load_model(self, fpath, running_online=False):
        super().load_model(fpath, running_online=running_online)
        with open(fpath, "rb") as file:
            model_dict = pickle.load(file)
        self._trained_output_range = model_dict.get("output_range", None)
        self._resolved = False
        # Resolve now so a bad config (or a plain-KF pickle without output_range) fails at load
        # time with a clear message, not on the first online bin. A ReFIT load is followed by
        # train_step, which learns/keeps the range, so defer in that case.
        if not self.is_refit:
            self._resolve()

    def reinitialize(self):
        super().reinitialize()
        self.last_raw_yhat = None

    # ------------------------------------------------------------------ hooks

    def _bound_recursive_state(self, yhat: np.ndarray) -> np.ndarray:
        """Raw state clipped to +-output_range / gain on the modified states; this, not the
        modified output, is what the next predict step recurses on."""
        if not self._resolved:
            self._resolve()
        state = yhat.copy()
        if self._idx.size:
            bound = self._range / self._gain
            state[self._idx] = np.clip(state[self._idx], -bound, bound)
        return state

    def _modify_output(self, yhat: np.ndarray) -> np.ndarray:
        """Gain -> threshold -> rescale, in normalized units, on the modified states."""
        if not self._resolved:
            self._resolve()
        self.last_raw_yhat = yhat.copy()
        if not self._idx.size:
            return yhat
        out = yhat.copy()
        g = self._gain * yhat[self._idx] / self._range
        mag = np.abs(g)
        modified = np.where(
            mag >= self._thr,
            np.sign(g) * (mag - self._thr) / (1.0 - self._thr),
            0.0,
        )
        out[self._idx] = modified * self._range
        return out

    # ------------------------------------------------------------------ helpers

    def modified_indices(self) -> np.ndarray:
        """State indices (into the [pos..., vel...] output) that the modification applies to."""
        if not self._resolved:
            self._resolve()
        return self._idx.copy()

    @staticmethod
    def _as_vector(value, k: int, name: str) -> np.ndarray:
        if np.isscalar(value):
            return np.full(k, float(value))
        vec = np.asarray(list(value), dtype=float)
        if vec.shape != (k,):
            raise ValueError(
                f"{name} must be a scalar or a list with one entry per modified state "
                f"({k}), got shape {vec.shape}"
            )
        return vec

    def _resolve(self):
        output_size = getattr(self, "output_size", None)
        if output_size is None:
            raise RuntimeError("ModifiedKalmanFilter: train or load a model before predicting")
        n_pos = output_size // 2
        if self.modify_states == "velocity":
            idx = np.arange(n_pos, output_size)
        elif self.modify_states == "position":
            idx = np.arange(0, n_pos)
        elif self.modify_states == "all":
            idx = np.arange(0, output_size)
        else:
            idx = np.arange(0)
        k = idx.size

        thr = self._as_vector(self.threshold, k, "threshold")
        gain = self._as_vector(self.gain, k, "gain")
        if self.output_range is not None:
            rng = self._as_vector(self.output_range, k, "output_range")
        elif self._trained_output_range is not None:
            trained = np.asarray(self._trained_output_range, dtype=float)
            if trained.shape != (output_size,):
                raise ValueError(
                    f"output_range in the model pickle has shape {trained.shape}, expected "
                    f"({output_size},)"
                )
            rng = trained[idx]
        elif k == 0:
            rng = np.zeros(0)
        else:
            raise ValueError(
                "ModifiedKalmanFilter needs an output_range: either train with this class "
                "(it is learned from the labels and saved in the pickle) or set "
                "model.params.output_range in the config."
            )

        if np.any(thr < 0) or np.any(thr >= 1):
            raise ValueError(f"threshold must be in [0, 1), got {thr}")
        if np.any(gain <= 0):
            raise ValueError(f"gain must be > 0, got {gain}")
        if k and (np.any(~np.isfinite(rng)) or np.any(rng <= 0)):
            raise ValueError(
                f"output_range must be finite and > 0, got {rng}; set model.params.output_range"
            )

        self._idx, self._thr, self._gain, self._range = idx, thr, gain, rng
        self._resolved = True
        if k:
            logging.info(
                f"ModifiedKalmanFilter: modify_states={self.modify_states} idx={idx.tolist()} "
                f"threshold={thr.tolist()} gain={gain.tolist()} output_range={rng.tolist()}"
            )
