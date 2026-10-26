"""Posh direct myoelectric control for the knee alone (Posh, Schmiedeler & Wensing, TNSRE 2023, Eq. 1).

The fit is the knee half of decoder_configs/posh_dmc/posh_dmc_fit.py (convention A), trained from
labelled bins instead of one run per movement, so the GUI's 9 Train can fit it from the cued runs:

  1. rest_ref = mean MAV of the rest bins
  2. MVA[c] = the larger of (mean flexion bins - rest_ref) and (mean extension bins - rest_ref)
  3. v = max((x - rest_ref) / MVA, 0)
  4. NNLS weights w_f, w_e >= 0 on the movement bins: target 1 on flexion (extension) bins, 0 on
     the others, including the zero classes (ankle-only movements, knee at rest)
  5. coef_ = [w_f; w_e] / MVA, intercept 0: the deployed linear map u = x @ coef_.T -> [u_f, u_e]
  6. Eq. 1 scalars on the deployed map: ray angles th_ag (flexion) and th_an (extension), th0 their
     bisector, r_cal the 95th percentile radius of the movement bins, g0 = mean + 2 SD of the rest
     radius / r_cal (the deadband)

Online, generic_decoder runs the linear map and the posh_dmc node runs Eq. 1 with these scalars.
The pickle is a plain sklearn LinearRegression (coef_ 2 x channels) carrying the scalars as
`posh_scalars`; save_model also writes them to <model>_scalars.json, which posh_dmc reads.
"""
import json
import os
import pickle
from typing import Any

import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from .Regression import Regression

EPS = 1e-9


def slope_angle(y, x):
    """Ray angle = arctan of the through-origin least-squares slope of y on x (posh_dmc_fit.py)."""
    denom = float(np.sum(x * x))
    slope = float(np.sum(x * y)) / denom if denom > EPS else 0.0
    return float(np.arctan(slope))


def eq1(a, b, p):
    """Posh Eq. 1 for one DOF, vectorised; identical to posh_dmc_fit.eq1_forward and the posh_dmc
    node. a = agonist (u_f), b = antagonist (u_e), p = {th_ag, th_an, th0, r_cal, g0}.
    Returns (u in [-1, 1], cocontraction c >= 0); u > 0 is flexion."""
    a = np.maximum(np.asarray(a, dtype=np.float64), 0.0)
    b = np.maximum(np.asarray(b, dtype=np.float64), 0.0)
    r = np.maximum(np.hypot(a, b) / p["r_cal"] - p["g0"], 0.0) / (1.0 - p["g0"])
    th = np.arctan2(a, b)
    num = th - p["th0"]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(th >= p["th0"], r * num / (p["th_ag"] - p["th0"]), -r * num / (p["th_an"] - p["th0"]))
    u = np.clip(np.nan_to_num(u), -1.0, 1.0)
    return u, np.maximum(r - np.abs(u), 0.0)


def fit_posh_knee(X, labels, flexion_classes, extension_classes, rest_classes, zero_classes,
                  r_cal_percentile=95.0, g0_sd=2.0):
    """X: (bins, channels) 200 ms MAV; labels: class name per bin. Returns (coef_ (2, channels), scalars)."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    flex = np.isin(labels, list(flexion_classes))
    ext = np.isin(labels, list(extension_classes))
    rest = np.isin(labels, list(rest_classes))
    zero = np.isin(labels, list(zero_classes))
    for name, m in (("flexion", flex), ("extension", ext), ("rest", rest)):
        if not m.any():
            raise ValueError(f"Posh knee fit: no {name} bins (check nwb_files and the class lists; "
                             f"classes present: {sorted(set(labels.tolist()))})")

    rest_ref = X[rest].mean(axis=0)
    MVA = np.clip(np.maximum(X[flex].mean(axis=0) - rest_ref, X[ext].mean(axis=0) - rest_ref), EPS, None)
    V = np.maximum((X - rest_ref) / MVA, 0.0)
    mov = flex | ext | zero
    w_f, _ = nnls(V[mov], flex[mov].astype(float))
    w_e, _ = nnls(V[mov], ext[mov].astype(float))
    coef = np.vstack([w_f, w_e]) / MVA[None, :]

    U = np.maximum(X @ coef.T, 0.0)                       # the deployed, rectified pair (u_f, u_e)
    th_ag = slope_angle(U[flex, 0], U[flex, 1])
    th_an = slope_angle(U[ext, 0], U[ext, 1])
    th0 = 0.5 * (th_ag + th_an)                           # direction gate: the bisector, not rest
    r_cal = max(float(np.percentile(np.hypot(U[flex | ext, 0], U[flex | ext, 1]), r_cal_percentile)), EPS)
    ratio = np.hypot(U[rest, 0], U[rest, 1]) / r_cal
    g0 = float(np.clip(np.mean(ratio) + g0_sd * np.std(ratio), 0.0, 0.95))   # radius gate (deadband)
    knee = {"th_ag": th_ag, "th_an": th_an, "th0": th0, "r_cal": r_cal, "g0": g0}
    scalars = {
        "convention": "A: coef_ folds 1/MVA, rest_ref dropped online (g0 deadband); knee only",
        "output_columns": ["u_f", "u_e"],
        "rest_ref": rest_ref.tolist(),
        "MVA": MVA.tolist(),
        "weights": {"w_f": w_f.tolist(), "w_e": w_e.tolist()},
        "classes": {"flexion": list(flexion_classes), "extension": list(extension_classes),
                    "rest": list(rest_classes), "zero": list(zero_classes)},
        "bins": {"flexion": int(flex.sum()), "extension": int(ext.sum()), "rest": int(rest.sum()),
                 "zero": int(zero.sum())},
        "knee": knee,
    }
    return coef, scalars


def scalars_path(model_path):
    return os.path.splitext(model_path)[0] + "_scalars.json"


class PoshKnee(Regression):
    """The knee's Posh linear map (u_f, u_e) plus its Eq. 1 scalars. Trained by PoshKneeTrainer;
    online it loads like any Regression pickle (generic_decoder's LinearDecoder)."""

    def __init__(self, params={}):
        super().__init__(params)
        self.model = SklearnLinearRegression(fit_intercept=False)
        self.class_names = None          # set by PoshKneeTrainer from the patterns' class list
        self.scalars = None

    def _opt(self, key, default):
        v = self.params.get(key, default) if self.params is not None else default
        return list(v) if isinstance(default, list) else v

    def labels(self, y):
        if self.class_names is None:
            raise ValueError("PoshKnee needs the class names (PoshKneeTrainer sets them)")
        return np.asarray(self.class_names)[np.rint(np.asarray(y, dtype=float)).astype(int).ravel()]

    def train_step(self, input_data: Any) -> None:
        X, y = input_data
        coef, self.scalars = fit_posh_knee(
            np.asarray(X), self.labels(y),
            self._opt("flexion_classes", ["knee_flexion"]),
            self._opt("extension_classes", ["knee_extension"]),
            self._opt("rest_classes", ["rest"]),
            self._opt("zero_classes", ["plantar_flexion", "dorsal_flexion"]),
            float(self._opt("r_cal_percentile", 95.0)), float(self._opt("g0_sd", 2.0)))
        self.model.coef_ = coef
        self.model.intercept_ = np.zeros(coef.shape[0])
        self.model.n_features_in_ = coef.shape[1]
        self.model.posh_scalars = self.scalars

    def knee_u(self, X):
        """Eq. 1 knee command u in [-1, 1] for 200 ms MAV bins (offline checks)."""
        U = np.asarray(X, dtype=np.float64) @ self.model.coef_.T
        return eq1(U[:, 0], U[:, 1], self.scalars["knee"])[0]

    def save_model(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scalars_path(filepath), 'w') as f:
            json.dump(self.scalars, f, indent=2)

    def load_model(self, fpath: str, running_online: bool = False) -> None:
        with open(fpath, 'rb') as f:
            self.model = pickle.load(f)
        self.scalars = getattr(self.model, "posh_scalars", None)
