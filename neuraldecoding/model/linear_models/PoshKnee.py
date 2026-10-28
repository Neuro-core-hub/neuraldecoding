"""Posh direct myoelectric control for the knee alone (Posh, Schmiedeler & Wensing, TNSRE 2023, Eq. 1).

The fit is the knee half of decoder_configs/posh_dmc/posh_dmc_fit.py (convention A), trained from
labelled bins instead of one run per movement, so the GUI's 9 Train can fit it from the cued runs:

  1. rest_ref = mean MAV of the rest bins
  2. MVA[c] = the larger of (mean flexion bins - rest_ref) and (mean extension bins - rest_ref)
  3. v = max((x - rest_ref) / MVA, 0)
  4. NNLS weights w_f, w_e >= 0 on the movement bins: target 1 on flexion (extension) bins, 0 on
     the others, including the zero classes (ankle-only movements, knee at rest). Optionally
     restricted per direction (flexion_channels / extension_channels, e.g. Biceps Femoris only
     for flexion and Rectus Femoris only for extension); other weights stay 0
  5. coef_ = [w_f; w_e] / MVA (+ per-direction scale x MVA); intercept_ 0, or with rest_subtraction
     -(rest mean + 2 SD) per direction: the deployed map u = x @ coef_.T + intercept_ -> [u_f, u_e]
  6. Eq. 1 scalars on the deployed map: ray angles th_ag (flexion) and th_an (extension) from the
     flexion / extension holds (the co-contraction slopes), th0 their bisector. Normalization: with
     mva_scale(_flexion/_extension) each direction is divided by scale x its MVA (99th percentile over
     its own max holds; Posh's per-muscle MVA, hand-lowered) and r_cal = 1; else r_cal = the
     r_cal_percentile (95th) radius of the movement bins. g0 = mean + 2 SD of the rest radius / r_cal

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


def ray_angle(y, x):
    """Ray angle of the principal direction through the origin (total least squares): symmetric in
    x and y, so points ON an axis give 0 or 90 deg (slope_angle, an OLS slope of y on x, returns 0
    when x is all zero -- e.g. a rest-subtracted flexion cloud with the extension signal at 0)."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    return float(0.5 * np.arctan2(2.0 * np.sum(x * y), np.sum(x * x) - np.sum(y * y)))


def eq1(a, b, p):
    """Posh Eq. 1 for one DOF, vectorised; identical to the posh_dmc node. a = agonist (u_f),
    b = antagonist (u_e), p = {th_ag, th_an, th0, r_cal, g0, direction, clipping}.
    u = size r x direction factor f; f linear in angle (default) or slope (p["direction"]).
    p["clipping"]: "paper" = r capped at 1 and f clamped to [-1, 1] before multiplying (Posh: pushing
    harder cannot make up for co-contraction); "final" (no key: models trained before the option)
    = only u clipped. Returns (u in [-1, 1], cocontraction c >= 0); u > 0 is flexion."""
    a = np.maximum(np.asarray(a, dtype=np.float64), 0.0)
    b = np.maximum(np.asarray(b, dtype=np.float64), 0.0)
    r = np.maximum(np.hypot(a, b) / p["r_cal"] - p["g0"], 0.0) / (1.0 - p["g0"])
    th = np.arctan2(a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        if p.get("direction", "angle") == "slope":
            # Posh TNSRE 2023 Eq. 1 as printed: fraction of the SLOPE gap, m = a/b clamped to the rays
            m_ag, m_an, m0 = np.tan(p["th_ag"]), np.tan(p["th_an"]), np.tan(p["th0"])
            m = np.clip(np.where(b > 0, a / np.where(b > 0, b, 1.0), m_ag), m_an, m_ag)
            f = np.where(m >= m0, (m - m0) / (m_ag - m0), -(m - m0) / (m_an - m0))
            f = np.where((a > 0) | (b > 0), f, 0.0)
        else:
            # default: the same fraction measured in ANGLE (symmetric about the bisector)
            num = th - p["th0"]
            f = np.where(th >= p["th0"], num / (p["th_ag"] - p["th0"]), -num / (p["th_an"] - p["th0"]))
    f = np.nan_to_num(f)
    if p.get("clipping", "final") == "paper":
        r = np.minimum(r, 1.0)
        f = np.clip(f, -1.0, 1.0)
    u = np.clip(r * f, -1.0, 1.0)
    return u, np.maximum(r - np.abs(u), 0.0)


def fit_posh_knee(X, labels, flexion_classes, extension_classes, rest_classes, zero_classes,
                  r_cal_percentile=95.0, g0_sd=2.0, flexion_cols=None, extension_cols=None,
                  mva_scale=None, channel_ids=None, direction="angle",
                  mva_scale_flexion=None, mva_scale_extension=None, mva_percentile=99.0,
                  rest_subtraction=False, clipping="final"):
    """X: (bins, channels) 200 ms MAV; labels: class name per bin. Returns (coef_ (2, channels), scalars).

    flexion_cols / extension_cols: column positions in X allowed to drive each direction (the other
    weights stay 0); None = all columns, the NNLS chooses.
    MVA normalization (Posh: each muscle / its MVA, hand-lowered per muscle): the training holds are
    maximal contractions. Each direction's combined signal is divided by scale x its MVA, MVA = the
    mva_percentile (99th: near-peak, one spike cannot set it) of that signal over its own max holds;
    scale = mva_scale_flexion / mva_scale_extension (each falls back to mva_scale; 0.9 = full output at
    90 % of max). Then 1 = full output in each direction (r_cal = 1). No scale set at all = the old
    rule: r_cal = the r_cal_percentile radius of the movement bins. channel_ids: for the record only.
    rest_subtraction (Posh: "the average EMG ... at rest was set as a volitional input baseline of 0
    for each muscle, and activity within 2 SD of this level was ignored"): per direction, the
    offset rest mean + g0_sd x rest SD is subtracted (as intercept_; the online rectifier / 0-1 clip
    then reads anything inside that band as 0) and the combined-radius deadband g0 becomes 0. False =
    no offset, g0 = mean + g0_sd SD of the combined rest radius (the July notebook convention).
    Returns coef_ and scalars; scalars["intercept"] is the deployed intercept_.
    """
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
    n_ch = X.shape[1]

    def nnls_on(cols, target):
        # only `cols` may carry weight for this direction (None = every column)
        cols = list(range(n_ch)) if cols is None else [int(c) for c in cols]
        w = np.zeros(n_ch)
        if cols:
            w[cols], _ = nnls(V[mov][:, cols], target)
        return w

    w_f = nnls_on(flexion_cols, flex[mov].astype(float))
    w_e = nnls_on(extension_cols, ext[mov].astype(float))
    coef = np.vstack([w_f, w_e]) / MVA[None, :]

    # per-direction rest offset (0 = none): rest mean + g0_sd x rest SD of each direction's signal
    R = X[rest] @ coef.T
    offset = (R.mean(axis=0) + g0_sd * R.std(axis=0)) if rest_subtraction else np.zeros(2)

    s_f = mva_scale_flexion if mva_scale_flexion is not None else mva_scale
    s_e = mva_scale_extension if mva_scale_extension is not None else mva_scale
    if s_f is not None or s_e is not None:
        # per-direction MVA, folded into coef_ so the online map already reads 1 at scale x MVA
        s_f = 1.0 if s_f is None else float(s_f)
        s_e = 1.0 if s_e is None else float(s_e)
        U0 = np.maximum(X @ coef.T - offset, 0.0)
        # nearest actual bin at or below the percentile (no interpolation): with < 100 bins that is
        # below the top bin, so a single spike can never set the MVA
        mva_f = max(float(np.percentile(U0[flex, 0], mva_percentile, method="lower")), EPS)
        mva_e = max(float(np.percentile(U0[ext, 1], mva_percentile, method="lower")), EPS)
        div = np.array([s_f * mva_f, s_e * mva_e])
        coef = coef / div[:, None]
        offset = offset / div
        r_cal = 1.0
        normalization = {"mode": "mva_per_direction", "mva_percentile": float(mva_percentile),
                         "mva_scale_flexion": s_f, "mva_scale_extension": s_e,
                         "mva_flexion": mva_f, "mva_extension": mva_e}
    else:
        r_cal = None
        normalization = {"mode": "percentile", "r_cal_percentile": float(r_cal_percentile)}

    U = np.maximum(X @ coef.T - offset, 0.0)              # the deployed, rectified pair (u_f, u_e)
    # co-contraction rays from the same (maximal) flexion / extension holds, in the normalized space
    ray = ray_angle if rest_subtraction else slope_angle      # the old OLS rays kept bit-identical
    th_ag = ray(U[flex, 0], U[flex, 1])
    th_an = ray(U[ext, 0], U[ext, 1])
    th0 = 0.5 * (th_ag + th_an)                           # direction gate: the bisector, not rest
    if r_cal is None:
        radius = np.hypot(U[flex | ext, 0], U[flex | ext, 1])
        r_cal = max(float(np.percentile(radius, r_cal_percentile)), EPS)
    ratio = np.hypot(U[rest, 0], U[rest, 1]) / r_cal
    if rest_subtraction:
        g0 = 0.0                                          # the deadband is the per-direction band above
    else:
        g0 = float(np.clip(np.mean(ratio) + g0_sd * np.std(ratio), 0.0, 0.95))   # radius gate (deadband)
    knee = {"th_ag": th_ag, "th_an": th_an, "th0": th0, "r_cal": r_cal, "g0": g0, "direction": direction,
            "clipping": clipping}
    scalars = {
        "convention": ("rest-subtracted: intercept_ = -(rest mean + g0_sd SD) per direction, g0 = 0"
                       if rest_subtraction else
                       "A: coef_ folds 1/MVA, rest_ref dropped online (g0 deadband)") + "; knee only",
        "intercept": (-offset).tolist(),
        "output_columns": ["u_f", "u_e"],
        "rest_ref": rest_ref.tolist(),
        "MVA": MVA.tolist(),
        "weights": {"w_f": w_f.tolist(), "w_e": w_e.tolist()},
        "channel_ids": None if channel_ids is None else [int(c) for c in channel_ids],
        "assigned_columns": {"flexion": None if flexion_cols is None else [int(c) for c in flexion_cols],
                             "extension": None if extension_cols is None else [int(c) for c in extension_cols]},
        "normalization": normalization,
        "classes": {"flexion": list(flexion_classes), "extension": list(extension_classes),
                    "rest": list(rest_classes), "zero": list(zero_classes)},
        "bins": {"flexion": int(flex.sum()), "extension": int(ext.sum()), "rest": int(rest.sum()),
                 "zero": int(zero.sum())},
        "knee": knee,
    }
    return coef, scalars


def posh_fit_options(params):
    """fit_posh_knee keyword arguments from a config's model.params (dict or DictConfig).

    flexion_channels / extension_channels are neural column ids, e.g. [8] / [14]; they must be among
    `channels` (the config sets it to the selected indices) and become positions in X. Unset = the
    fit chooses from every selected channel. mva_scale unset = the r_cal_percentile rule.
    """
    get = (lambda k, d=None: params.get(k, d)) if params is not None else (lambda k, d=None: d)
    as_list = lambda v: None if v is None else [x for x in v]
    channels = as_list(get("channels"))

    def cols(key):
        ids = as_list(get(key))
        if ids is None:
            return None
        if channels is None:
            raise ValueError(f"PoshKnee: {key} needs model.params.channels (the selected column ids)")
        missing = [c for c in ids if c not in channels]
        if missing:
            raise ValueError(f"PoshKnee: {key} {missing} not among the selected channels {channels}")
        return [channels.index(c) for c in ids]

    scale = get("mva_scale")
    sf, se = get("mva_scale_flexion"), get("mva_scale_extension")
    clipping = str(get("clipping", "final"))
    if clipping not in ("paper", "final"):
        raise ValueError(f"PoshKnee: clipping must be 'paper' or 'final', got {clipping!r}")
    direction = str(get("direction", "angle"))
    if direction not in ("angle", "slope"):
        raise ValueError(f"PoshKnee: direction must be 'angle' or 'slope', got {direction!r}")
    return dict(flexion_classes=as_list(get("flexion_classes", ["knee_flexion"])),
                extension_classes=as_list(get("extension_classes", ["knee_extension"])),
                rest_classes=as_list(get("rest_classes", ["rest"])),
                zero_classes=as_list(get("zero_classes", ["plantar_flexion", "dorsal_flexion"])),
                r_cal_percentile=float(get("r_cal_percentile", 95.0)), g0_sd=float(get("g0_sd", 2.0)),
                flexion_cols=cols("flexion_channels"), extension_cols=cols("extension_channels"),
                mva_scale=None if scale is None else float(scale), channel_ids=channels,
                direction=direction,
                mva_scale_flexion=None if sf is None else float(sf),
                mva_scale_extension=None if se is None else float(se),
                mva_percentile=float(get("mva_percentile", 99.0)),
                rest_subtraction=bool(get("rest_subtraction", False)),
                clipping=clipping)


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
        coef, self.scalars = fit_posh_knee(np.asarray(X), self.labels(y), **posh_fit_options(self.params))
        self.model.coef_ = coef
        self.model.intercept_ = np.asarray(self.scalars.get("intercept", np.zeros(coef.shape[0])), dtype=np.float64)
        self.model.n_features_in_ = coef.shape[1]
        self.model.posh_scalars = self.scalars

    def knee_u(self, X):
        """Eq. 1 knee command u in [-1, 1] for 200 ms MAV bins (offline checks)."""
        U = np.asarray(X, dtype=np.float64) @ self.model.coef_.T + self.model.intercept_
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
