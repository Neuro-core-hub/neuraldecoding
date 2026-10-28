import os
import numpy as np
import pickle
import torch
from typing import Any
from .LinearModel import LinearModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#: Keys consumed by the proportional-speed extension rather than by sklearn.
_PROPORTIONAL_KEYS = (
    "proportional",
    "rest_class",
    "amp_n_features",
    "amp_scale_pct",
    "amp_rest_pct",
    "amp_dead_floor",
)


class LDA(LinearModel):
    def __init__(self, params = {}):
        """
        Linear Discriminant Analysis (LDA) model for classification tasks.

        Optionally estimates a per-class amplitude scale alongside the
        classifier, so the decoder can drive grip SPEED proportionally to how
        hard the user contracts (Scheme et al. 2014, motion-normalised
        proportional control). The classifier picks the grip; the amplitude
        picks the speed. Both come from the same features, computed
        independently.

        Args:
            params (dict): Parameters for the LDA model
                (sklearn.LinearDiscriminantAnalysis), plus the optional
                proportional-control keys below. priors may also be "equal":
                1/n for each of the n classes in the training labels, set at
                fit time (null = each class's share of the training samples).

        Proportional-control params:
            proportional (bool): enable speed estimation. Default False.
            rest_class (int): class label treated as rest/idle. Required when
                proportional is on -- the rest centroid is the amplitude origin.
            amp_n_features (int): number of LEADING features that make up the
                current bin. With feature_type [["mav","history"]] and hist=N
                the feature vector is [mav_t, mav_t-1, ... mav_t-N], so this is
                the channel count and speed reacts to the newest bin only
                rather than lagging by the whole history window. Default: all
                features.
            amp_scale_pct (float): percentile of each class's training
                projection mapped to full speed. Default 80.
            amp_rest_pct (float): percentile of the REST projection used as
                that class's dead zone, i.e. move onset. Default 99, so ~1% of
                rest bins would cross it.
            amp_dead_floor (float): lower bound on the dead zone in normalised
                units, applied when rest is very quiet. Default 0.05.
        """
        super().__init__(params)
        params = dict(params)
        self.equal_priors = params.get("priors") == "equal"
        if self.equal_priors:
            params["priors"] = None   # set to 1/n in train_step, once the classes are known
        self.proportional = bool(params.pop("proportional", False))
        self.rest_class = params.pop("rest_class", None)
        self.amp_n_features = params.pop("amp_n_features", None)
        self.amp_scale_pct = float(params.pop("amp_scale_pct", 80.0))
        self.amp_rest_pct = float(params.pop("amp_rest_pct", 99.0))
        self.amp_dead_floor = float(params.pop("amp_dead_floor", 0.05))
        if self.proportional and self.rest_class is None:
            raise ValueError("LDA: 'rest_class' must be set when proportional=True")
        self.rest_mu = None       # amplitude origin (rest centroid)
        self.class_dirs = {}      # class -> unit direction away from rest
        self.class_scales = {}    # class -> projection mapped to full speed
        self.class_deads = {}     # class -> dead zone in normalised units
        self.model = LinearDiscriminantAnalysis(**params)

    def _amp_view(self, X):
        """Leading current-bin slice of the feature matrix."""
        if self.amp_n_features is None:
            return X
        return X[:, :int(self.amp_n_features)]

    def _fit_proportional(self, X, y):
        """Learn the amplitude origin, per-class direction, scale and dead zone.

        Everything here comes from the SAME training data as the classifier, so
        enabling proportional control needs no extra calibration block and no
        graded-effort cues.
        """
        nfeat = X.shape[1]
        if self.amp_n_features is not None:
            n = int(self.amp_n_features)
            if not 0 < n <= nfeat:
                raise ValueError(
                    f"LDA: amp_n_features={n} must be in (0, {nfeat}]. It is the "
                    "CHANNEL COUNT (len of the IndexSelectorBlock indices), not "
                    "the full feature width.")
            if nfeat % n != 0:
                # [mav_t, mav_t-1, ...] is n_channels * (hist + 1) wide, so a
                # non-divisor means amp_n_features and the channel selection
                # have drifted apart and the amplitude slice is misaligned.
                raise ValueError(
                    f"LDA: feature width {nfeat} is not a multiple of "
                    f"amp_n_features={n}. Set amp_n_features to len(indices) in "
                    "the preprocessing IndexSelectorBlock.")
        Xa = np.asarray(self._amp_view(X), dtype=np.float64)
        y = np.asarray(y).ravel()
        rest_mask = y == self.rest_class
        if not rest_mask.any():
            raise ValueError(
                f"LDA: rest_class={self.rest_class} not present in training labels "
                f"{np.unique(y)}; cannot anchor the amplitude origin.")
        self.rest_mu = Xa[rest_mask].mean(axis=0)
        D = Xa - self.rest_mu
        rest_D = D[rest_mask]

        self.class_dirs, self.class_scales, self.class_deads = {}, {}, {}
        for c in np.unique(y):
            if c == self.rest_class:
                continue
            centroid = D[y == c].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm <= 0:
                continue
            u = centroid / norm
            # Keys are normalised to float: labels reach us as int from some
            # paths and float from others (RoundToIntegerBlock), and a dict
            # keyed on the wrong one silently yields speed 0.
            key = float(c)
            self.class_dirs[key] = u
            scale = float(np.percentile(D[y == c] @ u, self.amp_scale_pct))
            self.class_scales[key] = scale if scale > 0 else 1.0
            # Dead zone: how far rest already projects along THIS class's
            # direction. Expressed in normalised units so it composes with the
            # per-class scale.
            dead = float(np.percentile(rest_D @ u, self.amp_rest_pct)) / self.class_scales[key]
            self.class_deads[key] = float(np.clip(max(dead, self.amp_dead_floor), 0.0, 0.95))
            if dead >= 0.5:
                # Rest wanders nearly as far along this grip's direction as the
                # grip itself does, so almost the whole amplitude range is dead
                # and the hand will barely move. Nearly always a training-data
                # problem (mislabelled rest, a drifting baseline, or the
                # participant not relaxing), not a tuning problem.
                print(
                    f"WARNING: LDA proportional -- class {c} dead zone is "
                    f"{dead:.2f} of full-speed amplitude"
                    f"{' (CLIPPED to 0.95)' if dead > 0.95 else ''}. Rest is not "
                    f"separable from this grip in the training data, so speed "
                    f"will stay 0 until contraction is near maximal. Check the "
                    f"rest trials before using this model.")

    def predict_proportional(self, data, classes=None):
        """Predict class AND a speed in [0, 1] for each sample.

        Speed is the contraction amplitude projected onto the predicted class's
        own direction, normalised by that class's training scale and rescaled
        so the dead zone maps to 0. Rest, and anything inside the dead zone,
        returns speed 0 -- that is the amplitude-threshold move-onset, and it
        does not depend on the classifier having a usable rest boundary.

        Args:
            data: input features [N, Nfeat].
            classes: optional class labels to score the amplitude against,
                instead of this model's own argmax. The online node passes its
                DEBOUNCED class here, so the speed always belongs to the grip
                actually being commanded -- otherwise, during a class change,
                the hand would move at a speed measured along a different
                grip's direction.

        Returns:
            (classes, speeds): np.ndarray [N], np.ndarray [N] float64
        """
        if not self.proportional:
            raise RuntimeError("LDA: predict_proportional requires proportional=True")
        if self.rest_mu is None:
            raise RuntimeError("LDA: proportional parameters missing; train or load a model first")
        X = np.asarray(data.detach().cpu().numpy() if hasattr(data, "detach") else data)
        if X.ndim != 2:
            raise ValueError("Input data must be 2-dimensional.")
        if classes is None:
            classes = np.asarray(self.model.predict(X)).ravel()
        else:
            classes = np.asarray(classes).ravel()
            if len(classes) != len(X):
                raise ValueError(
                    f"classes length ({len(classes)}) must match samples ({len(X)})")
        D = np.asarray(self._amp_view(X), dtype=np.float64) - self.rest_mu
        speeds = np.zeros(len(classes), dtype=np.float64)
        for i, c in enumerate(classes):
            key = float(c)
            u = self.class_dirs.get(key)
            if u is None:          # rest, or a class with no usable direction
                continue
            p = float(D[i] @ u) / self.class_scales[key]
            dead = self.class_deads[key]
            speeds[i] = np.clip((p - dead) / (1.0 - dead), 0.0, 1.0)
        return classes, speeds

    def __call__(self, data: Any) -> Any:
        """
        Wrapper function to make the instance callable.
        Args:
            data (torch.tensor or numpy.ndarray): Input data for the model (Nsamp, Nfeat).
        Returns:
            torch.tensor: Predictions from the model.
        """
        return self.forward(data)

    def train_step(self, input_data: Any) -> None:
        """
        Train the LDA model using the provided input data.
        Args:
            input_data (tuple): A tuple containing the input features (X, (Nsamp, Nfeat)) and target labels (y, (Nsamp)).
        """
        X, y = input_data
        if self.equal_priors:
            n = len(np.unique(y))
            self.model.set_params(priors=np.full(n, 1.0 / n))
        self.model.fit(X, y)
        self.Nfeats = X.shape[1]
        if self.proportional:
            self._fit_proportional(X, y)

    def forward(self, data: Any) -> Any:
        """
        Predict outputs given input data.
        Args:
            data (torch.tensor or numpy.ndarray): Input data for prediction (Nsamp, Nfeat).
        Returns:
            torch.tensor: Predicted outputs from the model.
        """
        if data.ndim != 2:
            raise ValueError("Input data must be 2-dimensional.")
        if data.shape[1] != self.Nfeats:
            raise ValueError(f"Number of features in input data ({data.shape[1]}) does not match the model's trained number of features ({self.Nfeats}).")
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data.copy())
        predictions = self.model.predict(data)
        return torch.tensor(predictions, dtype=torch.float64)
    
    def get_likelihood(self, data: Any) -> Any:
        """
        Get the likelihood of the input data given the model.
        Args:
            data (torch.tensor or numpy.ndarray): Input data for likelihood estimation (Nsamp, Nfeat).
        Returns:
            torch.tensor: Likelihood estimates from the model.
        """
        if data.ndim != 2:
            raise ValueError("Input data must be 2-dimensional.")
        if data.shape[1] != self.Nfeats:
            raise ValueError(f"Number of features in input data ({data.shape[1]}) does not match the model's trained number of features ({self.Nfeats}).")
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        likelihood = self.model.predict_proba(data)
        return torch.tensor(likelihood, dtype=torch.float64)

    def save_model(self, fpath: str) -> None:
        """
        Save the trained LDA model to a file.
        Args:
            fpath (str): Path to the file where the model will be saved.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        # Third element carries the proportional-speed parameters. Older
        # checkpoints are 2-tuples and still load (see load_model).
        proportional = {
            "proportional": self.proportional,
            "rest_class": self.rest_class,
            "amp_n_features": self.amp_n_features,
            "amp_scale_pct": self.amp_scale_pct,
            "amp_rest_pct": self.amp_rest_pct,
            "amp_dead_floor": self.amp_dead_floor,
            "rest_mu": self.rest_mu,
            "class_dirs": self.class_dirs,
            "class_scales": self.class_scales,
            "class_deads": self.class_deads,
        }
        with open(fpath, 'wb') as f:
            pickle.dump((self.model, self.Nfeats, proportional), f)

    def load_model(self, fpath: str, running_online: bool = False) -> None:
        """
        Load a trained LDA model from a file.
        Args:
            fpath (str): Path to the file from which the model will be loaded.
            running_online (bool): Whether the model is running online.
        """
        with open(fpath, 'rb') as f:
            blob = pickle.load(f)
        if len(blob) == 2:  # checkpoint saved before proportional control
            self.model, self.Nfeats = blob
            if self.proportional:
                raise ValueError(
                    f"LDA: proportional=True but {fpath} was saved without "
                    "proportional parameters. Retrain the model.")
            return
        self.model, self.Nfeats, prop = blob
        # The checkpoint is the source of truth: the scales were fitted with
        # the amp_n_features that produced them, so a mismatched config must
        # not silently reinterpret them.
        self.proportional = prop["proportional"]
        self.rest_class = prop["rest_class"]
        self.amp_n_features = prop["amp_n_features"]
        self.amp_scale_pct = prop["amp_scale_pct"]
        self.amp_rest_pct = prop["amp_rest_pct"]
        self.amp_dead_floor = prop["amp_dead_floor"]
        self.rest_mu = prop["rest_mu"]
        self.class_dirs = prop["class_dirs"]
        self.class_scales = prop["class_scales"]
        self.class_deads = prop["class_deads"]