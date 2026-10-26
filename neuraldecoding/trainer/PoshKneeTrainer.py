import numpy as np
from omegaconf import DictConfig

from neuraldecoding.preprocessing import Preprocessing
from neuraldecoding.trainer.LinearTrainer import LinearTrainer


class PoshKneeTrainer(LinearTrainer):
    """Fits a PoshKnee model (Posh Eq. 1, knee only) from labelled bins, e.g. the LL01 cued runs.

    The training pipeline must end with class numbers per bin (OneHotToClassNumberBlock ->
    FeatureExtractionBlock -> RoundToIntegerBlock, as the LDA configs do); the class names come
    from the patterns' units on the preprocessing bus. The fit uses every bin (train and
    validation split together): its parameters are averages and percentiles, not a regression
    that could overfit. The metrics are direction checks of the fitted model, in-sample:
      flexion_direction    flexion bins read as flexion (u > 0)
      extension_direction  extension bins read as extension (u < 0)
      rest_in_deadband     rest bins inside the deadband (u == 0)
    """

    def __init__(self, preprocessor: Preprocessing, config: DictConfig, dataset=None):
        super().__init__(preprocessor, config, dataset)

    def load_data(self, dataset):
        super().load_data(dataset)
        units = self.preprocessor.interpipeline_data.get("behavior_units")
        if units is None:
            raise ValueError("PoshKneeTrainer: no class names on the preprocessing bus (needs a patterns stream)")
        self.model.class_names = list(eval(units)) if isinstance(units, str) else list(units)

    def train_model(self, plot_results=False):
        X = np.concatenate([np.asarray(self.train_X), np.asarray(self.valid_X)])
        y = np.concatenate([np.asarray(self.train_Y).ravel(), np.asarray(self.valid_Y).ravel()])
        self.model.train_step((X, y))
        self.validate_model()
        sc = self.model.scalars
        k = sc["knee"]
        print(f"Posh knee fit: bins {sc['bins']}; weights w_f {np.round(sc['weights']['w_f'], 4).tolist()}, "
              f"w_e {np.round(sc['weights']['w_e'], 4).tolist()}; wedge {np.degrees(k['th_ag'] - k['th_an']):.1f} deg, "
              f"deadband g0 {k['g0']:.3f}")
        return self.model, self.logger

    def validate_model(self, plot_results=False):
        for split, X, y in (("train", self.train_X, self.train_Y), ("valid", self.valid_X, self.valid_Y)):
            labels = self.model.labels(y)
            u = self.model.knee_u(np.asarray(X))
            opts = self.model.scalars["classes"]
            checks = {
                "flexion_direction": (np.isin(labels, opts["flexion"]), lambda v: v > 0),
                "extension_direction": (np.isin(labels, opts["extension"]), lambda v: v < 0),
                "rest_in_deadband": (np.isin(labels, opts["rest"]), lambda v: v == 0),
            }
            for metric in self.metrics:
                if metric in checks:
                    mask, ok = checks[metric]
                    value = float(np.mean(ok(u[mask]))) if mask.any() else float("nan")
                    self.logger[metric][split].append(round(value, 3))
