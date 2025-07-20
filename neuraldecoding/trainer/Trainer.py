from abc import ABC, abstractmethod
from omegaconf import DictConfig

class Trainer(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # Each metric has a list of train and validation metrics
        self.logger = {metric: [[], []] for metric in self.cfg.metrics}
    
    @abstractmethod
    def train_model(self):
        pass
    
    def print_metrics(self):
        # FIXME: for now, just printing the last metric value
        for metric in self.logger:
            train_metric = self.logger[metric][0][-1]
            val_metric = self.logger[metric][1][-1]
            print(f"    {metric:>12}{': train = ':>12}{train_metric}")
            print(f"    {'':>12}{'  val = ':>12}{val_metric}")
    