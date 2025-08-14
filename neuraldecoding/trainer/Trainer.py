from abc import ABC, abstractmethod
from omegaconf import DictConfig
import os
class Trainer(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # Each metric has a list of train and validation metrics
        self.metrics = cfg.evaluation.metrics
        self.metric_params = cfg.evaluation.get("params", {})
        self.logger = {metric: [[], []] for metric in self.metrics}
        self.logger_save_path = cfg.evaluation.get("save_path", None)
        self.print_results = cfg.training.get("print_results", True)
        self.print_every = cfg.training.get("print_every", 1)
        if self.logger_save_path:
            os.makedirs(os.path.dirname(self.logger_save_path), exist_ok=True)
            with open(self.logger_save_path, 'a') as f:
                headers = ['epoch'] + [f'{metric}_train' for metric in self.metrics] + [f'{metric}_val' for metric in self.metrics]
                f.write(','.join(headers) + '\n')

    @abstractmethod
    def train_model(self):
        pass
    
    def save_print_log(self, epoch = 0, train_loss = None, val_loss = None):
        # Save log
        if self.logger_save_path:
            with open(self.logger_save_path, 'a') as f:
                entries = [epoch] + [f'"{self.logger[metric][0][-1]}"' for metric in self.metrics] + [f'"{self.logger[metric][1][-1]}"' for metric in self.metrics]
                f.write(','.join(map(str, entries)) + '\n')

        # Print log
        text = ""
        if self.print_results and (epoch % self.print_every == 0 or epoch == self.num_epochs - 1):
            if train_loss is None and val_loss is None:
                text += f"Epoch {epoch}/{self.num_epochs - 1}\n"
            else:
                text += f"Epoch {epoch}/{self.num_epochs - 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
            for metric in self.logger:
                train_metric = self.logger[metric][0][-1]
                val_metric = self.logger[metric][1][-1]
                text += f"    {metric:>12}{': train = ':>12}{train_metric}\n"
                text += f"    {'':>12}{'  val = ':>12}{val_metric}\n"
        if text:
            print(text)
        return text


    