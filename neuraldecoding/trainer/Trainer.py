from abc import ABC, abstractmethod
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from neuraldecoding.model.Model import Model

class Trainer(ABC):
    @abstractmethod
    def train_model(self):
        pass