from abc import ABC, abstractmethod
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from ..model import Model

class Trainer(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def train_model(self):
        pass