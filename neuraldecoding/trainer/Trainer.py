from abc import ABC, abstractmethod
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from neuraldecoding.model.Model import Model

class Trainer(ABC):
    @abstractmethod
    def train_model(self, 
                    train_data: DataLoader,
                    valid_data: DataLoader,
                    model: Model,
                    config: DictConfig) -> Model:
        """
        Trains the model using the provided training, validation, and test data.
        
        Parameters:
            train_data: Training dataset in the form of numpy DataLoader
            valid_data: Validation dataset in the form of numpy DataLoader
            model: An instance of Model to be trained.
            config: Hydra configuration file containing all required parameters
            
        Returns:
            A trained Model instance.
        """
        pass