from abc import ABC, abstractmethod
from typing import Any, List

class Model(ABC):
    def __init__(self, params: List[Any]) -> None:
        self.params = params
    
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        pass
    
    @abstractmethod
    def train_step(self, input_data: Any) -> None:
        pass

    @abstractmethod
    def forward(self, input: Any) -> Any:
        pass

    @abstractmethod
    def save_model(self, filepath: str) -> None:
        pass
    
    @abstractmethod
    def load_model(self, filepath: str, running_online: bool = False) -> None:
        pass


class DummyModel(Model):
    def train_step(self, input_data: Any) -> None:
        print("rocky theme plays")
        return
    
    def forward(self, input: Any) -> Any:
        print("plop")
    
    def save_model(self, filepath: str) -> None:
        print("this model is saving itself for marriage")
    
    def __call__(self, data: Any) -> Any:
        self.forward(data)
        return