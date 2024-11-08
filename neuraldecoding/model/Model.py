from abc import ABC, abstractmethod
from typing import Any, List

class Model:
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
    def load_model(self, filepath: str) -> None:
        pass



