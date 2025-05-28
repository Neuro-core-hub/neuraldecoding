from typing import Any
from neuraldecoding.model import Model

class NeuralNetworkModel(Model):

    def __call__(self, data: Any) -> Any:
        pass

    def train_step(self, input_data: Any) -> None:
        pass

    def forward(self, input: Any) -> Any:
        pass

    def save_model(self, filepath: str) -> None:
        pass
    
    def load_model(self, filepath: str) -> None:
        pass
