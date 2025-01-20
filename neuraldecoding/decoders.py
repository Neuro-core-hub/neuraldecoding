from abc import ABC, abstractmethod
from utils.data_storing import DataManager

class Decoder(ABC):
    def __init__(self, model, stabilization):
        """_summary_

        Args:
            model: trained model
            stabilization: trained stabilization method
        """
        self.model = model
        self.stabilization = stabilization

    def save(self):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, neural_data):
        raise NotImplementedError
    
class StreamingDecoder(Decoder):
    def predict(self, neural_data):
        raise NotImplementedError
    
    def update(self, neural_data):
        raise NotImplementedError
    
class OnlineDecoder(Decoder):
    def predict(self, neural_data):
        raise NotImplementedError
  
    def decode_off(self):
        raise NotImplementedError
    
class OfflineDecoder(Decoder):
    def predict(self, neural_data):

        stabilized_data = self.stabilization.stabilize(neural_data)
        
        prediction = self.model.forward(stabilized_data)
        
        return prediction
    

    