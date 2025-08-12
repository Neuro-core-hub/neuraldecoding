import torch.nn as nn
import torch.nn.functional as F
import torch
from neuraldecoding.model.neural_network_models.NeuralNetworkModel import NeuralNetworkModel
import os

def flatten(x, start_dim=1, end_dim=-1):
    return x.flatten(start_dim=start_dim, end_dim=end_dim)

class TCN(nn.Module, NeuralNetworkModel):
    def __init__(self, params):
        '''
        Initializes a TCFNN
        Args:
            model_params: dict containing the following model params:
                input_size:         number of input features
                hidden_size:        size of hidden state in model
                ConvSize:         number of convolutional filters
                ConvSizeOut:      number of output channels for convolutional layer
                num_states:        number of output features
                use_batch_norm:    whether to use batch normalization
                use_dropout:        whether to use dropout
                drop_prob:         probability of dropout
        '''
        super().__init__()
        self.model_params = params
        self.input_size = params["input_size"]
        self.hidden_size = params["hidden_size"]
        self.ConvSize = params["ConvSize"]
        self.ConvSizeOut = params["ConvSizeOut"]
        self.num_states = params["num_states"]
        self.use_batchnorm = params.get("use_batch_norm", True)
        self.use_dropout = params.get("use_dropout", True)
        self.drop_prob = params.get("drop_prob", 0.5)
        # assign layer objects to class attributes
        self.cn1 = nn.Conv1d(self.ConvSize, self.ConvSizeOut, 1, bias=True)
        self.fc1 = nn.Linear(self.input_size * self.ConvSizeOut, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.num_states)
        if self.use_batchnorm:
            self.bn0 = nn.BatchNorm1d(self.input_size)
            self.bn1 = nn.BatchNorm1d(self.input_size * self.ConvSizeOut)
            self.bn2 = nn.BatchNorm1d(self.hidden_size)
            self.bn3 = nn.BatchNorm1d(self.hidden_size)
            self.bn4 = nn.BatchNorm1d(self.hidden_size)
            self.bn5 = nn.BatchNorm1d(self.num_states)
        if self.use_dropout:
            self.do1 = nn.Dropout(p=self.drop_prob)
            self.do2 = nn.Dropout(p=self.drop_prob)
            self.do3 = nn.Dropout(p=self.drop_prob)

        # nn.init package contains convenient initialization methods
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        nn.init.kaiming_normal_(self.cn1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        nn.init.zeros_(self.cn1.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def __call__(self, data):
        return self.forward(data)
    
    def forward(self, x, BadChannels=()):
        if x.dim() != 3:
            raise ValueError(f"Input tensor must be 3D (batch_size, num_inputs, sequence_length), got shape {x.shape}")
        #TODO: check what input shape correspondence is
        if x.shape[1] != self.input_size:
            raise ValueError(f"Input feature dimension mismatch: expected {self.input_size}, got {x.shape[1]}")
        #TODO: check what input shape correspondence is
        x[:, BadChannels, :] = 0
        
        if self.use_batchnorm:
            x = self.bn0(x)
        
        x = self.cn1(x.permute(0, 2, 1))
        x = flatten(x)
        
        if self.use_batchnorm:
            x = F.relu(self.bn1(x))
        else:
            x = F.relu(x)
        
        x = self.fc1(x)
        if self.use_dropout:
            x = self.do1(x)
        if self.use_batchnorm:
            x = F.relu(self.bn2(x))
        else:
            x = F.relu(x)

        x = self.fc2(x)
        if self.use_dropout:
            x = self.do2(x)
        if self.use_batchnorm:
            x = F.relu(self.bn3(x))
        else:
            x = F.relu(x)
        
        x = self.fc3(x)
        if self.use_dropout:
            x = self.do3(x)
        if self.use_batchnorm:
            x = F.relu(self.bn4(x))
        else:
            x = F.relu(x)
        
        scores = self.fc4(x)
        if self.use_batchnorm:
            scores = (self.bn5(scores) - self.bn5.bias) / self.bn5.weight
        
        return scores
    
    def save_model(self, filepath):
        checkpoint_dict = {
            "model_state_dict": self.state_dict(),
            "model_params": self.model_params,
            "model_type": "TCFNN"
        }
        folder = os.path.dirname(filepath)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(checkpoint_dict, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)

        if checkpoint["model_type"] != "TCFNN":
            raise Exception("Tried to load model that isn't a TCFNN Instance")
        
        if self.model_params != checkpoint["model_params"]:
            raise ValueError("Model parameters do not match the checkpoint parameters")

        self.load_state_dict(checkpoint["model_state_dict"])

        self.model_params = checkpoint["model_params"]