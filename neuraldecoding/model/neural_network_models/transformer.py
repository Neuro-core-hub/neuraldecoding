import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torch import Tensor 
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from neuraldecoding.model.neural_network_models.NeuralNetworkModel import NeuralNetworkModel
import os

# most of the code from: https://github.com/eeyhsong/EEG-Conformer
class PatchEmbedding(nn.Module):
    def __init__(self, num_channels, emb_size=40, drop_prob=0.5):
        super().__init__()

        # Define the fixed smoothing kernel
        window_size = 5
        kernel = torch.ones((1, 1, 1, window_size)) / window_size  # Smoothing kernel

        # Initialize the convolutional layer with the fixed smoothing kernel
        self.fix_conv = nn.Conv2d(1, 40, (1, window_size), (1, 1))
        with torch.no_grad():
            self.fix_conv.weight = nn.Parameter(kernel.repeat(40, 1, 1, 1))
            self.fix_conv.weight.requires_grad = False  # Freeze the weights

        self.shallownet = nn.Sequential(
            # self.fix_conv,
            nn.Conv2d(1, 40, (1, 5), (1, 1)), # convolution over time
            # nn.Conv2d(40, 40, (num_channels, 1), (1, 1)), # convolution over channels
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 4), (1, 4)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(drop_prob)
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x.float())
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_prob):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_prob=0.5,
                 forward_expansion=4,
                 forward_drop_prob=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_prob),
                nn.Dropout(drop_prob)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_prob=forward_drop_prob),
                nn.Dropout(drop_prob)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, num_heads):
        super().__init__(*[TransformerEncoderBlock(emb_size, num_heads) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, num_outputs, drop_prob=0.5):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_outputs)
        )
        self.fc = nn.Sequential(
            nn.Linear(2560, 256),
            nn.ELU(),
            nn.Dropout(drop_prob),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        
        return out

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module, NeuralNetworkModel):

    def __init__(self, model_params):
        super(TransformerModel, self).__init__()
        self.model_params = model_params
        num_features = model_params['num_features']
        num_outputs = model_params['num_outputs']
        enc_nhead = model_params.get('enc_nhead', 2)
        enc_nhid = model_params.get('enc_nhid', 2048)
        enc_nlayers = model_params.get('enc_nlayers', 1)
        dropout = model_params.get('dropout', 0.5)

        self.model_name = 'TransformerModel'
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(num_features, dropout)
        encoder_layers = TransformerEncoderLayer(num_features, enc_nhead, enc_nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, enc_nlayers)
        self.decoder = nn.Linear(num_features, num_outputs)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # nn.init.uniform_(self.transformer_encoder.weight, -initrange, initrange) # TODO fix
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        '''
        Run data forward through the encoder and then an FC layer.
        We never mask the inputs (we're not using this for sequence generation).
        '''
        x = x.permute(2, 0, 1)  # put in format (sequence length (history), batches, features)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output[-1, :, :]

    def save_model(self, filepath):
        checkpoint_dict = {
            "model_state_dict": self.state_dict(),
            "model_params": self.model_params,
            "model_type": "Transformer"
        }
        folder = os.path.dirname(filepath)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(checkpoint_dict, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)

        if checkpoint["model_type"] != "Transformer":
            raise Exception("Tried to load model that isn't a Transformer Instance")

        if self.model_params != checkpoint["model_params"]:
            raise ValueError("Model parameters do not match the checkpoint parameters")

        self.load_state_dict(checkpoint["model_state_dict"])

        self.model_params = checkpoint["model_params"]
    


class TransformerGRUModel(nn.Module):

    def __init__(self, model_params):
        super(TransformerGRUModel, self).__init__()
        self.model_name = 'TransformerModel'
        self.src_mask = None
        
        self.model_params = model_params
        num_features = model_params['num_features']
        num_outputs = model_params['num_outputs']
        enc_nhead = model_params.get('enc_nhead', 2)
        enc_nhid = model_params.get('enc_nhid', 2048)
        enc_nlayers = model_params.get('enc_nlayers', 1)
        dropout = model_params.get('dropout', 0.5)
        rnn_nhid = model_params.get('rnn_nhid', 300)

        self.pos_encoder = PositionalEncoding(num_features, dropout)
        encoder_layers = TransformerEncoderLayer(num_features, enc_nhead, enc_nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, enc_nlayers)
        # self.decoder = nn.Linear(num_features, num_outputs)
        self.rnn_nhid = rnn_nhid
        self.rnn_nlayers = 1
        self.gru = nn.GRU(num_features, rnn_nhid, num_layers=1, batch_first=False)
        self.fc = nn.Linear(rnn_nhid, num_outputs)

    def forward(self, x, h=None):
        '''
        Run data forward through the encoder and then an FC layer.
        We never mask the inputs (we're not using this for sequence generation).
        '''
        if h is None:
            h = self.init_hidden(x.shape[0])

        x = x.permute(2, 0, 1)  # put in format (sequence length (history), batches, features)
        src = self.pos_encoder(x)
        out = self.transformer_encoder(src, self.src_mask)
        out, h = self.gru(out, h)
        out = self.fc(out[-1, :])
        return out, h

    def init_hidden(self, batch_size):
        return torch.zeros(self.rnn_nlayers, batch_size, self.rnn_nhid)

    def save_model(self, filepath):
        checkpoint_dict = {
            "model_state_dict": self.state_dict(),
            "model_params": self.model_params,
            "model_type": "TransformerGRU"
        }
        folder = os.path.dirname(filepath)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(checkpoint_dict, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)

        if checkpoint["model_type"] != "TransformerGRU":
            raise Exception("Tried to load model that isn't a TransformerGRU Instance")

        if self.model_params != checkpoint["model_params"]:
            raise ValueError("Model parameters do not match the checkpoint parameters")

        self.load_state_dict(checkpoint["model_state_dict"])

        self.model_params = checkpoint["model_params"]

class ConformerModel(nn.Module):
    def __init__(self, model_params):
        super(ConformerModel, self).__init__()
        '''
        model_params contains:
        - num_channels
        - num_outputs
        - good_channels_idx
        - emb_size: default 40
        - num_heads: default 10
        - num_layers: default 6
        - drop_prob: default 0.5
        '''
        self.model_name = 'ConformerModel'
        self.good_channels_idx = model_params['good_channels_idx']

        self.model_params = model_params
        num_channels = model_params['num_channels']
        num_outputs = model_params['num_outputs']
        emb_size = model_params.get('emb_size', 40)
        num_heads = model_params.get('num_heads', 10)
        num_layers = model_params.get('num_layers', 6)
        drop_prob = model_params.get('drop_prob', 0.5)

        self.patch_embedding = PatchEmbedding(num_channels, emb_size, drop_prob)
        self.transformer = TransformerEncoder(num_layers, emb_size, num_heads)
        self.class_head = ClassificationHead(emb_size, num_outputs, drop_prob)
    
    def forward(self, x):
        # take only the good channels
        x = x[:,self.good_channels_idx,:].unsqueeze(1)

        # take only the first 32 channels
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.class_head(x)

        return x

    def save_model(self, filepath):
        checkpoint_dict = {
            "model_state_dict": self.state_dict(),
            "model_params": self.model_params,
            "model_type": "Conformer"
        }
        folder = os.path.dirname(filepath)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(checkpoint_dict, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)

        if checkpoint["model_type"] != "Conformer":
            raise Exception("Tried to load model that isn't a Conformer Instance")

        if self.model_params != checkpoint["model_params"]:
            raise ValueError("Model parameters do not match the checkpoint parameters")

        self.load_state_dict(checkpoint["model_state_dict"])

        self.model_params = checkpoint["model_params"]