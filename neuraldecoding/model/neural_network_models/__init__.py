from neuraldecoding.model.neural_network_models.LSTM import LSTM, LSTMTrialInput
from neuraldecoding.model.neural_network_models.RNN import RecurrentModel
from neuraldecoding.model.neural_network_models.tcFNN import TCN
import warnings
try:
    import einops
    from neuraldecoding.model.neural_network_models.transformer import TransformerModel, TransformerGRUModel, ConformerModel
except ImportError:
    warnings.warn("Transformer models not available: einops package is required but not installed", ImportWarning)