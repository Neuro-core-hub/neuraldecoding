import sys
import os
from hydra import initialize, compose

neural_decoding_dir = "D:/neuraldecoding"
cfg_path = os.path.join("config.yaml")

sys.path.append(neural_decoding_dir)

from neuraldecoding.trainer.NeuralNetworkTrainer import NNTrainer
from neuraldecoding.preprocessing import Preprocessing
from neuraldecoding.utils.utils_general import export_preprocess_params

with initialize(version_base=None, config_path=os.path.dirname(cfg_path)):
    config = compose(config_name=os.path.basename(cfg_path))
config.trainer.model = export_preprocess_params(config.trainer.model, config.preprocessing)

preprocessor = Preprocessing(config.preprocessing)
trainer = NNTrainer(preprocessor, config.trainer)

NNTrainer.train_model()




