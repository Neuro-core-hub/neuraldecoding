import sys
import os

# Add the parent directory to the Python path to make the package importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the Dataset class from the neuraldecoding package
from neuraldecoding.dataset import Dataset
from hydra import initialize, compose

with initialize(version_base=None, config_path="../neuraldecoding/config"):
    cfg = compose("config")
dataset = Dataset(cfg.dataset, verbose=True)
dataset.load_data()
pass
