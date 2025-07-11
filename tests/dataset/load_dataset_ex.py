import neuraldecoding.dataset as neuraldataset
import yaml
from omegaconf import OmegaConf

# load the zstruct and save it out as an nwb

# check that an nwb filed was by default saved

sample_config_file = OmegaConf.load("C:\\Repos\\neuraldecoding\\neuraldecoding\\example_configs\\datasets\\xpc_jh_emg.yaml")

data = neuraldataset.Dataset(sample_config_file)