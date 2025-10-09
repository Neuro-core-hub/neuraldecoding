# Neural Decoding

# Setup
- First, create a conda environment (neuraldecoding) with  `conda create -n neuraldecoding python=3.11`
- Then, activate the environment with `conda activate neuraldecoding`
- Install the necessary packages with `pip install -r requirements.txt`
- Install the package with `pip install -e .`
- You should now be able to use the neuraldecoding package (when the neuraldecoding environment is active)


## Loading Data
The core way data gets used in this repo is through the `dataset` module. Datasets are loaded using a standard config YAML file. To handle loading data from different sources, we have different loading mechanisms for each dataset type which reformat the data into a more standardized NWB format. Please refer to the wiki for details on specific loaders.

# Running Tests
To run all tests, in root folder run:

Run `python -m unittest discover -s tests -p "test*.py"`
