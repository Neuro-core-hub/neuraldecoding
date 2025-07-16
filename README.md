# neural_decoding

to install run:
```
pip install -e .
```
while in this folder

## Loading Data
The core way data gets used in this repo is through the `dataset` module. Datasets are loaded using a standard config YAML file. To handle loading data from different sources, we have different loading mechanisms for each dataset type which reformat the data into a more standardized NWB format. Please refer to the wiki for details on specific loaders.

# neural_decoding

Install environment: `conda env create -f environment.yaml`

# Running Tests

Run `python -m unittest discover -s tests -p "test*.py"`

# Running Examples

1. Run `python examples/decoder/kalman_filter.py`
2. Run `python examples/decoder/lstm.py`
3. Run `python examples/decoder/ridge_regression.py`