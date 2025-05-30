# neural_decoding

1. Add conda forge to conda channel: `conda config --append channels conda-forge`

2. Install environment: `conda create -n ND —file requirement.txt`

3. Install pytorch from: `https://pytorch.org/get-started/locally/`


# Running Tests
Run `python -m unittest discover -s tests -p "test*.py"`

# Running Examples

1. Run `python examples/decoder/kalman_filter.py`
2. Run `python examples/decoder/lstm.py`
3. Run `python examples/decoder/ridge_regression.py`
