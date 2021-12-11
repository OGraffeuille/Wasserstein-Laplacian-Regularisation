# Wasserstein MDN using PyTorch
This is the PyTorch source code for "Semi-Supervised Conditional Density Estimation with Wasserstein Mixture Density Networks".
This is accompanying code for the AAAI 2021 paper by Olivier Graffeuille, Yun Sing Koh, Jörg Wicker and Moritz Lehmann.

### What is implemented
WMDN code is supplied (WMDN.py) along with a running Jupyter notebook example (run_WMDN.ipynb).

### Dependencies
Code was tested with:
- Python 3.8.5
- torch 1.8.1 with Cuda 11.1
- numpy 1.19.3
- scikit-learn 0.23.2

### Datasets
All datasets are available in the data folder, with the exception of the Real chlorophyll-a dataset as it is not currently publically available, and the Song Year and Electric datasets due to size restrictions.
