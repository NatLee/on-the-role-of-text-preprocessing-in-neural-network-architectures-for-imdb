"""Reproduce the IMDb experiments of "On the Role of Text Preprocessing in Neural Network Architectures"
(Camacho-Collados and Pilehvar, BlackboxNLP 2018) with a CNN or CNN+LSTM text classifier in Keras."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("preproc-imdb")
except PackageNotFoundError:  # running from a source checkout without installing
    __version__ = "0+unknown"
