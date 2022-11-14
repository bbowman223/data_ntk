
This directory contains the files to compute the NTK spectrum.  Below is an explanation of each directory

****Feedforward/****
Implements a feedforward network using the NTK parameterization

****data/****
Directory to store the data

  ****utils/****
This directory contains utility files

  ****notebooks/****
The file `NTK spectra for Caltech101 and Gaussian Data.ipynb` plots the NTK spectrum for [Caltech101](https://data.caltech.edu/records/mzrjq-6wc02) and isotropic Gaussian data for both feedforward networks and CNNs.  The definitions of the networks are inline in the file.

The file `plot_asymptotic_spectrum.ipynb` estimates the asymptotic decay of the spectrum for input data that is uniform on the sphere $\mathbb{S}^2$.  This file uses the feedforward model defined in the `./Feedforward` directory
