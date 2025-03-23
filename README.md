# AMATERASU - AutoMATic Equivalent-width Retrieval for Activity Signal Unveiling

## What is AMATERASU?

AMATERASU (AutoMATic Equivalent-width Retrieval for Activity Signal Unveiling) is a simple Python tool to check for periods in spectral activity indices similar to an input period. This way, by running AMATERASU for a spectral line, the user can see if the input period may be correlated with activity. It's important to note that AMATERASU follows a methodology similar to the one described in Gomes da Silva et al. 2025 (in prep) and is heavily inspired by ACTIN2.

## How does AMATERASU work?

AMATERASU computes the equivalent width (EW) of a spectral line in a normalization independent way, by using the 90th percentile of the flux in a given window as the continuum level. It computes the EW for an array of bandpasses, going from 0.1 $\AA$ up to a user defined width.

This way, the input includes the spectral line center, bandpass width and a window that includes both the line and some continuum. By default, the flux is interpolated inside the window, with a step similar to the original spectrum's step. The maximum bandpass and the interpolation window can be given manually or automatically. Automatically, the spectra are coadded and then the code uses the find_peaks function of scipy to find spectral lines in a given order and then retrieves the FWHM of the closest line to be studied (threshold of 0.1 A). The bandpass window is a multiple (rounded) of the FWHM retrieved (by default 4 times) and the interpolation window is triple that (12 times the FWHM).

Having retrieved a time series of EWs measurements for a given bandpass, AMATERASU cleans the data by 3-sigma sequential clipping and binning the data by night. 

Finally, AMATERASU runs GLS periodograms (with a bunch of technicalities) and if the significant peak with most power is similar to the tested/input period, the program warns the user.

The user can choose one of the predefined indices in the ``ind_table.csv`` table or define a new indice.

Accepts a list of input periods and a list of lines to analyse.
Accepts both 1D and 2D spectra.

Besides the GLSPs, AMATERASU can compute the correlation of all the central bandpasses with an input array, that can be a known activity indice. It then prints the bandpass that maximizes correlation (positive or negative).

Output options:
    - Standard: warns if some input period was detect, in which line and with which FAP. Prints and saves a dataframe.
    - Full: saves all analysis data in a directory.

## Caveats and future upgrades

Caveats:
- AMATERASU was tested using NIRPS spectra only, so the predefined indices are NIR lines.
- AMATERASU was only tested with spectral lines that were more or less simmetrical and with a decent depth, so spectral lines like He I 10830 $\AA$ or Paschen $\beta$ were not considered.
- Interp_win should be big enough to not include flux from the line. If not calibrated the signal detected through GLSPs can be degraded and starts raising the year/2 signal.

Future upgrades:
- Include option to convert the wavelength to RV space.
  
## Running AMATERASU

- To run AMATERASU, make sure to have the packages needed downloaded. The Python version used to build AMATERASU was 3.10.12, ran inside WSL Ubuntu with VsCode.
- Then, in the directory of AMATERASU, the user runs

sudo python setup.py install
