# AMATERASU

## What is AMATERASU?

AMATERASU (AutoMATic Equivalent-width Retrieval for Activity Signal Unveiling) is a simple Python tool to check for periods in spectral activity indices similar to an input period. This way, by running AMATERASU for a spectral line, the user can see if the input period may be related to stellar activity.

## How does AMATERASU work?

AMATERASU was designed to have a modular architecture, allowing the user to easily access the different classes of the tool (e.g., cleaning data, running GLS periodograms). It also comes with several tunable parameters, giving the user full control, while keeping an easy "blind" implementation.

AMATERASU computes the pseudo equivalent-width (pEW) of a spectral line in a normalization independent way, using bandpasses from a grid that goes from 0.1 Å up to a user-defined maximum. 
When computing the pEW, the flux is interpolated with a step similar to original spectrum's step, aligning the spectrum portion with the bandpass edges, and the continuum level is approximated as a user-defined percentile of the flux around the spectral line.
Having retrieved a time series of pEWs measurements for a given bandpass, by default AMATERASU cleans the data by applying a 3-$\sigma$ sequential clipping and binning the data by night. 

Then, AMATERASU runs Generalized Lomb-Scargle (GLS) periodograms and extracts the significant peaks, the ones not close to peaks in the window function periodogram and with a false alarm probability (FAP) under the specified threshold. If any of the significant peaks is close within the specified threshold to the input period, the tool warns the user.
AMATERASU can also compute the Spearman correlation ranked coefficient and associated p-value between the pEW arrays and an input array, that can be, for example, a known activity indicator.

The basic input consists of a spectral data cube (time-series of 1D or 2D spectra), various analysis options and information on the spectral line (line center, maximum central bandpass window and continuum window).
The maximum central bandpass and continuum windows can be given either manually or computed automatically. 
In the automatic way, the spectra are coadded into a master spectrum, using the first spectrum as the common reference grid, which is then smoothed with a moving average. Then, the target spectral line is found and its full-width-at-half-maximum (FWHM) is retrieved. 
The bandpass window is set as an integer multiple of the FWHM retrieved (by default 5 times) and the continuum window is 6 times that (30 times the FWHM). 

The user can choose one of the predefined lines in the ``ind_table.csv`` table or test a new index. 
The tool accepts a list of input periods and a list of lines to analyse, as well as 1D or 2D spectra. 
To be more flexible with the instruments tested, the spectra have to be given in an array format: for 2D spectra, the array should have dimensions of ($N_\text{obs}$, $N_\text{spectrum axis}$, $N_\text{orders}$, $N_\text{pixels}$), where $N_\text{spectrum axis}$ refers for the wavelength, flux and flux error of each spectrum. An array with the dates of observations in BJD is also needed.

AMATERASU has two output modes: "standard" and "full". The standard output is the fastest, as it only computes the pEW time-series and correspondent GLS periodograms, prints wether any of the periods is close to the input one and saves (if wanted) a .csv file with the matching periods and respective central bandpasses and a .csv file with the Spearman correlations with the input array per bandpass (if wanted). 
The full output allows to save all analysis data in a directory, including:

- table with the pEW values and errors for each central bandpass, as well as the BJD time
- table with the periods matching input
- table with all significant periods found
- table with all the periodogram information
- table with the Spearman correlations per central bandpass with input array
- plot of time-series of each pEW computed and corresponding GLS periodogram
- separate periodogram and WF periodogram plots for each pEW (for debugging purposes)

AMATERASU is completely instrument agnostic, but we mostly used NIRPS data when building the tool, so the predefined indices available are NIR spectral lines.
Nevertheless, the tool has been proven to work with optical HARPS spectra (Soldevilla et al., in prep). 
Additionally, AMATERASU's automatic window definition only works for symmetrical absorption lines, often failing for lines with core emission, resolved Zeeman splitting or with line blends. In those cases, the user can still use AMATERASU with user-defined windows.

## Installation and Running

To run AMATERASU, make sure to have the packages needed installed. The Python version used to build AMATERASU was 3.10.19, ran inside WSL Ubuntu with VSCode.

To install run:

git clone https://github.com/telmonteiro/AMATERASU
cd AMATERASU
pip install -e .

A tutorial notebook is also provided, as well as a sample of Proxima Centauri observations taken with the NIRPS spectrograph.
