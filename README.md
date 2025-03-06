## What is AMATERASU?

AMATERASU (AutoMATic Equivalent-width Retrieval for Activity Signal Unveiling) is a simple Python tool to check for periods in spectral activity indices similar to an input period. This way, by running AMATERASU for a spectral line, the user can see if the input period may be correlated with activity. It's important to note that AMATERASU follows a methodology similar to the one described in Gomes da Silva et al. 2025 (in prep) and is heavily inspired by ACTIN2.

## How does AMATERASU work?

AMATERASU computes the equivalent width (EW) of a spectral line in a normalization independent way, by using the 90th percentile of the flux in a given window as the continuum level. It computes the EW for an array of bandpasses, going from 0.1 Å up to a user defined width.

This way, the input includes the spectral line center, bandpass width and a window that includes both the line and some continuum. By default, the flux is interpolated inside the window, with a step similar to the original spectrum's step.

Having retrieved a time series of EWs measurements for a given bandpass, AMATERASU cleans the data by 3-sigma sequential clipping and binning the data by night. 

Finally, AMATERASU runs GLS periodograms (with a bunch of technicalities) and if the significant peak with most power is similar to the tested/input period, the program warns the user.

The user can choose one of the predefined indices in the ``ind_table.csv`` table or define a new indice.

## Caveats and future upgrades

Caveats:
- AMATERASU was tested using NIRPS spectra only, so the predefined indices are NIR lines.
- AMATERASU was only tested with spectral lines that were more or less simmetrical and with a decent depth, so spectral lines like He I 10830 Å or Paschen $\beta$ were not considered.
- The interpolation included seems to behave weirdly for narrow bandpasses. For example, with a narrow bandpass and ACTIN I could detect Proxima's rotation period, but this is hardly the case in AMATERASU (for K I (c), the rotation period was detected with bandpasses of 0.3 Å and above). The interpolation method is basically the same as the "new" method in ACTIN2, and not the "true" method.

Future upgrades:
- Code more organized and some more functionalities (similar to ACTIN, e.g. line plots, more user friendly, etc).
- Include option to convert the wavelength to RV space.
- Improve interpolation method.
