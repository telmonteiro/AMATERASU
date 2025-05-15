---
title: 'AMATERASU: a tool to track stellar activity'
tags:
  - Python
  - astronomy
  - stellar activity
  - stellar spectra
authors:
  - name: Telmo Monteiro
    orcid: 0000-0001-8991-4615
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: João Gomes da Silva
    orcid: 0000-0001-8056-9202
    affiliation: 2
  - name: Elisa Delgado-Mena
    orcid: 0000-0003-4434-2195
    affiliation: 3
  - name: Nuno C. Santos
    orcid: 0000-0002-0580-0475
    affiliation: "1,2"
affiliations:
 - name: Departamento de Física e Astronomia, Faculdade de Ciências, Universidade do Porto, Rua do Campo Alegre, 4169-007 Porto, Portugal
   index: 1
 - name: Instituto de Astrofísica e Ciências do Espaço, Universidade do Porto, CAUP, Rua das Estrelas, 4150-762 Porto, Portugal
   index: 2
 - name: Centro de Astrobiología (CAB), CSIC-INTA, Camino Bajo del Castillo s/n, 28692, Villanueva de la Cañada (Madrid), Spain
   index: 3
date: 14 May 2025
bibliography: paper.bib

---

# Summary

AMATERASU (AutoMATic Equivalent-width Retrieval for Activity Signal Unveiling) is a Python tool to check for periods in spectral activity indices similar to an input period. This way, by running AMATERASU for a specific spectral line, the user can see if the input period may be correlated with activity. The aim of this program is to provide an easy and quick way to check for clues that the period observed in, for example, RV is of stellar origin. AMATERASU follows a methodology similar to @GomesdaSilva:2025 (accepted) and is inspired by ACTIN [@GomesdaSilva:2018], [@GomesdaSilva:2021].  

# Statement of need

Stellar variability can impact planetary signals detected via the RV method. This is often addressed by tracking spectral lines sensitive to magnetic or/and temperature changes in the stellar atmosphere. With the growing use of NIR instruments like NIRPS, understanding NIR activity indicators is crucial, as their sensitivity may vary with stellar properties.
While several tools exist to compute spectral indices (e.g., ACTIN), few provide a straightforward way to systematically test whether a given periodicity, such as a candidate planet signal, may be reproduced in activity indicators. Additionally, tools like ACTIN only compute one version of a given activity indicator (e.g. with a fixed central bandpass), which may not be optimal to track stellar activity [@GomesdaSilva:2022].

In this context, AMATERASU fills a gap by enabling users to efficiently compute equivalent-widths (EWs) of spectral lines, search for periodicities in their time-series and identify whether those periods match an input period. 
By combining flexible input formats, automatic window definition, and batch analysis of multiple lines and periods, AMATERASU provides a targeted, user-friendly and quick solution to validate the stellar or planetary origin of RV signals.


# Description

AMATERASU computes the equivalent-width (EW) of a spectral line in a normalization independent way, by using the 90th percentile of the flux in a given window as the continuum level. It then computes the EW for an array of bandpasses, going from 0.1 \r{A} up to a user defined width.
This way, the input includes the spectral line center, maximum bandpass width window and a window that includes both the line and some continuum flux. The flux is interpolated inside the window using the \texttt{specutils} package, with a step similar to the original spectrum's step, to align the spectrum portion with the bandpass edges. 

The maximum bandpass and interpolation window can be given manually or automatically. In the automatic way, the spectra are coadded into a master spectrum, using the first spectrum as the common reference grid, which is then smoothed with a moving average. Then, the \texttt{find\_peaks} function from the \texttt{scipy} package is used to find the spectral lines and then retrieves the FWHM of the closest line to be studied (with a threshold of 0.1 \r{A}). The bandpass window is a multiple (rounded) of the FWHM retrieved (by default 5 times) and the interpolation window is triple that (15 times the FWHM). Having retrieved a time series of EWs measurements for a given bandpass, AMATERASU cleans the data by applying a 3-$\sigma$ sequential clipping and binning the data by night. 

Finally, AMATERASU runs Generalized Lomb-Scargle (GLS) periodograms [@Zechmeister:2009] and extracts the significant peaks. These are the ones not close to peaks in the window function periodogram and with a false alarm probability (FAP) under the specified FAP threshold. If any of the significant peaks is close within the specified threshold to the input period, the program warns the user.

The user can choose one of the predefined indices in the ``ind_table.csv`` table or test a new indice. 
The tool accepts a list of input periods and a list of lines to analyse, as well as 1D or 2D spectra. To be more flexible with the instruments tested, the spectra have to be given in an array format: for 2D spectra, the array should have dimensions of ($N_\text{obs}$,$N_\text{spectrum axis}$,$N_\text{orders}$,$N_\text{pixels}$), where $N_\text{spectrum axis}$ refers for the wavelength, flux and flux error of each spectrum. An array with the dates of observations in BJD is also needed.

Besides the GLS periodograms, AMATERASU can also compute the Spearman ranked correlation coefficient between all the central bandpasses and an input array, for example a known activity indice, like the CCF FWHM of the observations.

AMATERASU has two modes: standard and full output. The standard output is the fastest, as it only computes the EW time-series and correspondent GLS periodograms, prints wether any of the periods is close to the input one and saves (if wanted) a \texttt{.csv} file with the matching periods and respective central bandpasses and a \texttt{.csv} file with the correlations with the input array per bandpass (if wanted). The full output allows to save all analysis data in a directory, including:

- table with the EW values and errors for each central bandpass, as well as the BJD time
- plot of time-series of each EW computed and corresponding GLS periodogram
- separate periodogram and WF periodogram for each EW (for debugging purposes)
- table with the periods matching input
- table with all significant periods found
- table with all the information on the periodograms
- table with the Spearman correlations per bandpass with input array

So far, we only tested AMATERASU with NIRPS data, so the predefined indices consist in NIR lines. Additionally, AMATERASU's automatic window definition only works for simmetrical absorption lines with high depth, so it will fail for lines with assimetrical profiles, with core emission or with blends.

The code is available and will be updated on [GitHub][] and can be easily installed using pip.

[GitHub]: https://github.com/telmonteiro/AMATERASU

# References
