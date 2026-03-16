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
    affiliation: "1, 2" 
  - name: João Gomes da Silva
    orcid: 0000-0001-8056-9202
    affiliation: "2"
  - name: Elisa Delgado-Mena
    orcid: 0000-0003-4434-2195
    affiliation: "3"
  - name: Nuno C. Santos
    orcid: 0000-0002-0580-0475
    affiliation: "1,2"
  - name: Ana Soldevilla
    orcid:
    affiliation: "3"
affiliations:
 - name: Departamento de Física e Astronomia, Faculdade de Ciências, Universidade do Porto, Rua do Campo Alegre, 4169-007 Porto, Portugal
   index: 1
 - name: Instituto de Astrofísica e Ciências do Espaço, Universidade do Porto, CAUP, Rua das Estrelas, 4150-762 Porto, Portugal
   index: 2
 - name: Centro de Astrobiología (CAB), CSIC-INTA, Camino Bajo del Castillo s/n, 28692, Villanueva de la Cañada (Madrid), Spain
   index: 3
date: 11 March 2026
bibliography: paper.bib

---

# Summary

AMATERASU (AutoMATic Equivalent-width Retrieval for Activity Signal Unveiling) is a Python tool to check for periods in spectral activity indices similar to an input period. This way, by running AMATERASU for a specific spectral line, the user can see if the input period may be correlated with activity. The aim of this program is to provide an easy and quick way to check for clues that the period observed in, for example, RV is of stellar origin. AMATERASU follows a methodology similar to @GomesdaSilva:2025 and is inspired by ACTIN [@GomesdaSilva:2018; @GomesdaSilva:2021].  

# Statement of need

Stellar variability can impact planetary signals detected via the radial velocity (RV) method. This is often addressed by tracking spectral lines sensitive to magnetic or/and temperature changes in the stellar atmosphere. 

For this, different methods are currently used. For example, one can examine the variability in the cross-correlation function (CCF) parameters. 
The use of individual (or doublets and triplets) spectral lines as proxies for stellar activity is also popular, either in the optical or in the near infrared. 
Regarding these last ones, one can measure activity related line distortions through properties like the full width at half maximum (FWHM), depth or bisector, or through overall changes in the flux in the line, like equivalent-widths (EW) or simple flux ratios.

With the exoplanet hunters community pushing through to find Earth twins, understanding activity indicators behavior is crucial, as their sensitivity may vary with stellar properties, to detrend stellar activity from the signal from a orbiting planet.

In this context, we built the AutoMATic Equivalent-width Retrieval for Activity Signal Unveiling code (AMATERASU). This tool enables users to efficiently compute pseudo equivalent-widths (pEW) of spectral lines, search for periodicities in their time-series and identify whether those periods match an input. 
By combining instrument independent inputs, automatic adjustments and batch analysis of multiple lines and periods, AMATERASU provides a user-friendly and quick way to validate the stellar or planetary origin of RV signals.

# State of the field   

Some tools already exist to compute spectral indices, like ACTIN [@GomesdaSilva:2018;@GomesdaSilva:2021] and iSTARMOD [@Labarga:2025;@Labarga:2026]. 
ACTIN computes the ratio between the flux inside a spectral line, delimited by a given central bandpass, and the flux in reference regions, usually the spectral continuum.
iSTARMOD uses the spectral subtraction technique, subtracting the total flux emitted by a line by its photospheric contribution, obtained via spectral synthesis from a reference star with the same properties. The final activity index is the EW of the residual profile.

While these tools are very useful for a variety of tasks, they only compute one version of a given activity index (with a fixed central bandpass), which may not be optimal to follow activity in different stars  (@GomesdaSilva:2022, Monteiro et al., in prep).
Additionally, when testing a new index with ACTIN, the user needs to define the best reference regions, which is not straightforward when dealing with spectra affected by telluric lines (or their correction residuals) or lacking well defined continuum (often the case in M dwarves spectra). 

Moreover, these tools do not provide a straightforward way to systematically test whether a given periodicity, such as a candidate planet signal, may be reproduced in activity indices.

AMATERASU fills this gap, allowing easy computation of spectral pEWs and search for periodicities in their time-series. 
In practice, AMATERASU tests if a given period can be obtained by adjusting different central bandpasses from a grid to a given target line, thereby finding, or not, if the period is also manifested through line deformations. 

# Software design

AMATERASU was designed to have a modular architecture, allowing the user to easily access the different classes of the tool (e.g., cleaning data, running GLS periodograms). It also comes with several tunable parameters, giving the user full control, while keeping an easy "blind" implementation.

AMATERASU computes the pseudo equivalent-width (pEW) of a spectral line in a normalization independent way, using bandpasses from a grid that goes from 0.1 \r{A} up to a user-defined maximum. 
When computing the pEW, the flux is interpolated with a step similar to original spectrum's step, aligning the spectrum portion with the bandpass edges, and the continuum level is approximated as a user-defined percentile of the flux around the spectral line.
Having retrieved a time series of pEWs measurements for a given bandpass, by default AMATERASU cleans the data by applying a 3-$\sigma$ sequential clipping and binning the data by night. 
Then, AMATERASU runs Generalized Lomb-Scargle (GLS) periodograms [@Zechmeister:2009] and extracts the significant peaks, the ones not close to peaks in the window function periodogram and with a false alarm probability (FAP) under the specified threshold. If any of the significant peaks is close within the specified threshold to the input period, the tool warns the user.
AMATERASU can also compute the Spearman correlation ranked coefficient and associated p-value between the pEW arrays and an input array, that can be, for example, a known activity indicator or the RV timeseries.

The basic input consists of a spectral data cube (time-series of 1D or 2D spectra), various analysis options and information on the spectral line (line center, maximum central bandpass window and continuum window).
The maximum central bandpass and continuum windows can be given either manually or computed automatically. 
In the automatic way, the spectra are coadded into a master spectrum, using the first spectrum as the common reference grid, which is then smoothed with a moving average. 
Then, the \texttt{find\_peaks} function from the \texttt{scipy} package [@Scipy2020] is used to find the target spectral line and retrieve its full-width-at-half-maximum (FWHM). 
The bandpass window is set as an integer multiple of the FWHM retrieved (by default 5 times) and the continuum window is 6 times that (30 times the FWHM). 

The user can choose one of the predefined lines in the ``ind_table.csv`` table or test a new index. 
The tool accepts a list of input periods and a list of lines to analyse, as well as 1D or 2D spectra. 
To be more flexible with the instruments tested, the spectra have to be given in an array format: for 2D spectra, the array should have dimensions of ($N_\text{obs}$,$N_\text{spectrum axis}$,$N_\text{orders}$,$N_\text{pixels}$), where $N_\text{spectrum axis}$ refers for the wavelength, flux and flux error of each spectrum. An array with the dates of observations in BJD is also needed.

AMATERASU has two output modes: "standard" and "full". The standard output is the fastest, as it only computes the pEW time-series and correspondent GLS periodograms, prints wether any of the periods is close to the input one and saves (if wanted) a \texttt{.csv} file with the matching periods and respective central bandpasses and a \texttt{.csv} file with the Spearman correlations with the input array per bandpass (if wanted). 
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

The code is available and will be regularly updated on [GitHub][] and can be easily installed using pip. We also provide a tutorial notebook.

[GitHub]: https://github.com/telmonteiro/AMATERASU

# Research impact statement

This tool could decrease false-positives arising from misidentified periodicities in RV and help improve both planet search surveys and our understanding of stellar variability. 
AMATERASU is already being in research (Monteiro et al., in prep; Soldevilla et al., in prep).

# AI usage disclosure

Generative AI tools were used during development for code completion and debugging. All AI-generated content was reviewed and refined by the authors. 

# References
