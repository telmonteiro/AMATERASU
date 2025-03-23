import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from SpecFunc import SpecFunc
SpecFunc = SpecFunc()

from AMATERASU import AMATERASU

star = "GJ285"

#gather spectra
_, files = SpecFunc._gather_spectra(star_name=star, instrument="NIRPS",type="2d")

spectra_observations = []
bjd_observations = np.zeros((len(files)))

fwhm_array = np.zeros((len(files)))
fwhm_err_array = np.zeros((len(files)))

for i, file in enumerate(files):

    spectrum_raw, header_raw = SpecFunc._Read(file, mode="vac")
    spectrum, header = SpecFunc._RV_correction(spectrum_raw, header_raw)

    spectra_observations.append([spectrum["wave"], spectrum["flux"], spectrum["flux_error"]])

    bjd_observations[i] = header["bjd"]

    fwhm_array[i] = header["fwhm"]
    fwhm_err_array[i] = header["fwhm_err"]

spectra_observations = np.array(spectra_observations, dtype=object) 

#------------------------------------------------------------------------------
#options for AMATERASU

data = [bjd_observations,spectra_observations]


run_gls = True
period_test = [[2.8,0.5],[3,0.8],[5,0.1]]
ptol = 0.5
fap_treshold = 0.001
plot_gls=True
gls_options = [run_gls, period_test, ptol, fap_treshold, plot_gls]


run_correlation=True
df_input = pd.DataFrame({"BJD":bjd_observations,"fwhm":fwhm_array,"fwhm_err":fwhm_err_array})
cols_input = ["fwhm"]
cols_err_input = ["fwhm_err"]
abs_corr_threshold=0.4
pval_threshold=0.001
correlation_options = [run_correlation, df_input, cols_input, cols_err_input, abs_corr_threshold, pval_threshold]


indices = ["KIa","KIb","KIc","FeI"]

indice_info = {"KIa":{'ln_ctr':11772.862},
               "KIb":{'ln_ctr':12435.647},
               "KIc":{'ln_ctr':12525.544},
               "FeI":{'ln_ctr':11693.408}
               }

#------------------------------------------------------------------------------

amaterasu = AMATERASU(star, data, indices, indice_info, 
                      gls_options, correlation_options,
                      output="full", 
                      fixed_bandpass=None, interp=True, plot_line=True, folder_path=f"amaterasu_tests")