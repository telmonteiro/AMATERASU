import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
from specutils import Spectrum1D, manipulation
from scipy.stats import spearmanr
from astropy.nddata import StdDevUncertainty
import astropy.units as u

from amaterasu.time_series_clean import time_series_clean
from amaterasu.gls_periodogram import gls_periodogram
from amaterasu.SpectraTreat import SpectraTreat


#AutoMATic Equivalent-width Retrieval for Activity Signal Unveiling
class AMATERASU:
    """Computes a time series of Equivalent Width (EW) of a given spectral line for different bandpasses and retrieves the GLS periods and correlations with given array.

    Args:
        star (str): ID of stellar object.
        data (list): list where first element is the 1D array of BJD of observations (bjd_observations) and second element is array of spectral data (spectra_observations) with format (N_spectra, N_axis, N_orders, N_pixels). N_axis = 3 (wavelength, flux and flux error) and N_orders = 1 in case of 1D spectra.
        indice (list): list with identifiers of indices / spectral lines.
        indice_info (dict): Dictionary containing the line identifier ``ln_id``, the line center ``ln_ctr``, the maximum bandpass ``ln_win`` and the interpolation window ``total_win``. If only the ``ln_id`` is given, it computes the windows automatically. If None, fetches ind_table.csv.
        gls_options (list): list with options for GLS periodograms. Contains run_gls (bool) to run GLS periodograms or not, period_test (list of lists) list with period and error (in days) to test, ptol (float or list) tolerance to define a period as close to period_test, fap_treshold (float) maximum FAP to classify the period as significant and plot_gls (bool) to save GLS periodogram plot.
        correlation_options (list): list with options for correlation with given arrays. Contains run_correlation (bool) to run correlation or not, df_input (pd.DataFrame) with columns bjd, values and errors, cols_input (list) values in df_input, cols_err_input (list) columns of errors of values in df_input, abs_corr_threshold (float) minimum absolute correlation to be strong, pval_threshold (float) maximum p-value to be significant.
        output (str): 'standard' or 'full' output.
        fixed_bandpass (float): if user wants to test one specific bandpass.
        interp (bool): to interpolate the flux inside the interpolation window or not.
        plot_line (bool): spectral line plot.
        folder_path (str): path to save all plots and dataframes.
    """
    def __init__(self, star, data, indice, indice_info, gls_options, correlation_options, output="standard", fixed_bandpass=None, interp=True, plot_line=False, folder_path=None):

        print(f"AMATERASU instance created for {star}")

        if type(indice) != list: 
            indice = [indice]

        bjd_observations, spectra_observations = data
        run_gls = gls_options[0]
        run_correlation = correlation_options[0]

        for ind in indice:

            try:
                ln_ctr, ln_win, interp_win, spectra_obs = SpectraTreat(spectra_observations, ind, indice_info=indice_info, plot_line=plot_line).results
            except Exception as e:
                print(f"[ERROR] Could not treat spectra for {ind}: {e}")
                continue

            print("-"*80)
            print(f"Computing Equivalent-Widths.\nLine ID: {ind}\nLine center: {ln_ctr}\nNumber of spectra: {bjd_observations.shape[0]}")
            print("-"*80)
            print(f"Maximum Bandpass: {ln_win}\nInterpolation window: {interp_win}")
            print("-"*80)

            if fixed_bandpass: 
                bandpasses = np.array([fixed_bandpass])
            else: 
                bandpasses = np.arange(0.1,ln_win+0.1,0.1)

            ew = np.zeros((bjd_observations.shape[0],len(bandpasses)))
            ew_error = np.zeros((bjd_observations.shape[0],len(bandpasses)))
            bjd = np.zeros((bjd_observations.shape[0]))
            flux_continuum = np.zeros((bjd_observations.shape[0]))

            for i in tqdm.tqdm(range(bjd_observations.shape[0])):

                wave, flux, flux_error, bjd[i] = spectra_obs[i][0], spectra_obs[i][1], spectra_obs[i][2], bjd_observations[i]
    
                #continuum flux
                mask_c = (wave >= ln_ctr - interp_win/2) & (wave <= ln_ctr + interp_win/2)
                flux_c = np.percentile(np.array(flux, dtype=float)[mask_c][(np.isnan(np.array(flux, dtype=float)[mask_c])==False)], 85)
                flux_continuum[i] = flux_c
            
                for j,bp in enumerate(bandpasses):

                    if interp:
                        wave_i, flux_i, flux_err_i = self._spec_interpolation(wave,flux,flux_error,ln_ctr,bp,interp_win)
                    else:
                        mask = (wave >= ln_ctr-bp/2) & (wave <= ln_ctr+bp/2)    
                        wave_i, flux_i, flux_err_i = wave[mask], flux[mask], flux_error[mask]
                    
                    spec_interp = {"wave":wave_i, "flux":flux_i, "flux_err":flux_err_i}

                    ew[i,j], ew_error[i,j] = self._compute_EW(spec_interp, flux_c, ln_ctr, bp)

            ew_cols = [f"{ind}{int(round(bp*10,0)):02d}_EW" for bp in bandpasses]
            ew_error_cols = [f"{ind}{int(round(bp*10,0)):02d}_EW_error" for bp in bandpasses]

            df_raw = pd.DataFrame(np.column_stack([bjd, flux_continuum, ew, ew_error]),columns=["BJD","flux_continuum"] + ew_cols + ew_error_cols)
            df_clean = time_series_clean(df_raw, ew_cols, ew_error_cols, sigma=3).df

            if folder_path and output=="full":
                
                spec_lines_folder = f"{folder_path}/{star}/{ind}/"
                os.makedirs(spec_lines_folder, exist_ok=True)
                os.makedirs(spec_lines_folder+"periodograms/", exist_ok=True)

                df_clean.to_csv(spec_lines_folder+f"df_EW_{star}_{ind}.csv")

            else: 
                spec_lines_folder = None


            if run_gls:

                period_test, ptol, fap_treshold, plot_gls = gls_options[1:]
                print("Computing GLS Periodograms for all bandpasses.")

                gls_df, all_gls_df = self._compute_gls(star,ind,bandpasses,period_test,fap_treshold,df_clean,ew_cols,plot_gls,output,spec_lines_folder)

                if output=="full":

                    gls_df.to_csv(spec_lines_folder+f"periods_{star}_{ind}.csv")
                    all_gls_df.to_csv(spec_lines_folder+f"periods_full_{star}_{ind}.csv")

                if type(period_test[0]) != list: 
                    period_test = [period_test]

                df_flag = pd.DataFrame()

                for j, p_rot_val in enumerate(period_test):
                    
                    p_rot, _ = p_rot_val
                    
                    for i in range(len(gls_df.index)):
                        
                        if isinstance(ptol, list): 
                            ptol_i = ptol[j]
                        else: 
                            ptol_i = ptol

                        close_periods = []
                        close_FAPs = []

                        for k in range(len(gls_df["period"][i])):
 
                            if np.isclose(gls_df["period"][i][k], p_rot, atol=ptol_i):

                                close_periods.append(gls_df["period"][i][k])
                                close_FAPs.append(gls_df["FAP"][i][k])

                        if len(close_periods) > 0: 
                            
                            df_flag_row = pd.DataFrame({"bandpass": [gls_df["bandpass"][i]], 
                                                    "period": [tuple(close_periods)], 
                                                    "FAP": [tuple(close_FAPs)],      
                                                    "input_period": [p_rot]})
                        
                            df_flag = pd.concat([df_flag, df_flag_row], axis=0).reset_index(drop=True)
                
                if len(df_flag) > 0:

                    df_flag = df_flag.groupby(["bandpass", "period", "FAP"]).agg({"input_period": list}).reset_index()
                    df_flag = df_flag[["bandpass", "period", "FAP", "input_period"]]

                if len(df_flag["period"][0]) == 0:

                    print("No period similar to inputs detected!")
                    gls_df_cleaned = gls_df[~gls_df["period"].apply(lambda p: all(pd.isna(x) for x in p))]

                    if gls_df_cleaned.empty:
                    
                        print("No period found with FAP below threshold!")
                    
                    else:
                        optimal_bandpass, optimal_period, optimal_fap = max(zip(gls_df_cleaned["bandpass"], [p[0] for p in gls_df_cleaned["period"]], [f[0] for f in gls_df_cleaned["FAP"]]), key=lambda x: -x[2])
                        
                        print(f"Bandpass with lowest FAP: {optimal_bandpass}")
                        print(f"Period [d]: {optimal_period}\nFAP: {optimal_fap:.2e}")
                
                else:

                    df_flag_cleaned = df_flag[~df_flag["period"].apply(lambda p: all(pd.isna(x) for x in p))]
                    optimal_bandpass, optimal_period, optimal_fap = max(zip(df_flag_cleaned["bandpass"], [p[0] for p in df_flag_cleaned["period"]], [f[0] for f in df_flag_cleaned["FAP"]]), key=lambda x: -x[2]) 
                    
                    print(f"Bandpass with period near to input and lowest FAP: {optimal_bandpass}")
                    print(f"Period [d]: {optimal_period}\nFAP: {optimal_fap:.2e}")
                    print(df_flag)

                if folder_path:
                    if output == "full": 
                        folder = spec_lines_folder
                    else: 
                        folder = folder_path+"/"
                    df_flag.to_csv(folder+f"periods_found_{star}_{ind}.csv")

            plt.close("all")

            if run_correlation:

                df_input, cols_input, cols_err_input, abs_corr_threshold, pval_threshold = correlation_options[1:]

                print("-"*80)
                print("Computing Spearman correlations with input arrays.")

                df_correlations = self._compute_correlation(df_clean, bandpasses, ew_cols, df_input, cols_input, cols_err_input)

                if output=="full":
                    df_correlations.to_csv(spec_lines_folder+f"correlations_{star}_{ind}.csv")

                for j,input_col in enumerate(cols_input):

                    df_correlations_strong = pd.DataFrame()

                    for i in range(len(df_correlations.index)):

                        if df_correlations[f"pval_{input_col}"][i] < pval_threshold and np.abs(df_correlations[f"rho_{input_col}"][i]) >= abs_corr_threshold:
                            df_correlations_strong_row = pd.DataFrame({"bandpass": [df_correlations["bandpass"][i]], f"rho_{input_col}": [df_correlations[f"rho_{input_col}"][i]], f"pval_{input_col}": [df_correlations[f"pval_{input_col}"][i]]})
                            df_correlations_strong = pd.concat([df_correlations_strong,df_correlations_strong_row],axis=0).reset_index(drop=True)

                    #select bandpass of max |correlation|
                    if df_correlations_strong.empty == False:
                        optimal_bandpass, optimal_corr, optimal_pval = max(zip(df_correlations_strong["bandpass"],np.abs(df_correlations_strong[f"rho_{input_col}"]),df_correlations_strong[f"pval_{input_col}"]), key=lambda x: x[1])
                        print(rf"Bandpass with highest absolute correlation with {input_col}: {optimal_bandpass}")
                        print(f"Spearman correlation: {optimal_corr}\np-value: {optimal_pval:.2e}")
                        print("-"*80)

                        if folder_path:
                            if output == "full": 
                                folder = spec_lines_folder
                            else: 
                                folder = folder_path+"/"
                            df_correlations_strong.to_csv(folder+f"{input_col}_correlations_strong_{star}_{ind}.csv")
                

    def _compute_EW(self, spec_interp, flux_c, ln_ctr, ln_win):

        wave, flux, flux_err = spec_interp["wave"], spec_interp["flux"], spec_interp["flux_err"]
        
        mask_line = (wave >= ln_ctr - ln_win/2) & (wave <= ln_ctr + ln_win/2)        
        flux_lambda = flux[mask_line]
        flux_error_lambda = flux_err[mask_line]
        wave_lambda = wave[mask_line]

        delta_lambda = np.median(np.diff(wave_lambda))
        ew = np.sum((1- flux_lambda/flux_c) * delta_lambda)
        ew_error = 1/flux_c * np.sqrt( np.sum( flux_error_lambda**2 * delta_lambda**2 ) )

        return ew, ew_error
    

    def _spec_interpolation(self, wave,flux,flux_error,ln_ctr,ln_win,interp_win):

        lambda_max = ln_ctr+ln_win/2; lambda_min = ln_ctr-ln_win/2
        mask_interp = (wave >= ln_ctr-interp_win/2) & (wave <= ln_ctr+interp_win/2)    

        wave_win = wave[mask_interp]
        flux_win = flux[mask_interp]
        flux_error_win = flux_error[mask_interp]

        wstep = np.median(np.diff(wave_win))
        n_points = int((lambda_max-lambda_min)/wstep)+1
        x_interp = np.linspace(lambda_min, lambda_max, n_points)

        resampler = manipulation.SplineInterpolatedResampler()
        spec = Spectrum1D(spectral_axis=wave_win*u.AA, flux=flux_win*u.dimensionless_unscaled, uncertainty=StdDevUncertainty(flux_error_win*u.dimensionless_unscaled))
        spec_re = resampler(spec, x_interp * u.AA)

        return spec_re.wavelength.value, spec_re.flux.value, spec_re.uncertainty.array


    def _compute_gls(self,star,ind,bandpasses,period_test,fap_treshold,df_clean,ew_cols,plot_gls,output,spec_lines_folder):

            gls_list = []; gls_results_lists = []

            for i, col in enumerate(tqdm.tqdm(ew_cols)):

                df = df_clean[["BJD",col,col+"_error"]]
                df = df.dropna()
                
                x = np.asarray(df["BJD"])
                y = np.asarray(df[col])
                yerr = np.asarray(df[col+"_error"])

                t_span = max(x) - min(x)

                if output == "full": 
                    folder_path = spec_lines_folder
                else: 
                    folder_path = None

                gls_results = gls_periodogram(star, period_test, col, x, y, yerr, pmin=1.5, pmax=t_span, folder_path=folder_path).results

                gls_df = pd.DataFrame({key: [value] for key, value in gls_results.items()})
                gls_results_lists.append(gls_df)

                significant_periods = gls_results["sel_peaks_period"]
                significant_FAPs = gls_results["sel_peaks_FAP"]

                periods_fap_threshold = []
                FAPS_fap_threshold = []
                
                for j in range(len(significant_periods)):
 
                    if significant_FAPs[j] < fap_treshold:

                        periods_fap_threshold.append(significant_periods[j])
                        FAPS_fap_threshold.append(significant_FAPs[j])
                
                sorted_pairs = sorted(zip(FAPS_fap_threshold, periods_fap_threshold))

                if sorted_pairs:
                    FAPS_fap_threshold_sorted, periods_fap_threshold_sorted = zip(*sorted_pairs)
                    FAPS_fap_threshold_sorted = list(FAPS_fap_threshold_sorted)
                    periods_fap_threshold_sorted = list(periods_fap_threshold_sorted)
                else:
                    FAPS_fap_threshold_sorted = [np.nan]
                    periods_fap_threshold_sorted = [np.nan]

                if type(bandpasses[i]) == str:
                    bp = round(bandpasses[i],1)
                else:
                    bp = round(bandpasses[i],1)
                    
                gls_dic = pd.DataFrame({"bandpass":[bp], 
                                        "period":[periods_fap_threshold_sorted], 
                                        "FAP":[FAPS_fap_threshold_sorted]})
                gls_list.append(gls_dic)

            gls_df = pd.concat(gls_list, axis=0, ignore_index=True)
            all_gls_df = pd.concat(gls_results_lists, axis=0, ignore_index=True)

            if plot_gls and output=="full":
                gls_periodogram.GLS_plot(star, ind, period_test, bandpasses, df_clean, all_gls_df, spec_lines_folder)

            return gls_df, all_gls_df
    
    @staticmethod
    def _compute_correlation(df_ew, bandpasses, ew_cols, df_input_raw, cols_input, cols_err_input):
        
        df_input = time_series_clean(df_input_raw, cols_input, cols_err_input, sigma=3).df

        df_correlations = pd.DataFrame({"bandpass":np.around(bandpasses,1)})

        for col in cols_input:

            col_corr_list = []
            col_pval_list = []

            for col1 in ew_cols:

                rho, pval = spearmanr(df_ew[col1], df_input[col], nan_policy = "omit")
                col_corr_list.append(np.around(rho,5))
                col_pval_list.append(pval)

            df_correlations = pd.concat([df_correlations, pd.DataFrame({f"rho_{col}":col_corr_list,f"pval_{col}":col_pval_list})],axis=1)

        return df_correlations
