import pandas as pd, numpy as np, matplotlib.pyplot as plt, tqdm, os, gc # type: ignore
from specutils.manipulation import SplineInterpolatedResampler #type: ignore
from specutils import Spectrum1D #type: ignore
from astropy.nddata import StdDevUncertainty #type: ignore
import astropy.units as u #type: ignore
from time_series_clean import time_series_clean
from gls_periodogram import gls_periodogram
from SpectraTreat import SpectraTreat

#AutoMATic Equivalent-width Retrieval for Activity Signal Unveiling
class AMATERASU:
    """Computes a time series of Equivalent Width (EW) of a given spectral line for different bandpasses and retrieves the GLS periods.

    Args:
        star (str): ID of stellar object.
        bjd_observations (numpy array): 1D array of BJD of observations.
        spectra_observations (numpy array): array of spectral data with format (N_spectra, N_axis, N_orders, N_pixels). N_axis = 3 (wavelength, flux and flux error) and N_orders = 1 in case of 1D spectra.
        period_test (list): List with period and error (in days) to test.
        ptol (float): tolerance to define a period as close to period_test.
        fap_treshold (float): maximum FAP (%) to classify the period as significant.
        indice (str): identifier of indice / spectral line.
        fixed_bandpass (float): if user wants to test one specific bandpass.
        indice_info (dict): Dictionary containing the line identifier ``ln_id``, the line center ``ln_ctr``, the maximum bandpass ``ln_win`` and the interpolation window ``total_win``. If None, fetches ind_table.csv.
        interp (bool): to interpolate the flux inside the interpolation window or not.
        run_gls (bool): run GLS periodograms or not.
        plot_gls (bool): GLS periodogram plot.
        plot_line (bool): spectral line plot.
        folder_path (str): path to save GLSPs auxiliar plots.

    Attributes:
        EWs (pandas DataFrame): data frame with the cleaned EW for each bandpass and the BJD array.
        ew_cols (list): List of EW columns.
        gls_results (pandas DataFrame): concise data frame with the bandpass, period, flag and FAP.
        gls_results_all (pandas DataFrame): extensive results of the GLS periodograms.
        periods_flagged (pandas DataFrame): data frame with the rows of gls_results where the period is close to period_test.
        bandpasses (numpy array): array of bandpasses tested.
        cols_names (list): List of columns without the EW segment.
    """
    def __init__(self, star, bjd_observations, spectra_observations, period_test, ptol, fap_treshold, indice, output="standard", fixed_bandpass=None, indice_info=None, interp=True, run_gls=True, plot_gls=False, plot_line=False, folder_path=None):

        print(f"AMATERASU instance created for {star}")

        if type(indice) != list: indice = [indice]

        for ind in indice:

            ln_ctr, ln_win, interp_win, spectra_obs = SpectraTreat(spectra_observations, ind, indice_info=indice_info).results
            print(ln_win,interp_win)

            if fixed_bandpass: bandpasses = np.array([fixed_bandpass])
            else: bandpasses = np.arange(0.1,ln_win+0.1,0.1)

            ew = np.zeros((bjd_observations.shape[0],len(bandpasses)))
            ew_error = np.zeros((bjd_observations.shape[0],len(bandpasses)))
            bjd = np.zeros((bjd_observations.shape[0]))

            print(f"Computing {ind} EWs for {bjd_observations.shape[0]} spectra.")

            for i in tqdm.tqdm(range(bjd_observations.shape[0])):

                wave, flux, flux_error, bjd[i] = spectra_obs[i][0], spectra_obs[i][1], spectra_obs[i][2], bjd_observations[i]
    
                # Continuum flux
                mask_c = (wave >= ln_ctr - interp_win/2) & (wave <= ln_ctr + interp_win/2)
                flux_c = np.percentile(np.array(flux, dtype=float)[mask_c][(np.isnan(np.array(flux, dtype=float)[mask_c])==False)], 90)
            
                for j,bp in enumerate(bandpasses):

                    if interp:
                        wave_i, flux_i, flux_err_i = self._spec_interpolation(wave,flux,flux_error,ln_ctr,bp,interp_win)
                    else:
                        mask = (wave >= ln_ctr-bp/2) & (wave <= ln_ctr+bp/2)    
                        wave_i, flux_i, flux_err_i = wave[mask], flux[mask], flux_error[mask]
                    
                    spec_interp = {"wave":wave_i, "flux":flux_i, "flux_err":flux_err_i}

                    ew[i,j], ew_error[i,j] = self._compute_EW(spec_interp, flux_c, ln_ctr, bp)

                    if plot_line:
                        plt.figure(figsize=(9, 3.5))
                        plt.title(f"{star} - {ind}")
                        plt.xlabel(r"$\lambda$ [Å]")
                        plt.ylabel("Flux")
                        plt.axhline(flux_c,lw=1,c="black")
                        plt.errorbar(wave[mask_c], flux[mask_c], flux_error[mask_c])
                        plt.axvspan(ln_ctr - bp / 2, ln_ctr + bp / 2, alpha=0.1, color='yellow', ec = "black", lw = 2)
                        plt.errorbar(wave_i, flux_i, flux_err_i)
                        plt.show()
                        plt.close("all")
                        gc.collect()

            ew_cols = [f"{ind}{int(round(bp*10,0)):02d}_EW" for bp in bandpasses]
            ew_error_cols = [f"{ind}{int(round(bp*10,0)):02d}_EW_error" for bp in bandpasses]

            df_raw = pd.DataFrame(np.column_stack([bjd, ew, ew_error]),columns=["BJD"] + ew_cols + ew_error_cols)
            df_clean = time_series_clean(df_raw, ew_cols, ew_error_cols, sigma=3).df

            if folder_path and output=="full":
                if not os.path.isdir(folder_path): os.mkdir(folder_path)
                spec_lines_folder = f"{folder_path}/{ind}/"
                if not os.path.isdir(spec_lines_folder): os.mkdir(spec_lines_folder)
                if not os.path.isdir(spec_lines_folder+"periodograms/"): os.mkdir(spec_lines_folder+"periodograms/")
                df_clean.to_csv(spec_lines_folder+f"df_EW_{star}_{ind}.csv")
            else: 
                spec_lines_folder = None

            if run_gls:
                print("Computing GLS Periodograms.")
                gls_df, all_gls_df = self._compute_gls(star,ind,bandpasses,period_test,df_clean,ew_cols,plot_gls,output,spec_lines_folder)

                if output=="full":
                    gls_df.to_csv(spec_lines_folder+f"periods_{star}_{ind}.csv")
                    all_gls_df.to_csv(spec_lines_folder+f"periods_full_{star}_{ind}.csv")

                if type(period_test[0]) != list: period_test = [period_test]

                df_flag = pd.DataFrame()
                for j,p_rot_val in enumerate(period_test):
                    p_rot, _ = p_rot_val
                    for i in range(len(gls_df.index)):
                        if type(ptol) == list: ptol_i = ptol[j]
                        else: ptol_i = ptol
                        if gls_df["FAP"][i] < fap_treshold and np.isclose(gls_df["period"][i], p_rot, atol=ptol_i):
                            df_flag_row = pd.DataFrame({"bandpass": [gls_df["bandpass"][i]], "period": [gls_df["period"][i]], "FAP": [gls_df["FAP"][i]], "input_period": [p_rot]})
                            df_flag = pd.concat([df_flag,df_flag_row],axis=0).reset_index(drop=True)
                
                df_flag = df_flag.groupby(["bandpass", "period", "FAP"]).agg({"input_period": list}).reset_index()
                df_flag = df_flag[["bandpass", "period", "FAP", "input_period"]]
                print(df_flag)
                if folder_path:
                    df_flag.to_csv(spec_lines_folder+f"periods_found_{star}_{ind}.csv")
                


            

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

        resampler = SplineInterpolatedResampler()
        spec = Spectrum1D(spectral_axis=wave_win*u.AA, flux=flux_win*u.dimensionless_unscaled, uncertainty=StdDevUncertainty(flux_error_win*u.dimensionless_unscaled))
        spec_re = resampler(spec, x_interp * u.AA)

        return spec_re.wavelength.value, spec_re.flux.value, spec_re.uncertainty.array


    def _compute_gls(self,star,ind,bandpasses,period_test,df_clean,ew_cols,plot_gls,output,spec_lines_folder):

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

                gls_dic = pd.DataFrame({"bandpass":[round(bandpasses[i],1)], "period":[gls_results["period_best"]], "FAP":[gls_results["fap_best"]]})
                gls_list.append(gls_dic)
                gls_df = pd.DataFrame({key: [value] for key, value in gls_results.items()})
                gls_results_lists.append(gls_df)

            gls_df = pd.concat(gls_list, axis=0, ignore_index=True)
            all_gls_df = pd.concat(gls_results_lists, axis=0, ignore_index=True)

            if plot_gls and output=="full":
                gls_periodogram.GLS_plot(star, ind, period_test, ew_cols, df_clean, all_gls_df, spec_lines_folder)

            return gls_df, all_gls_df