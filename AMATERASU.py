import pandas as pd, numpy as np, matplotlib.pyplot as plt, tqdm, os, gc # type: ignore
from scipy.interpolate import interp1d
from specutils.manipulation import SplineInterpolatedResampler #type: ignore
from specutils import Spectrum1D #type: ignore
from astropy.nddata import StdDevUncertainty #type: ignore
import astropy.units as u #type: ignore
from useful_funcs import bin_data, seq_sigma_clip
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

    def __init__(self, star, bjd_observations, spectra_observations, period_test, ptol, fap_treshold, indice, fixed_bandpass=None, indice_info=None, interp=True, run_gls=True, plot_gls=False, plot_line=False, folder_path=None):

        print(f"AMATERASU instance created for {star}")

        ln_ctr, ln_win, interp_win, spectra_obs = SpectraTreat(spectra_observations, indice, indice_info=indice_info).results

        if fixed_bandpass:
            bandpasses = np.array([fixed_bandpass])
        else:
            bandpasses = np.arange(0.1,ln_win+0.1,0.1)

        ew = np.zeros((bjd_observations.shape[0],len(bandpasses)))
        ew_error = np.zeros((bjd_observations.shape[0],len(bandpasses)))
        bjd = np.zeros((bjd_observations.shape[0]))

        print(f"Computing EWs for {bjd_observations.shape[0]} spectra.")

        for i in tqdm.tqdm(range(bjd_observations.shape[0])):

            wave, flux, flux_error, bjd[i] = spectra_obs[i][0], spectra_obs[i][1], spectra_obs[i][2], bjd_observations[i]
  
            mask_c = (wave >= ln_ctr - interp_win/2) & (wave <= ln_ctr + interp_win/2)
            flux_c = np.percentile(np.array(flux, dtype=float)[mask_c][(np.isnan(np.array(flux, dtype=float)[mask_c])==False)], 90)
        
            for j,bp in enumerate(bandpasses):

                if interp:
                    wave_i, flux_i, flux_err_i = self._spec_interpolation(wave,flux,flux_error,ln_ctr,bp,interp_win)
                else:
                    mask = (wave >= ln_ctr-bp/2) & (wave <= ln_ctr+bp/2)    
                    wave_i, flux_i, flux_err_i = wave[mask], flux[mask], flux_error[mask]
                
                spec_interp = {"wave":wave_i, "flux":flux_i, "flux_err":flux_err_i}

                ew[i,j], ew_error[i,j] = self.compute_EW(spec_interp, flux_c, ln_ctr, bp)

                if plot_line:
                    plt.figure(figsize=(9, 3.5))
                    plt.title(f"{star} - {indice}")
                    plt.xlabel(r"$\lambda$ [Å]")
                    plt.ylabel("Flux")
                    plt.axhline(flux_c,lw=1,c="black")
                    plt.errorbar(wave[mask_c], flux[mask_c], flux_error[mask_c])
                    plt.axvspan(ln_ctr - bp / 2, ln_ctr + bp / 2, alpha=0.1, color='yellow', ec = "black", lw = 2)
                    plt.errorbar(wave_i, flux_i, flux_err_i)
                    plt.show()
                    plt.close("all")
                    gc.collect()

        ew_cols = [f"{indice}{int(round(bp * 10,0)):02d}_EW" for bp in bandpasses]
        ew_error_cols = [f"{indice}{int(round(bp * 10,0)):02d}_EW_error" for bp in bandpasses]
        AMATERASU.bandpasses = bandpasses
        AMATERASU.ew_cols = ew_cols

        df_raw = pd.DataFrame(np.column_stack([bjd, ew, ew_error]),columns=["BJD"] + ew_cols + ew_error_cols)
        df_clean = self.time_series_clean(df_raw, ew_cols, ew_error_cols, sigma=3)
        AMATERASU.EWs = df_clean

        if run_gls:

            if folder_path:
                if not os.path.isdir(folder_path):
                    os.mkdir(folder_path)
                spec_lines_folder = f"{folder_path}/{indice}/"
                if not os.path.isdir(spec_lines_folder):
                    os.mkdir(spec_lines_folder)
            else:
                spec_lines_folder = None

            print("Computing GLS Periodograms.")

            gls_list = []; gls_results_lists = []; df_flag = pd.DataFrame()

            for i, col in enumerate(tqdm.tqdm(ew_cols)):
                df = df_clean[["BJD",col,col+"_error"]]
                df = df.dropna()

                x = np.asarray(df["BJD"])
                y = np.asarray(df[col])
                yerr = np.asarray(df[col+"_error"])

                t_span = max(x) - min(x)
                gls_results = gls_periodogram(star, period_test, col, x, y, yerr, pmin=1.5, pmax=t_span, save=True, folder_path=spec_lines_folder).results

                gls_dic = pd.DataFrame({"bandpass":[round(bandpasses[i],1)], "period":[gls_results["period_best"]], "flag_period":[gls_results["flag"]], "fap_best":[gls_results["fap_best"]]})
                gls_list.append(gls_dic)
                gls_df = pd.DataFrame({key: [value] for key, value in gls_results.items()})
                gls_results_lists.append(gls_df)

                if gls_results["fap_best"] < fap_treshold and np.isclose(gls_results["period_best"], period_test[0], atol=ptol):
                    print(rf"Bandpass of {round(bandpasses[i],1)} A with period = {gls_results['period_best']} d and FAP {gls_results['fap_best']*100}%")
                    df_flag = pd.concat([df_flag,pd.DataFrame(gls_dic)],axis=0).reset_index(drop=True)
                
            if gls_list:
                gls_df = pd.concat(gls_list, axis=0, ignore_index=True)
                all_gls_df = pd.concat(gls_results_lists, axis=0, ignore_index=True)
            
            AMATERASU.gls_results = gls_df
            AMATERASU.gls_results_all = all_gls_df
            AMATERASU.periods_flagged = df_flag

            if plot_gls:
                self.GLS_plot(star, period_test, ew_cols, df_clean, all_gls_df)


    def compute_EW(self, spec_interp, flux_c, ln_ctr, ln_win):

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

    
    def time_series_clean(self, df_raw, inds, inds_errors, sigma=3):
        #sequential sigma clip
        for j in range(len(inds)):
            df = seq_sigma_clip(df_raw, inds[j], sigma=sigma, show_plot=False) #indices values
            df = seq_sigma_clip(df, inds_errors[j], sigma=sigma, show_plot=False) #indices error values

        #binning the data to days
        table = pd.DataFrame()
        bin_bjd = None

        for i, column in enumerate(df.columns):
            if pd.api.types.is_numeric_dtype(df[column]):
                #if the column contains numeric values, apply binning
                bin_bjd, bin_column_values, _ = bin_data(df["BJD"], df[column])
                table[column] = bin_column_values
                if bin_bjd is not None and i == 0:
                    table["BJD"] = bin_bjd
            else:
                table[column] = df[column]

        df_clean = table.apply(pd.to_numeric, errors='ignore')

        return df_clean
    

    @staticmethod
    def GLS_plot(star, ind, p_rot_lit_list, cols, df_clean, all_gls_df, folder):

        nrows = len(cols)
        fig, axes = plt.subplots(nrows=nrows,ncols=2,figsize=(14, 0.5+nrows*2), sharex="col")

        for i,col in enumerate(tqdm.tqdm(cols)):

            results = all_gls_df.iloc[i].to_dict()

            axes[i,0].errorbar(df_clean["BJD"] - 2450000, df_clean[col], df_clean[col+"_error"], fmt="k.")
            axes[i,0].set_ylabel(rf"{col[:-3]} - EW [$\AA$]", fontsize=13)
            if i == len(cols) - 1:
                axes[i,0].set_xlabel("BJD $-$ 2450000 [d]", fontsize=13)
            axes[i,0].tick_params(axis="both", direction="in", top=True, right=True, which='both')

            axes[i,1].semilogx(results["period"], results["power"], "k-")
            axes[i,1].plot([min(results["period"]), max(results["period"])], [results["fap_01"]] * 2,"--",color="black",lw=0.7)
            axes[i,1].plot([min(results["period"]), max(results["period"])], [results["fap_1"]] * 2,"--",color="black",lw=0.7)
            axes[i,1].set_ylabel("Norm. Power", fontsize=13)
            axes[i,1].set_xlabel("Period [d]", fontsize=13)
            axes[i,1].tick_params(axis="both", direction="in", top=True, right=True, which='both')

            for p_rot_val in p_rot_lit_list:
                p_rot, p_rot_err = p_rot_val
                axes[i,1].axvspan(p_rot - p_rot_err, p_rot + p_rot_err, alpha=0.6, color="orange")
                
            axes[i,1].text(0.05, 0.9,f"P = {np.around(results['period_best'], 1)} d ({results['flag']})",
                        fontsize=13,transform=axes[i,1].transAxes,verticalalignment="top")

        fig.subplots_adjust(hspace=0.0)
        fig.text(0.13, 0.89, f"{star}", fontsize=17)
        plt.savefig(folder+f"{star}_{ind}_GLS.pdf",dpi=1000, bbox_inches="tight")
        