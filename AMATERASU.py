import pandas as pd, numpy as np, matplotlib.pyplot as plt, tqdm, os # type: ignore
from specutils.manipulation import SplineInterpolatedResampler #type: ignore
from specutils import Spectrum1D #type: ignore
from astropy.nddata import StdDevUncertainty #type: ignore
import astropy.units as u #type: ignore
from useful_funcs import bin_data, seq_sigma_clip
from gls_periodogram import gls_periodogram
from SpecFunc import SpecFunc
SpecFunc = SpecFunc()

#AutoMATic Equivalent-width Retrieval for Activity Signal Unveiling

class AMATERASU:
    """Computes a time series of Equivalent Width (EW) of a given spectral line for different bandpasses and retrieves the GLS periods.

    Args:
        files (list): List of 2D spectra paths.
        star (str): ID of stellar object.
        period_test (list): List with period and error (in days) to test.
        ptol (float): tolerance to define a period as close to period_test.
        fap_treshold (float): maximum FAP (%) to classify the period as significant.
        indice (str): identifier of indice / spectral line.
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

    def __init__(self, files, star, period_test, ptol, fap_treshold, indice, indice_info=None, interp=True, run_gls=True, plot_gls=False, plot_line=False, folder_path=None):

        print(f"AMATERASU instance created for {star}")

        if indice_info == None:
            ind_table = pd.read_csv("ind_table.csv")
            indice_info = ind_table[ind_table["ln_id"]==indice].to_dict(orient='records')[0]

        ln_ctr, ln_win, total_win = indice_info["ln_ctr"], indice_info["ln_win"], indice_info["total_win"]

        bandpasses = np.arange(0.1,ln_win+0.1,0.1)

        ew = np.zeros((len(files),len(bandpasses)))
        ew_error = np.zeros((len(files),len(bandpasses)))
        bjd = np.zeros((len(files)))

        print(f"Computing EWs for {len(files)} spectra.")

        for i, spec_file in enumerate(tqdm.tqdm(files)):

            wave, flux, flux_error, bjd_i = self.read_spectrum(spec_file, ln_ctr, ln_win)
            bjd[i] = bjd_i

            if interp:
                wave_i, flux_i, flux_err_i = self._spec_interpolation(wave,flux,flux_error,ln_ctr,total_win)
            else:
                lambda_max = ln_ctr+total_win/2; lambda_min = ln_ctr-total_win/2
                mask = (wave >= lambda_min) & (wave <= lambda_max)    
                wave_i, flux_i, flux_err_i = wave[mask], flux[mask], flux_error[mask]

            flux_c = np.percentile(flux_i[(np.isnan(flux_i)==False)], 90)
            spec_interp = {"wave_interp":wave_i, "flux_interp":flux_i, "flux_err_interp":flux_err_i}

            if plot_line:
                plt.figure(figsize=(9, 3.5))
                plt.title(f"{star} - {indice}")
                plt.errorbar(wave_i, flux_i, flux_err_i)
                plt.axhline(flux_c,lw=1,c="black")
                plt.xlabel(r"$\lambda$ [Å]")
                plt.ylabel("Flux")

            for j,bp in enumerate(bandpasses):

                ew[i,j], ew_error[i,j] = self.compute_EW(spec_interp, flux_c, ln_ctr, bp)

                if plot_line:
                    plt.axvspan(ln_ctr - bp / 2, ln_ctr + bp / 2, alpha=0.1, color='yellow', ec = "black", lw = 2)
            
            if plot_line:
                plt.show()

        ew_cols = [f"{indice}{int(bp * 10):02d}_EW" for bp in bandpasses]
        ew_error_cols = [f"{indice}{int(bp * 10):02d}_EW_error" for bp in bandpasses]

        df_raw = pd.DataFrame(np.column_stack([bjd, ew, ew_error]),columns=["BJD"] + ew_cols + ew_error_cols)
        df_clean = self.time_series_clean(df_raw, [f"{indice}{int(bp * 10):02d}" for bp in bandpasses], sigma=3)

        AMATERASU.EWs = df_clean
        AMATERASU.ew_cols = ew_cols
        AMATERASU.bandpasses = bandpasses
        AMATERASU.cols_names = [f"{indice}{int(bp * 10):02d}" for bp in bandpasses]

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
                gls = gls_periodogram(star = star, period_test = period_test, ind = col, 
                                    x = x, y = y, y_err=yerr, pmin=1.5, pmax=t_span, steps=1e6, verb = False, save=True, folder_path=spec_lines_folder)
                results = gls.run()

                gls_dic = pd.DataFrame({f"bandpass":[round(bandpasses[i],1)],
                                        f"period":[results["period_best"]],
                                        f"flag_period":[results["flag"]],
                                        f"fap_best":[results["fap_best"]]})
                gls_list.append(gls_dic)
    
                gls_df = pd.DataFrame({key: [value] for key, value in results.items()})
                gls_results_lists.append(gls_df)

                if results["fap_best"] < fap_treshold and np.isclose(results["period_best"], period_test[0], atol=ptol):
                    print(rf"Bandpass of {round(bandpasses[i],1)} A with period = {results['period_best']} d and FAP {results['fap_best']*100}%")
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

        wave_interp, flux_interp, flux_err_interp = spec_interp["wave_interp"], spec_interp["flux_interp"], spec_interp["flux_err_interp"]
        
        mask_line = (wave_interp >= ln_ctr - ln_win/2) & (wave_interp <= ln_ctr + ln_win/2)        
        flux_lambda = flux_interp[mask_line]
        flux_error_lambda = flux_err_interp[mask_line]
        wave_lambda = wave_interp[mask_line]

        delta_lambda = np.median(np.diff(wave_lambda))

        ew = np.sum((1- flux_lambda/flux_c) * delta_lambda)
        ew_error = 1/flux_c * np.sqrt( np.sum( flux_error_lambda**2 * delta_lambda**2 ) )

        return ew, ew_error

    def read_spectrum(self, spec_file, ln_ctr, ln_win):
        spectrum_raw, header_raw = SpecFunc._Read(spec_file, mode="vac")
        spectrum, header = SpecFunc._RV_correction(spectrum_raw, header_raw)
        spec_order = SpecFunc._spec_order(spectrum["wave"], ln_ctr, ln_win)
        return spectrum["wave"][spec_order], spectrum["flux"][spec_order], spectrum["flux_error"][spec_order], header["bjd"]

    def _spec_interpolation(self,wave,flux,flux_error,ind_ctr,interp_win):

        lambda_max = ind_ctr+interp_win/2; lambda_min = ind_ctr-interp_win/2
        mask = (wave >= lambda_min) & (wave <= lambda_max)    
        wave_win = wave[mask]
        flux_win = flux[mask]
        flux_error_win = flux_error[mask]

        wstep = np.median(np.diff(wave_win))
        n_points = int((lambda_max-lambda_min)/wstep)
        x_interp = np.linspace(lambda_min, lambda_max, n_points)

        resampler = SplineInterpolatedResampler('nan_fill')
        mask = np.isnan(flux_win) | np.isnan(flux_error_win)
        wave_win = wave_win[~mask]
        flux_win = flux_win[~mask]
        flux_error_win = flux_error_win[~mask]

        spec = Spectrum1D(spectral_axis=wave_win*u.AA, flux=flux_win*u.dimensionless_unscaled, uncertainty=StdDevUncertainty(flux_error_win*u.dimensionless_unscaled))
        spec_re = resampler(spec, x_interp * u.AA)

        return spec_re.wavelength.value, spec_re.flux.value, spec_re.uncertainty.array
    
    def time_series_clean(self, df_raw, inds, sigma=3):
        #sequential sigma clip
        for col in inds:
            df = seq_sigma_clip(df_raw, col+"_EW", sigma=sigma, show_plot=False) #indices values
            df = seq_sigma_clip(df, col+"_EW_error", sigma=sigma, show_plot=False) #indices error values

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
    def GLS_plot(star, p_rot_lit_list, cols, df_clean, all_gls_df):

        nrows = len(cols)
        fig, axes = plt.subplots(nrows=nrows,ncols=2,figsize=(14, 0.5+nrows*2), sharex="col")

        for i,col in enumerate(tqdm.tqdm(cols)):

            results = all_gls_df.iloc[i].to_dict()

            axes[i,0].errorbar(df_clean["BJD"] - 2450000, df_clean[col+"_EW"], df_clean[col+"_EW_error"], fmt="k.")
            axes[i,0].set_ylabel(rf"{col} - EW [$\AA$]", fontsize=13)
            if i == len(cols) - 1:
                axes[i,0].set_xlabel("BJD $-$ 2450000 [d]", fontsize=13)
            axes[i,0].tick_params(axis="both", direction="in", top=True, right=True, which='both')

            axes[i,1].semilogx(results["period"], results["power"], "k-")
            axes[i,1].plot([min(results["period"]), max(results["period"])], [results["fap_01"]] * 2,"--",color="black",lw=0.7)
            axes[i,1].plot([min(results["period"]), max(results["period"])], [results["fap_1"]] * 2,"--",color="black",lw=0.7)
            axes[i,1].plot([min(results["period"]), max(results["period"])], [results["fap_5"]] * 2,"--",color="black",lw=0.7)
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
        plt.show()