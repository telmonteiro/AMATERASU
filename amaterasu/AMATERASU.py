import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tqdm
import os

from specutils import Spectrum, manipulation
from scipy.stats import spearmanr
from astropy.nddata import StdDevUncertainty
import astropy.units as u

from .time_series_clean import time_series_clean
from .gls_periodogram import gls_periodogram
from .SpectraTreat import SpectraTreat


class AMATERASU:
    """Main class of the AMATERASU package (AutoMATic Equivalent-width Retrieval for Activity Signal Unveiling).    
    Computes a time series of pseudo-Equivalent Width (EW) measurements of a given spectral line with different central 
    bandpasses and retrieves the GLS periods and Spearman correlations with a given array.

    Parameters
    ----------
    star : str 
        ID of target star.
    data : list 
        List where first element is the 1D array of BJD of observations (bjd_observations) and second element is array of 
        spectral data (spectra_observations) with format (N_spectra, N_axis, N_orders, N_pixels). 
        N_axis = 3 (wavelength, flux and flux error) and N_orders = 1 in case of 1D spectra.
    indices : list
        List or identifier of indices / spectral lines.
    indices_info : dict
        Dictionary containing the line identifier ``ln_id``, the line center ``ln_ctr``, the maximum bandpass ``ln_win`` 
        and the interpolation window ``total_win``. If only the ``ln_id`` is given, it computes the windows automatically. 
        If None, fetches ind_table.csv.
    fixed_bandpass : float, optional
        Central bandpass if user wants to test one specific bandpass.
    percentile_cont : float, optional
        Percentile for continuum estimation (default: 75).
    automatic_windows_mult : list, optional
        List with multipliers for automatic window calculation, first element for line window and second for interpolation 
        window (default: [5,6]).
    
    Methods
    -------
    run(gls_options, correlation_options, output="standard", interp=True, plot_line=False, sigma_clip=True, bin_night=True, folder_path=None)
        Runs the full process of computing EWs, GLS periodograms and correlations with given options.
    _compute_continuum(spectrum)
        Computes the continuum flux level using a specified mask.
    _compute_EW(spec_interp, flux_c, bp)
        Computes the pseudo-Equivalent Width and its error for a given spectrum.
    _spec_interpolation(spec, bp)
        Interpolates the spectrum around the line center with a given interpolation window.
    _compute_gls(star, index, bandpasses, period_test, fap_treshold, df_clean, ew_cols, plot_gls, output, spec_lines_folder)
        Computes GLS periodograms for the EW time series and identifies significant periods.
    _compute_correlation(df_ew, bandpasses, ew_cols, df_input_raw, cols_input, cols_err_input, sigma_clip, bin_night)
        Computes Spearman correlations between the EW time series and given input arrays.
    plot_gls_and_correlations(index, plot, folder_path, gls_plot_type="highest peak", gls_y_lims=[75,85])
        Plots the main peak of GLS periodograms and Spearman correlations in function of central bandpasses for a given index.
    """
    def __init__(self, star:str, data:list, indices:list, indices_info:dict, fixed_bandpass:float=None, percentile_cont:float=75, automatic_windows_mult:list=[5,6]):

        print(f"AMATERASU instance created for {star}")

        self.star = star
        self.data = data

        self.indices = indices
        self.indices_info = indices_info

        self.fixed_bandpass = fixed_bandpass

        self.percentile_cont = percentile_cont
        self.automatic_windows = automatic_windows_mult


    def run(self, gls_options:dict, correlation_options:dict, output:str="standard", interp:bool=True, plot_line:str="simple", sigma_clip:int=3, bin_night:bool=True, folder_path:str=None):
        """Runs the full process of computing EWs, GLS periodograms and correlations with given options.

        Parameters
        ----------
        gls_options : dict
            Dictionary with keys "run", "period_test", "period_tolerance", and "fap_threshold".
        correlation_options : dict
            Dictionary with keys "run", "df_input", "cols_input", "cols_err_input", "abs_corr_threshold", and "pval_threshold".
            for correlation, third element is list of column names of input arrays, fourth element is list of column names of 
            errors of input arrays, fifth element is absolute correlation threshold and sixth element is p-value threshold 
            for significant correlations.
        output : str, optional
            Level of output to save, "standard" for only final results and "full" for all intermediate results and plots (default: "standard").
        interp : bool, optional
            Whether to interpolate the spectra around the line center before computing EWs (default: True).
        plot_line : str, optional
            Whether to plot the coadded and smoothed spectra line with the defined windows. Options are "simple" for a simple plot of the coadded line with the windows, and "complete" for a more complete plot with the detected peaks and the defined windows (default: "simple").
        sigma_clip : int, optional
            Number of standard deviations to use for sigma clipping of the EW time series (default: 3). If None, does not apply.
        bin_night : bool, optional
            Whether to bin the EW time series by night (default: True).
        folder_path : str, optional
            Path to folder where to save outputs if output is "full" (default: None).
        
        Attributes
        ----------
        df_EW : pandas DataFrame
            DataFrame containing the time series of EWs and their errors for each bandpass.
        gls_df : pandas DataFrame
            DataFrame containing the significant periods and their FAPs for each bandpass.
        df_correlations : pandas DataFrame
            DataFrame containing the Spearman correlation coefficients and p-values between the EW time series and input arrays for each bandpass.
        """        
        bjd_observations, spectra_observations = self.data

        run_gls = gls_options["run"]
        run_correlation = correlation_options["run"]

        for index in self.indices:

            try:
                results = SpectraTreat(spectra_observations, 
                                       index, 
                                       index_info=self.indices_info[index],
                                       percentile_cont=self.percentile_cont, 
                                       automatic_windows_mult=self.automatic_windows, 
                                       plot_line=plot_line,
                                       folder=folder_path,
                                       star=self.star).results
                
                self.ln_ctr = results["ln_ctr"]
                self.ln_win = results["ln_win"]
                self.interp_win = results["interp_win"]
                self.spectra_obs = results["spectra_obs"]
            
            except Exception as e:
                print(f"[ERROR] Could not treat spectra for {index}: {e}")
                continue

            print("-"*100)
            print(f"Computing Equivalent-Widths.")
            print(f"Line ID: {index}")
            print(f"Line center: {self.ln_ctr}")
            print(f"Number of spectra: {bjd_observations.shape[0]}")
            print("-"*100)
            print(f"Maximum Bandpass: {self.ln_win}")
            print(f"Interpolation window: {self.interp_win}")
            print("-"*100)

            if self.fixed_bandpass: # manually given bandpasses
                bandpasses = np.array([self.fixed_bandpass])
            else: 
                bandpasses = np.arange(0.1, self.ln_win+0.1, 0.1)

            ew = np.zeros((bjd_observations.shape[0],len(bandpasses)))
            ew_error = np.zeros((bjd_observations.shape[0],len(bandpasses)))
            bjd = np.zeros((bjd_observations.shape[0]))
            flux_continuum = np.zeros((bjd_observations.shape[0]))

            for i in tqdm.tqdm(range(bjd_observations.shape[0])):

                wave = self.spectra_obs[i][0]
                flux = self.spectra_obs[i][1]
                flux_error = self.spectra_obs[i][2]
                bjd[i] = bjd_observations[i]
    
                # continuum flux estimation
                flux_c = self._compute_continuum(self.spectra_obs[i])
                flux_continuum[i] = flux_c
            
                for j, bp in enumerate(bandpasses):

                    if interp:
                        wave_i, flux_i, flux_err_i = self._spec_interpolation(self.spectra_obs[i], bp)
                    else:
                        mask = (wave >= self.ln_ctr-bp/2) & (wave <= self.ln_ctr+bp/2)    
                        wave_i = wave[mask]
                        flux_i = flux[mask]
                        flux_err_i = flux_error[mask]
                    
                    spec_interp = {"wave":wave_i, "flux":flux_i, "flux_err":flux_err_i}

                    ew[i,j], ew_error[i,j] = self._compute_EW(spec_interp, flux_c, bp)

            ew_cols = [f"{index}{int(round(bp*10,0)):02d}_EW" for bp in bandpasses]
            ew_cols_err = [f"{col}_error" for col in ew_cols]

            # construct data frames, clean it and store
            df_raw = pd.DataFrame(np.column_stack([bjd, flux_continuum, ew, ew_error]), columns=["BJD","flux_continuum"]+ew_cols+ew_cols_err)

            df_clean = time_series_clean(df_raw, list(zip(ew_cols, ew_cols_err)), True if sigma_clip else False, bin_night, sigma=sigma_clip).df

            self.df_raw = df_raw
            self.df_EW = df_clean

            if folder_path and output=="full":
                
                spec_lines_folder = f"{folder_path}/{self.star}/{index}/"
                os.makedirs(spec_lines_folder, exist_ok=True)
                os.makedirs(spec_lines_folder+"periodograms/", exist_ok=True)

                df_clean.to_csv(spec_lines_folder+f"df_EW_{self.star}_{index}.csv")
            else: 
                spec_lines_folder = None


            if run_gls:

                period_test = gls_options["period_test"]
                ptol = gls_options["ptol"]
                fap_treshold = gls_options["fap_treshold"]
                plot_gls = gls_options["plot_gls"]

                print("Computing GLS Periodograms for all bandpasses.")

                gls_df, all_gls_df = self._compute_gls(self.star, index, bandpasses, period_test, 
                                                       fap_treshold, df_clean, ew_cols, plot_gls, 
                                                       output, spec_lines_folder)

                if output=="full":

                    gls_df.to_csv(spec_lines_folder+f"periods_{self.star}_{index}.csv")
                    all_gls_df.to_csv(spec_lines_folder+f"periods_full_{self.star}_{index}.csv")

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

                if len(df_flag) == 0:
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
                    print(f"Number of bandpasses that detect period near to input: {len(df_flag)}")
                    print(df_flag)

                if folder_path:
                    if output == "full": 
                        folder = spec_lines_folder
                    else: 
                        folder = folder_path+"/"
                    df_flag.to_csv(folder+f"periods_found_{self.star}_{index}.csv")

            plt.close("all")

            if run_correlation:
                
                df_input = correlation_options["df_input"]
                cols_input = correlation_options["cols_input"]
                cols_err_input = correlation_options["cols_err_input"]
                abs_corr_threshold = correlation_options["abs_corr_threshold"]
                pval_threshold = correlation_options["pval_threshold"]

                print("-"*80)
                print("Computing Spearman correlations with input arrays.")

                df_correlations = self._compute_correlation(df_clean, bandpasses, ew_cols, df_input, cols_input, cols_err_input, sigma_clip, bin_night)

                if output=="full":
                    df_correlations.to_csv(spec_lines_folder+f"correlations_{self.star}_{index}.csv")

                for j, input_col in enumerate(cols_input):

                    df_correlations_strong = pd.DataFrame()

                    for i in range(len(df_correlations.index)):

                        if df_correlations[f"pval_{input_col}"][i] < pval_threshold and np.abs(df_correlations[f"rho_{input_col}"][i]) >= abs_corr_threshold:
                            df_correlations_strong_row = pd.DataFrame({"bandpass": [df_correlations["bandpass"][i]], f"rho_{input_col}": [df_correlations[f"rho_{input_col}"][i]], f"pval_{input_col}": [df_correlations[f"pval_{input_col}"][i]]})
                            df_correlations_strong = pd.concat([df_correlations_strong,df_correlations_strong_row],axis=0).reset_index(drop=True)

                    # select bandpass of max |correlation|
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
                            df_correlations_strong.to_csv(folder+f"{input_col}_correlations_strong_{self.star}_{index}.csv")


    def _compute_continuum(self, spectrum:np.ndarray):
        """Computes the continuum flux level using a specified mask.

        Parameters
        ----------
        spectrum : np.ndarray
            Spectrum with wavelength at index 0 and flux at index 1.
        
        Returns
        -------
        flux_c : float
            Estimated continuum flux level.
        """
        wave = spectrum[0]
        flux = spectrum[1]

        mask_c = ((wave >= self.ln_ctr - self.interp_win/2) &
                (wave <= self.ln_ctr + self.interp_win/2) &
                ((wave < self.ln_ctr - self.ln_win/2) | (wave > self.ln_ctr + self.ln_win/2)))

        flux_float_c = np.array(flux, dtype=float)[mask_c]
        flux_float_c = flux_float_c[(np.isnan(flux_float_c)==False)]

        flux_c = np.percentile(flux_float_c, self.percentile_cont)

        return flux_c


    def _spec_interpolation(self, spec:np.ndarray, bp:float):
        """Interpolates the spectrum around the line center with a given interpolation window.
        
        Parameters
        ----------
        spec : np.ndarray
            Spectrum to interpolate with wavelength at index 0, flux at index 1, and flux error at index 2.
        bp : float
            Bandpass width for EW computation.
        
        Returns
        -------
        wave_interp : array-like
            Interpolated wavelength array.
        flux_interp : array-like
            Interpolated flux array.
        flux_error_interp : array-like
            Interpolated flux error array.
        """
        wave = spec[0]
        flux = spec[1]
        flux_error = spec[2]

        lambda_max = self.ln_ctr+bp/2
        lambda_min = self.ln_ctr-bp/2

        mask_interp = (wave >= self.ln_ctr-self.interp_win/2) & (wave <= self.ln_ctr+self.interp_win/2)    

        wave_win = wave[mask_interp]
        flux_win = flux[mask_interp]
        flux_error_win = flux_error[mask_interp]

        wstep = np.median(np.diff(wave_win))
        n_points = int((lambda_max-lambda_min)/wstep)+1
        x_interp = np.linspace(lambda_min, lambda_max, n_points)

        resampler = manipulation.SplineInterpolatedResampler()
        spec = Spectrum(spectral_axis=wave_win*u.AA, flux=flux_win*u.dimensionless_unscaled, uncertainty=StdDevUncertainty(flux_error_win*u.dimensionless_unscaled))
        spec_re = resampler(spec, x_interp * u.AA)

        wave_interp = spec_re.wavelength.value
        flux_interp = spec_re.flux.value
        flux_error_interp = spec_re.uncertainty.array

        return wave_interp, flux_interp, flux_error_interp
    

    def _compute_EW(self, spec_interp:dict, flux_c:float, bp:float):
        """Computes the pseudo-Equivalent Width and its error.
        
        Parameters
        ----------
        spec_interp : dict
            interpolated spectrum around the line center with keys "wave", "flux" and "flux_err".
        flux_c : float
            Continuum flux level.
        bp : float
            Bandpass width for EW computation.
        
        Returns
        -------
        ew : float
            Computed pseudo-Equivalent Width for the given spectrum and bandpass.
        ew_error : float
            Estimated error of the computed pseudo-Equivalent Width.
        """
        wave = spec_interp["wave"]
        flux = spec_interp["flux"]
        flux_err = spec_interp["flux_err"]
        
        mask_line = (wave >= self.ln_ctr - bp/2) & (wave <= self.ln_ctr + bp/2)        
        flux_lambda = flux[mask_line]
        flux_error_lambda = flux_err[mask_line]
        wave_lambda = wave[mask_line]

        delta_lambda = np.median(np.diff(wave_lambda))

        ew = np.sum((1- flux_lambda/flux_c) * delta_lambda)
        ew_error = 1/flux_c * np.sqrt(np.sum( flux_error_lambda**2 * delta_lambda**2 ))

        return ew, ew_error
    

    @staticmethod
    def _compute_gls(star:str, index:str, bandpasses:list, period_test:list, fap_treshold:float, df_clean:pd.DataFrame, ew_cols:list, plot_gls:bool, output:str, spec_lines_folder:str):
        """Computes GLS periodograms for the EW time series and identifies significant periods.
        
        Parameters
        ----------
        star : str
            ID of target star.
        index : str
            Identifier of the spectral line.
        bandpasses : list
            List of bandpasses corresponding to the EW columns.
        period_test : list
            List of tuples with period test values and their names for plots.
        fap_treshold : float
            FAP threshold for identifying significant periods.
        df_clean : pandas DataFrame
            Cleaned DataFrame containing the EW time series and their errors.
        ew_cols : list
            List of column names in df_clean corresponding to the EWs for each bandpass.
        plot_gls : bool
            Whether to plot the GLS periodograms or not.
        output : str
            Level of output to save, "standard" for only final results and "full" for all intermediate results and plots.
        spec_lines_folder : str
            Path to folder where to save outputs for this spectral line if output is "full".

        Returns
        -------
        gls_df : pandas DataFrame
            DataFrame containing the significant periods and their FAPs for each bandpass.
        all_gls_df : pandas DataFrame
            DataFrame containing the full GLS periodogram results for each bandpass (very memory consuming).
        """
        gls_list = []
        gls_results_lists = []

        for i, col in enumerate(tqdm.tqdm(ew_cols)):

            df = df_clean[["BJD", col, col+"_error"]]
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
                bp = bandpasses[i]
            else:
                bp = round(bandpasses[i],1)
                
            gls_dic = pd.DataFrame({"bandpass":[bp], 
                                    "period":[periods_fap_threshold_sorted], 
                                    "FAP":[FAPS_fap_threshold_sorted]})
            gls_list.append(gls_dic)

        gls_df = pd.concat(gls_list, axis=0, ignore_index=True)
        all_gls_df = pd.concat(gls_results_lists, axis=0, ignore_index=True)

        if plot_gls:
            gls_periodogram.GLS_plot(star, index, period_test, bandpasses, df_clean, all_gls_df, spec_lines_folder)
            plt.show()

        return gls_df, all_gls_df
    
    @staticmethod
    def _compute_correlation(df_ew:pd.DataFrame, bandpasses:list, ew_cols:list, df_input_raw:pd.DataFrame, cols_input:list, cols_err_input:list, sigma:float, bin_night:float):
        """Compute Spearman correlations between EW columns and input columns.

        Parameters
        ----------
        df_ew : pandas DataFrame
            DataFrame containing the EW time series with columns for each bandpass.
        bandpasses : list
            List of bandpasses corresponding to the EW columns.
        ew_cols : list
            List of column names in df_ew corresponding to the EWs for each bandpass.
        df_input_raw : pandas DataFrame
            DataFrame containing the input arrays for correlation before cleaning.
        cols_input : list
            List of column names in df_input_raw corresponding to the input arrays for correlation.
        cols_err_input : list
            List of column names in df_input_raw corresponding to the errors of the input arrays for correlation.
        sigma : float
            Number of standard deviations to use for sigma clipping of the EW time series. If 0, no sigma clipping is applied.
        bin_night : bool
            Whether to bin the EW time series by night (if True, binning is applied; otherwise no binning).
        
        Returns
        -------
        df_correlations : pandas DataFrame
            DataFrame containing the Spearman correlation coefficients and p-values between the EW time series and input arrays for each bandpass.
        """
        # clean input data frame
        df_input = time_series_clean(df_input_raw, list(zip(cols_input, cols_err_input)), bin_night, True if sigma else False, sigma=sigma).df

        df_correlations = pd.DataFrame({"bandpass":np.around(bandpasses,1)})

        for col in cols_input:

            col_corr_list = []
            col_pval_list = []

            for col1 in ew_cols:

                rho, pval = spearmanr(df_ew[col1], df_input[col], nan_policy = "omit") # Spearman correlation
                col_corr_list.append(np.around(rho,5))
                col_pval_list.append(pval)

            df_correlations = pd.concat([df_correlations, pd.DataFrame({f"rho_{col}":col_corr_list,f"pval_{col}":col_pval_list})],axis=1)

        return df_correlations
    

    def plot_gls_and_correlations(self, index:str, plot:str, folder_path:str, gls_plot_type:str="highest peak", gls_y_lims:list=[75,85]):
        """Plots the main peak of GLS periodograms and Spearman correlations in function of central bandpasses for a given index.

        Parameters
        ----------
        index : str
            Identifier of the spectral line.
        plot : str
            Type of plot to show, "gls" for GLS periods and FAPs, "correlation" for Spearman correlations and p-values, and 
            "both" for both plots.
        folder_path : str
            Path to folder where to load the GLS and correlation results for the given index. It should contain the files 
            "periods_{star}_{index}.csv" or "periods_{star}_{index}_full.csv" and "correlations_{star}_{index}.csv".
        gls_plot_type : str, optional
            Type of GLS plot to show if plot is "gls" or "both", "highest peak" for only the period with lowest FAP at each bandpass and "wavelet" for a wavelet-like plot with all periods and FAPs (default: "highest peak").
        gls_y_lims : list, optional
            Y-axis limits for the GLS period plot if gls_plot_type is "highest peak" (default: [75,85]). If gls_plot_type is "wavelet", the limits are set automatically to the min and max of the periods.
        """
        def _to_list(x):
            if isinstance(x, str):
                x = x.replace("np.float64(", "").replace(")", "")
                try:
                    x = ast.literal_eval(x)
                except Exception:
                    return []
            if isinstance(x, (list, tuple, np.ndarray)):
                return list(x)
            return [x]
        
        result = {}

        if plot in ["gls", "both"]:
            if gls_plot_type == "highest peak":
                result["gls"] = pd.read_csv(f"{folder_path}/{self.star}/{index}/periods_{self.star}_{index}.csv")
            elif gls_plot_type == "wavelet":
                result["gls"] = pd.read_csv(f"{folder_path}/{self.star}/{index}/periods_full_{self.star}_{index}.csv")
    
        if plot in ["correlation", "both"]:
            result["correlations"] = pd.read_csv(f"{folder_path}/{self.star}/{index}/correlations_{self.star}_{index}.csv")

        cmap = plt.get_cmap("plasma")

        fig, axes = plt.subplots(nrows=len(result.keys()), ncols=1, figsize=(6.5, 0.5+2.5*len(result.keys())), sharex=True)

        if plot in ["gls"]:
            ax_gls = axes
            ax_gls.set_title(f"{self.star} - {index}", loc="left", fontsize=14)
            ax_gls.set_xlabel(r"Central bandpass $\Delta \lambda$ [$\AA$]", fontsize=12)
        elif plot in ["correlation"]:
            ax_corr = axes
            ax_corr.set_title(f"{self.star} - {index}", loc="left", fontsize=14)
        elif plot in ["both"]:
            ax_gls = axes[0]
            ax_corr = axes[1]
            ax_gls.set_title(f"{self.star} - {index}", loc="left", fontsize=14)
            ax_corr.set_xlabel(r"Central bandpass $\Delta \lambda$ [$\AA$]", fontsize=12)

        log_FAP_min, log_FAP_max = 0, -23
        boundaries_FAP = np.arange(log_FAP_max, log_FAP_min + 1, 1)
        norm_FAP = mcolors.BoundaryNorm(boundaries_FAP, cmap.N, clip=True)

        log_pval_min, log_pval_max = 0, -23
        boundaries_corr = np.arange(log_pval_max, log_pval_min + 1, 1)
        norm_corr = mcolors.BoundaryNorm(boundaries_corr, cmap.N, clip=True)

        if plot in ["gls", "both"]:
            
            if gls_plot_type == "highest peak":
                periods = result["gls"]["period"].apply(lambda x: _to_list(x)[0] if len(_to_list(x)) > 0 else np.nan)  # get first value
                faps = result["gls"]["FAP"].apply(lambda x: _to_list(x)[0] if len(_to_list(x)) > 0 else np.nan)  
            else:
                periods = result["gls"]["period"].apply(_to_list)
                faps = result["gls"]["FAP"].apply(_to_list)

            bandpasses = np.array(range(1, len(result["gls"]["period"]) + 1)) / 10

            # highest (lowest FAP) GLS peak vs bandpass
            if gls_plot_type == "highest peak":
                im = ax_gls.scatter(bandpasses, periods, c=np.log10(faps), cmap=cmap, marker="o", s=100, norm=norm_FAP)

                ax_gls.set_ylabel(r"Period with lowest FAP [d]", fontsize=12)
                ax_gls.set_ylim(gls_y_lims)

            # wavelet like plot
            elif gls_plot_type == "wavelet":
                ax_gls.set_ylabel("Period [d]", fontsize=12)
                
                all_periods = [] # period grid (y-axis)
                for plist in periods:
                    for p in plist:
                        if np.isfinite(p):
                            all_periods.append(float(p))

                pmin, pmax = np.nanmin(all_periods), np.nanmax(all_periods)

                if gls_y_lims is None:
                    gls_y_lims = (pmin, pmax)

                period_grid = np.arange(gls_y_lims[0], gls_y_lims[1] + 1, 1)
                    
                # Z[y, x] = lowest log10(FAP) for each period bin at each bandpass
                Z = np.full((len(period_grid), len(bandpasses)), np.nan)

                for ix, (plist, flist) in enumerate(zip(periods, faps)):
                    for p, f in zip(plist, flist):
                        if not (np.isfinite(p) and np.isfinite(f) and f > 0):
                            continue
                        iy = np.argmin(np.abs(period_grid - float(p)))
                        v = np.log10(float(f))
                        if np.isnan(Z[iy, ix]) or v < Z[iy, ix]:
                            Z[iy, ix] = v

                im = ax_gls.imshow(np.ma.masked_invalid(Z), origin="lower", aspect="auto",
                                    extent=[bandpasses.min() - 0.05, bandpasses.max() + 0.05, period_grid.min(), period_grid.max()],
                                    cmap="plasma", interpolation="bilinear",
                                    vmin=-23, vmax=0)

            cbar1 = fig.colorbar(im, ax=ax_gls, label=r"log$_{10}$(FAP)", pad=0.02, fraction=0.045)
            cbar1.set_label(r"log$_{10}$(FAP)", fontsize=12)
            cbar1.ax.tick_params(labelsize=11)
            cbar1.ax.axhline(y=np.log10(0.001), color='black', linestyle="-", linewidth=1.5)

            ax_gls.tick_params(axis="both", direction="in", top=True, right=True, which="both", labelsize=11)

        if plot in ["correlation", "both"]:
            # Spearman correlation vs bandpass
            im1 = ax_corr.scatter(result["correlations"]["bandpass"], result["correlations"]["rho_fwhm"], 
                                c=np.log10(result["correlations"]["pval_fwhm"]), cmap=cmap, 
                                marker="o", s=100, norm=norm_corr)

            cbar2 = fig.colorbar(im1, ax=ax_corr, label=r"log$_{10}$(p-value)", pad=0.02, fraction=0.045)
            cbar2.set_label(r"log$_{10}$(p-value)", fontsize=12)
            cbar2.ax.tick_params(labelsize=11)
            cbar2.ax.axhline(y=np.log10(0.001), color='black', linestyle="-", linewidth=1.5)

            ax_corr.tick_params(axis="both", direction="in", top=True, right=True, which="both", labelsize=11)
            ax_corr.set_ylabel(r"$\rho$ with NIRPS CCF FWHM", fontsize=12)

        fig.subplots_adjust(hspace=0, left=0.1, right=0.9, top=0.95, bottom=0.1)
    
        if folder_path is not None:
            plt.savefig(f"{folder_path}/{self.star}/{index}/period_correlation_bandpass.pdf", dpi=400, bbox_inches="tight")
    
        plt.show()