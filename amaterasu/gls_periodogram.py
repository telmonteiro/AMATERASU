import numpy as np
import gc
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
import pandas as pd

from astropy.timeseries import LombScargle
from scipy.signal import find_peaks


class gls_periodogram:
    """Generalised Lomb-Scargle Periodogram using the `astropy` implementation.

    Parameters
    ----------
    star : str
        Star ID.
    period_test : list
        Literature rotation period and its error for comparison.
    ind : str
        Activity indicator name.
    x : array-like
        Time coordinate.
    y : array-like
        Y-coordinate (signal).
    y_err : array-like, optional
        Y error if provided.
    pmin : float, optional
        Minimum period to compute periodogram. Default is 1.5.
    pmax : float, optional
        Maximum period to compute periodogram. Default is 1e3.
    steps : int, optional
        Number of steps in the frequency grid. Default is 1e5.
    folder_path : str, optional
        Path to save the plots. Default is None.

    Attributes
    ----------
    results : dict
        Dictionary containing periodogram results.

    Methods
    -------
    _get_sign_gls_peaks(gls, dict_peaks, dict_peaks_WF, atol_frac=0.05)
        Get GLS significant peaks and exclude peaks close to window function peaks in BJD.
    plot_periodogram(index, star, period_test, results, folder_path)
        Generate and save the periodogram plot.
    plot_window_function(index, star, results, folder_path)
        Generate and save the window function periodogram plot.
    GLS_plot(star, index, period_test, bandpasses, df_clean, all_gls_df, folder)
        Generate and save the GLS periodogram plots for all bandpasses of a given activity indicator.
    """
    def __init__(self, star:str, period_test:list, index:str, x:np.ndarray, y:np.ndarray, y_err:np.ndarray=None, pmin:float=1.5, pmax:float=1e3, steps:float=1e4, folder_path:str=None):

        x = np.asarray(x)
        y = np.asarray(y)
        t = max(x) - min(x)
        n = len(x)

        results = {}

        # compute the periodogram and window function.
        # Nyquist frequencies computation
        k1 = pmax/t;     fmin = 1./(k1 * t)
        k2 = pmin * n/t; fmax = n/(k2 * t)
        freq = np.linspace(fmin, fmax, int(steps))
        period = 1./freq

        # get power from GLS
        gls = LombScargle(x, y, y_err)
        power = gls.power(freq)

        # detect peaks
        peaks, _ = find_peaks(power)
        sorted_peak_indices = np.argsort(power[peaks])[::-1]  # sort in descending order of power
        sorted_peaks = peaks[sorted_peak_indices]
        peaks_power = power[sorted_peaks]
        peaks_period = period[sorted_peaks]

        # False alarm probabilities (FAP)
        fap_max_power = gls.false_alarm_probability(np.nanmax(power))
        faps = gls.false_alarm_probability(power)
        fap_levels = np.array([0.01, 0.001]) # 1% and 0.1% FAP levels
        fap1_power, fap01_power = gls.false_alarm_level(fap_levels)

        results.update({'freq': list(freq), 'period': list(period), 'power': list(power), 'fap_maxp': fap_max_power,
                        'fap01_power': fap01_power, 'fap1_power': fap1_power, 'FAP': list(faps)})

        # window function
        power_win = LombScargle(x, np.ones_like(y), fit_mean=False, center_data=False).power(freq)

        results.update({'power_win':list(power_win), 'period_best_WF':period[np.argmax(power_win)]})
        
        peaks_WF, _ = find_peaks(power_win)
        sorted_peak_indices = np.argsort(power_win[peaks_WF])[::-1]
        sorted_peaks = peaks_WF[sorted_peak_indices]
        peaks_power_win = power_win[sorted_peaks]
        peaks_period_win = period[sorted_peaks]

        # get significant peaks
        dict_peaks = {"peaks_period":peaks_period,"peaks_power":peaks_power}
        dict_peaks_WF = {"peaks_period_win":peaks_period_win,"peaks_power_win":peaks_power_win}

        # top 5 time gaps for reference
        gaps = np.diff(x[np.argsort(x)])
        gaps = gaps[np.argsort(gaps)][-5:]

        sel_peaks_dict = self._get_sign_gls_peaks(gls, dict_peaks, dict_peaks_WF, atol_frac=0.05)
        
        results.update({'gaps':list(gaps), 
                        'sel_peaks_period':list(sel_peaks_dict["sel_peaks_period"]), 
                        'sel_peaks_power':list(sel_peaks_dict["sel_peaks_power"]), 
                        "sel_peaks_FAP": list(sel_peaks_dict["sel_peaks_FAP"])})

        if len(sel_peaks_dict["sel_peaks_period"]) > 0:

            period_best = round(sel_peaks_dict["sel_peaks_period"][0], 3)
            power_best = sel_peaks_dict["sel_peaks_power"][0]
            fap_best = sel_peaks_dict["sel_peaks_FAP"][0]

        else: #there is no significant peaks in the periodogram

            period_best = np.nan
            power_best = np.nan
            fap_best = np.nan

        results.update({'period_best':period_best, 'power_best':power_best, 'fap_best':fap_best})

        if folder_path: # save plots
            self.plot_periodogram(index, star, period_test, results, folder_path)
            self.plot_window_function(index, star, results, folder_path)

        self.results = results


    def _get_sign_gls_peaks(self, gls, dict_peaks:dict, dict_peaks_WF:dict, atol_frac:float=0.05):
        """Get GLS significant peaks and excludes peaks close to window function peaks in BJD.

        Parameters
        ----------
        gls : LombScargle object
            Fitted LombScargle model.
        dict_peaks : dict
            Dictionary containing 'peaks_period' and 'peaks_power' from the GLS periodogram.
        dict_peaks_WF : dict
            Dictionary containing 'peaks_period_win' and 'peaks_power_win' from the window function periodogram.
        atol_frac : float
            Fraction of the peak period to use as an absolute tolerance when excluding peaks close to window function peaks. 
            Default is 0.05 (5%).
        
        Returns
        -------
        sel_peaks_dict : dict
            Dictionary containing 'sel_peaks_period', 'sel_peaks_power', and 'sel_peaks_FAP' for the selected significant peaks.
        """
        sign_peaks_fap = gls.false_alarm_probability(dict_peaks['peaks_power'])
        sign_peaks_win = [per for per, fap in zip(dict_peaks_WF['peaks_period_win'], sign_peaks_fap) if fap < 0.001] # peaks in the window function with FAP < 0.1%

        # exclude peaks close to win peaks
        exc_peaks = []
        exc_peaks_power = []

        for ind,peak in enumerate(dict_peaks['peaks_period']):

            atol = peak * atol_frac

            for peak_win in sign_peaks_win:

                if np.isclose(peak, peak_win, atol=atol):
                    exc_peaks.append(peak)
                    exc_peaks_power.append(ind)   

        sel_peaks_period = [peak for peak in dict_peaks['peaks_period'] if peak not in exc_peaks]  
        sel_peaks_power = [pow for ind, pow in enumerate(dict_peaks['peaks_power']) if ind not in exc_peaks_power]
        sel_peaks_fap = [fap for ind, fap in enumerate(sign_peaks_fap) if ind not in exc_peaks_power]
        
        sel_peaks_dict = {"sel_peaks_period":sel_peaks_period, "sel_peaks_power":sel_peaks_power, "sel_peaks_FAP":sel_peaks_fap}

        return sel_peaks_dict
    

    def plot_periodogram(self, index:str, star:str, period_test:list, results:dict, folder_path:str=None):
        """Generate and save the periodogram plot.

        Parameters
        ----------
        index : str
            Activity indicator name.
        star : str
            Star ID.
        period_test : list
            Test rotation period and its error for comparison.
        results : dict
            Dictionary containing periodogram results.
        folder_path : str
            Path to save the plot.
        """
        period = results["period"]
        power = results["power"]
        sel_peaks = results["sel_peaks_period"]
        sel_peaks_fap = results["sel_peaks_FAP"]
        fap1 = results['fap1_power']
        fap01 = results['fap01_power']
        
        fig, axes = plt.subplots(figsize=(10, 4))

        axes.text(0.13, 0.89, f"{star} GLS {index}", fontsize=12, transform=plt.gcf().transFigure)
        axes.set_xlabel("Period [d]")
        axes.set_ylabel("Normalized Power")

        axes.semilogx(period, power, 'k-')

        plevels = [fap1,fap01]
        fap_levels = np.array([0.01, 0.001]) # 1% and 0.1% FAP levels

        for i in range(len(fap_levels)):
            axes.plot([min(period), max(period)], [plevels[i]]*2, '--', label="FAP = %4.1f%%" % (fap_levels[i]*100))

        if type(period_test[0]) != list: 
            period_test = [period_test]

        for p_rot_val in period_test: # plot tested periods with their errors as shaded regions
            p_rot, p_rot_err = p_rot_val
            axes.axvspan(p_rot - p_rot_err, p_rot + p_rot_err, alpha=0.6, color="orange")

        for i,peak in enumerate(sel_peaks):
            if sel_peaks_fap[i] < 0.001: # only plot peaks with FAP < 0.1%
                axes.axvline(peak, ls='--', lw=0.8, color='red')

        vline_legend = Line2D([0], [0], color='red', linestyle='--', label='Significant peaks', markersize=0.5)
        handles, _ = axes.get_legend_handles_labels()
        handles.append(vline_legend)
        axes.legend(handles=handles, loc="best")

        axes.text(0.65, 0.89, f"P = {np.around(results['period_best'],1)} d, FAP = {results['fap_best']*100:.2e}%", 
                  fontsize=12, transform=plt.gcf().transFigure)

        plt.savefig(folder_path+f"periodograms/{star}_{index}_GLS.pdf", bbox_inches="tight")
        plt.close('all')
        gc.collect()


    def plot_window_function(self, index:str, star:str, results:dict, folder_path:str=None):
        """Generate and save the window function periodogram plot.

        Parameters
        ----------
        index : str
            Activity indicator name.
        star : str
            Star ID.
        results : dict
            Dictionary containing periodogram results.
        folder_path : str
            Path to save the plot.
        """
        period = results["period"]
        power_win = results["power_win"]
        fap1 = results['fap1_power']
        fap01 = results['fap01_power']

        fig, axes = plt.subplots(figsize=(10, 4))

        axes.set_xlabel("Period [days]")
        axes.set_ylabel("Power")

        axes.semilogx(period, power_win, 'b-',label="WF")
        axes.semilogx(period, results["power"], 'r-',lw=0.7,label="data")

        axes.text(0.13, 0.89, f"{star} WF GLS {index}", fontsize=12, transform=plt.gcf().transFigure)

        plevels = [fap1,fap01]
        fap_levels = np.array([0.01, 0.001])

        for i in range(len(fap_levels)):
            axes.plot([min(period), max(period)], [plevels[i]]*2, '--', label="FAP = %4.1f%%" % (fap_levels[i]*100))

        for gap in results["gaps"]:
            axes.axvline(gap, ls='--', lw=0.7, color='green')

        axes.legend(loc="best")

        plt.savefig(folder_path+f"periodograms/{star}_{index}_GLS_WF.pdf", bbox_inches="tight")
        plt.close('all')
        gc.collect()


    @staticmethod
    def GLS_plot(star:str, index:str, period_test:list, bandpasses:list, df_clean:pd.DataFrame, all_gls_df:pd.DataFrame, folder:str):
        """Generate and save the GLS periodogram plots for all bandpasses of a given activity indicator.
        
        Parameters
        ----------
        star : str
            Star ID.
        index : str
            Activity indicator name.
        period_test : list
            Test rotation period and its error for comparison.
        bandpasses : list
            central bandpasses for the activity indicator.
        df_clean : pandas DataFrame
            Cleaned DataFrame containing the time series data for the activity indicator.
        all_gls_df : pandas DataFrame
            DataFrame containing the GLS periodogram results for all bandpasses of the activity indicator.
        folder : str
            Path to save the plot.
        """
        nrows = len(bandpasses)
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(14, 0.5+nrows*2), sharex="col")

        for i, bp in enumerate(bandpasses):

            if len(bandpasses) == 1:
                ax = axes
            else:
                ax = axes[i]

            if type(bp) == str:
                col = bp
                ax[0].set_ylabel(col, fontsize=13)
            else:
                col = f"{index}{int(round(bp*10,0)):02d}_EW"
                ax[0].set_ylabel(rf"EW - {int(round(bp*10,0)):02d} [$\AA$]", fontsize=13)

            results = all_gls_df.iloc[i].to_dict()

            ax[0].errorbar(df_clean["BJD"] - 2450000, df_clean[col], df_clean[col+"_error"], fmt="k.")
            
            if i == len(bandpasses) - 1:
                ax[0].set_xlabel("BJD $-$ 2450000 [d]", fontsize=13)

            ax[0].tick_params(axis="both", direction="in", top=True, right=True, which='both')

            ax[1].semilogx(results["period"], results["power"], "k-")
            ax[1].plot([min(results["period"]), max(results["period"])], [results["fap01_power"]] * 2,"--",color="black",lw=0.7)
            ax[1].plot([min(results["period"]), max(results["period"])], [results["fap1_power"]] * 2,"--",color="black",lw=0.7)

            ax[1].set_ylabel("Norm. Power", fontsize=13)
            ax[1].set_xlabel("Period [d]", fontsize=13)
            ax[1].tick_params(axis="both", direction="in", top=True, right=True, which='both')

            if type(period_test[0]) != list:
                period_test = [period_test]

            for p_rot_val in period_test:  # plot tested periods with their errors as shaded regions
                p_rot, p_rot_err = p_rot_val
                ax[1].axvspan(p_rot - p_rot_err, p_rot + p_rot_err, alpha=0.6, color="orange")

            ax[1].text(0.05, 0.9, f"P = {np.around(results['period_best'], 1)} d \nFAP = {results['fap_best']*100:.2e}%", fontsize=12, transform=ax[1].transAxes, verticalalignment="top")

        fig.subplots_adjust(hspace=0.0)
        fig.text(0.13, 0.890-0.00016*nrows, f"{star} - {index}", fontsize=17)

        if folder is not None:
            plt.savefig(folder+f"{star}_{index}_GLS.pdf", dpi=300, bbox_inches="tight")