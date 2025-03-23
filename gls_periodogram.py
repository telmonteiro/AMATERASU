import numpy as np
import gc
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

class gls_periodogram:
    """Generalised Lomb-Scargle Periodogram using the `astropy` implementation.
    Args:
        star (str): Star ID.
        p_rot_lit (list): Literature rotation period and its error for comparison.
        ind (str): Activity indicator name.
        x (array-like): Time coordinate.
        y (array-like): Y-coordinate (signal).
        y_err (array-like, optional): Y error if provided.
        pmin (float): Minimum period to compute periodogram.
        pmax (float): Maximum period to compute periodogram.
        steps (int): Number of steps in the frequency grid.
        save (bool): Whether to save the plots or not.
        folder_path (str, optional): Path to save the plots.

    Returns:
        dict: Dictionary containing periodogram results.
    """
    def __init__(self, star, period_test, ind, x, y, y_err, pmin=1.5, pmax=1e3, steps=5e3, folder_path=None):

        x = np.asarray(x)
        y = np.asarray(y)
        t = max(x) - min(x)
        n = len(x)

        results = {}

        #Compute the periodogram and window function.
        #Nyquist frequencies computation
        k1 = pmax/t;     fmin = 1./(k1 * t)
        k2 = pmin * n/t; fmax = n/(k2 * t)
        freq = np.linspace(fmin, fmax, int(steps))
        period = 1./freq

        #get power from GLS
        gls = LombScargle(x, y, y_err)
        power = gls.power(freq)

        #detect peaks
        peaks, _ = find_peaks(power)
        sorted_peak_indices = np.argsort(power[peaks])[::-1]  #sort in descending order of power
        sorted_peaks = peaks[sorted_peak_indices]
        peaks_power = power[sorted_peaks]
        peaks_period = period[sorted_peaks]

        #False alarm probabilities (FAP)
        fap_max_power = gls.false_alarm_probability(np.nanmax(power))
        faps = gls.false_alarm_probability(power)
        fap_levels = np.array([0.01, 0.001])
        fap1, fap01 = gls.false_alarm_level(fap_levels)

        results.update({'freq': freq,'period': period,'power': power,'fap_maxp': fap_max_power,'fap_01': fap01,'fap_1': fap1,'FAPS': faps})

        #Window function
        power_win = LombScargle(x, np.ones_like(y), fit_mean=False, center_data=False).power(freq)
        results.update({'power_win': power_win, 'period_best_WF': period[np.argmax(power_win)]})
        peaks_WF, _ = find_peaks(power_win)
        sorted_peak_indices = np.argsort(power_win[peaks_WF])[::-1]
        sorted_peaks = peaks_WF[sorted_peak_indices]
        peaks_power_win = power_win[sorted_peaks]
        peaks_period_win = period[sorted_peaks]

        #Get significant peaks
        dict_peaks = {"peaks_period":peaks_period,"peaks_power":peaks_power}
        dict_peaks_WF = {"peaks_period_win":peaks_period_win,"peaks_power_win":peaks_power_win}

        #Top 5 time gaps
        gaps = np.diff(x[np.argsort(x)])
        gaps = gaps[np.argsort(gaps)][-5:]
        sel_peaks_dict = self._get_sign_gls_peaks(gls, dict_peaks, dict_peaks_WF, fap1, atol_frac=0.1)
        
        results.update({'gaps': gaps,'sel_peaks_period': sel_peaks_dict["sel_peaks_period"],'sel_peaks_power': sel_peaks_dict["sel_peaks_power"], "sel_peaks_FAP": sel_peaks_dict["sel_peaks_FAP"]})

        if len(sel_peaks_dict["sel_peaks_period"]) > 0:

            period_best = sel_peaks_dict["sel_peaks_period"][0]
            power_best = sel_peaks_dict["sel_peaks_power"][0]
            fap_best = sel_peaks_dict["sel_peaks_FAP"][0]

        else: #there is no significant peaks in the periodogram

            period_best = 0
            power_best = 0
            fap_best = 0

        results.update({'period_best': round(period_best,3), 'power_best': power_best, 'fap_best': fap_best})

        if folder_path:
            self.plot_periodogram(ind, star, period_test, results, results["period"], results["power"], results["sel_peaks_period"], results['fap_1'], results['fap_01'], folder_path)
            self.plot_window_function(ind, star, results, results["period"], results["power_win"], results['fap_1'], results['fap_01'], folder_path)

        gls_periodogram.results = results


    def _get_sign_gls_peaks(self, gls, dict_peaks, dict_peaks_WF, fap1, atol_frac=0.05):
        """Get GLS significant peaks and excludes peaks close to window function peaks in BJD.
        Args:
            df_peaks (dict): star ID.
            df_peaks_WF (dict): columns to compute the statistics on.
            fap1 (float): 1% false alarm probability.
            atol_frac (float):

        Returns:
            sel_peaks_dict (dict): period and power of significant peaks.
        """
        sign_peaks = [per for per, power in zip(dict_peaks['peaks_period'], dict_peaks['peaks_power'])]
        sign_peaks_power = [power for per, power in zip(dict_peaks['peaks_period'], dict_peaks['peaks_power'])]
        sign_peaks_fap = gls.false_alarm_probability(sign_peaks_power)
        sign_peaks_win = [per for per, power in zip(dict_peaks_WF['peaks_period_win'], dict_peaks_WF['peaks_power_win']) if power > fap1]

        # exclude peaks close to win peaks
        exc_peaks = []; exc_peaks_power = []
        for ind,peak in enumerate(sign_peaks):

            atol = peak * atol_frac

            for peak_win in sign_peaks_win:

                if np.isclose(peak, peak_win, atol=atol):
                    exc_peaks.append(peak)
                    exc_peaks_power.append(ind)   

        sel_peaks_period = [peak for peak in sign_peaks if peak not in exc_peaks]  
        sel_peaks_power = [pow for ind, pow in enumerate(sign_peaks_power) if ind not in exc_peaks_power]
        sel_peaks_fap = [fap for ind, fap in enumerate(sign_peaks_fap) if ind not in exc_peaks_power]
        sel_peaks_dict = {"sel_peaks_period":sel_peaks_period,"sel_peaks_power":sel_peaks_power,"sel_peaks_FAP":sel_peaks_fap}

        return sel_peaks_dict
    

    def plot_periodogram(self, ind, star, period_test, results, period, power, sel_peaks, fap1, fap01, folder_path):
        """Generate and save the periodogram plot."""
        fig, axes = plt.subplots(figsize=(10, 4))

        axes.text(0.13, 0.89, f"{star} GLS {ind}", fontsize=12, transform=plt.gcf().transFigure)
        axes.set_xlabel("Period [d]")
        axes.set_ylabel("Normalized Power")

        axes.semilogx(period, power, 'k-')

        plevels = [fap1,fap01]
        fap_levels = np.array([0.01, 0.001])

        for i in range(len(fap_levels)):
            axes.plot([min(period), max(period)], [plevels[i]]*2, '--', label="FAP = %4.1f%%" % (fap_levels[i]*100))

        if type(period_test[0]) != list: 
            period_test = [period_test]

        for p_rot_val in period_test:
            p_rot, p_rot_err = p_rot_val
            axes.axvspan(p_rot - p_rot_err, p_rot + p_rot_err, alpha=0.6, color="orange")

        for peak in sel_peaks:
            axes.axvline(peak, ls='--', lw=0.8, color='red')

        vline_legend = Line2D([0], [0], color='red', linestyle='--', label='Significant peaks',markersize=0.5)
        handles, _ = axes.get_legend_handles_labels()
        handles.append(vline_legend)
        axes.legend(handles=handles, loc="best")
        axes.text(0.65, 0.89, f"P = {np.around(results['period_best'],1)} d, FAP = {results['fap_best']*100:.2e}%", fontsize=12, transform=plt.gcf().transFigure)

        plt.savefig(folder_path+f"periodograms/{star}_{ind}_GLS.pdf", bbox_inches="tight")
        plt.close('all')
        gc.collect()


    def plot_window_function(self, ind, star, results, period, power_win, fap1, fap01, folder_path):
        """Generate and save the window function periodogram plot."""
        fig, axes = plt.subplots(figsize=(10, 4))
        axes.set_xlabel("Period [days]")
        axes.set_ylabel("Power")

        axes.semilogx(period, power_win, 'b-',label="WF")
        axes.semilogx(period, results["power"], 'r-',lw=0.7,label="data")

        axes.text(0.13, 0.89, f"{star} WF GLS {ind}", fontsize=12, transform=plt.gcf().transFigure)

        plevels = [fap1,fap01]
        fap_levels = np.array([0.01, 0.001])

        for i in range(len(fap_levels)):
            axes.plot([min(period), max(period)], [plevels[i]]*2, '--', label="FAP = %4.1f%%" % (fap_levels[i]*100))

        for gap in results["gaps"]:
            axes.axvline(gap, ls='--', lw=0.7, color='green')

        axes.legend(loc="best")

        plt.savefig(folder_path+f"periodograms/{star}_{ind}_GLS_WF.pdf", bbox_inches="tight")
        plt.close('all')
        gc.collect()

    @staticmethod
    def GLS_plot(star, ind, p_rot_lit_list, bandpasses, df_clean, all_gls_df, folder):

        nrows = len(bandpasses)
        fig, axes = plt.subplots(nrows=nrows,ncols=2,figsize=(14, 0.5+nrows*2), sharex="col")

        for i,bp in enumerate(bandpasses):

            col = f"{ind}{int(round(bp*10,0)):02d}_EW"
            results = all_gls_df.iloc[i].to_dict()

            axes[i,0].errorbar(df_clean["BJD"] - 2450000, df_clean[col], df_clean[col+"_error"], fmt="k.")
            axes[i,0].set_ylabel(rf"EW - {int(round(bp*10,0)):02d} [$\AA$]", fontsize=13)

            if i == len(bandpasses) - 1:
                axes[i,0].set_xlabel("BJD $-$ 2450000 [d]", fontsize=13)

            axes[i,0].tick_params(axis="both", direction="in", top=True, right=True, which='both')

            axes[i,1].semilogx(results["period"], results["power"], "k-")
            axes[i,1].plot([min(results["period"]), max(results["period"])], [results["fap_01"]] * 2,"--",color="black",lw=0.7)
            axes[i,1].plot([min(results["period"]), max(results["period"])], [results["fap_1"]] * 2,"--",color="black",lw=0.7)

            axes[i,1].set_ylabel("Norm. Power", fontsize=13)
            axes[i,1].set_xlabel("Period [d]", fontsize=13)
            axes[i,1].tick_params(axis="both", direction="in", top=True, right=True, which='both')

            if type(p_rot_lit_list[0]) != list: 
                p_rot_lit_list = [p_rot_lit_list]

            for p_rot_val in p_rot_lit_list:
                p_rot, p_rot_err = p_rot_val
                axes[i,1].axvspan(p_rot - p_rot_err, p_rot + p_rot_err, alpha=0.6, color="orange")
                
            axes[i,1].text(0.05, 0.9,f"P = {np.around(results['period_best'], 1)} d \nFAP = {results['fap_best']*100:.2e}%",fontsize=12,transform=axes[i,1].transAxes,verticalalignment="top")

        fig.subplots_adjust(hspace=0.0)
        fig.text(0.13, 0.881, f"{star} - {ind}", fontsize=17)
        plt.savefig(folder+f"{star}_{ind}_GLS.pdf",dpi=2000, bbox_inches="tight")