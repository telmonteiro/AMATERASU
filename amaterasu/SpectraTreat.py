import matplotlib.pyplot as plt
import gc
import numpy as np

from specutils import Spectrum, manipulation 
import astropy.units as u
from scipy.signal import find_peaks, peak_widths


class SpectraTreat:
    """Given an array of 1D or 2D spectra, finds the spectral order of the line (in the case of 2D spectra), computes the master 
    spectrum and defines the automatic bandpass and interpolation windows.

    Parameters
    ----------
    spectra_observations : numpy array
        array of spectral data with format (N_spectra, N_axis, N_orders, N_pixels). N_axis = 3 (wavelength, flux and flux error) 
        and N_orders = 1 in case of 1D spectra.
    index : str
        identifier of index / spectral line.
    index_info : dict or str 
        Dictionary containing the line identifier ``ln_id``, the line center ``ln_ctr``, the maximum bandpass ``ln_win`` 
        and the interpolation window ``interp_win``. If it's a string, it interprets it as a path for an .csv file, fetches 
        it and converts into a dict.
    automatic_windows_mult : list 
        List with multipliers for automatic window calculation, first element for line window and second for interpolation 
        window (default: [5,6]).
    plot_line : str
        Whether to plot the line with the defined windows. Options are "simple" for a simple plot of the coadded line with the windows, 
        and "complete" for a more complete plot with the detected peaks and the defined windows (default: "simple").
    folder : str
        folder path to save the complete plot. If None, the plot will not be saved (default: None).
    star : str
        star name for the plot title. 
        
    Attributes
    -------
    results : dict 
        dictionary with keys "ln_ctr", "ln_win", "interp_win", and "spectra_obs".

    Methods
    -------
    _wavelength_2D_grid(spectra_obs)
        Given an array of 2D spectra, finds the common wavelength grid for all spectra.
    _coadd_spectra(spectra_obs, wave_grid=None)
        Coadd 1D or 2D spectra, using a common wavelength grid.
    plot_simple(index, wave_c, flux_c, ln_ctr, ln_win, percentile_cont)
        Simple plot coadded and smoothed spectral line.
    plot_complete(star, index, wave_coadd, flux_coadd, peaks_indices, wave_c, flux_c, ln_ctr, ln_fwhm, ln_ctr_new, ctr_flux, interp_win, ln_win, folder)
        Complete plot coadded and smoothed spectral line with detected peaks and defined windows.
    """
    def __init__(self, spectra_observations:np.ndarray, index:str, index_info:dict, percentile_cont:float=80, automatic_windows_mult:list=[5,6], plot_line:str="simple", folder:str=None, star:str=None):
        
        ln_ctr = index_info["ln_ctr"]

        if len(spectra_observations.shape) > 3:

            spectra_obs = []

            for n in range(spectra_observations.shape[0]): # N_observations

                orders = []
                min_dist = []

                for i in range(spectra_observations.shape[2]): # N_orders

                    wave = spectra_observations[n,0,i,:]

                    if ln_ctr > wave[0] and ln_ctr < wave[-1]:
                        orders.append(i)
                        dist = ((ln_ctr - wave[0], wave[-1] - ln_ctr)) # minimize distance to the line center from the edges of the order
                        min_dist.append(np.min(dist))

                spec_order = orders[np.argmax(min_dist)]
                spectra_obs.append(spectra_observations[n,:,spec_order,:])

            spectra_obs = np.array(spectra_obs, dtype=object) # N_obs, spectrum

        else:
            spectra_obs = np.array(spectra_observations, dtype=object)

        if "ln_win" in list(index_info.keys()) and "interp_win" in list(index_info.keys()): # if given manually
            ln_win = index_info["ln_win"]
            interp_win = index_info["interp_win"]

        else:
            # coadd, find_peaks and define windows
            if len(spectra_observations.shape) > 3: # 2D
                wave_grid = self._wavelength_2D_grid(spectra_obs)
                wave_coadd, flux_coadd = self._coadd_spectra(spectra_obs, wave_grid)
            else:
                wave_coadd, flux_coadd = self._coadd_spectra(spectra_obs)
                flux_coadd = flux_coadd[(wave_coadd >= ln_ctr - 25) & (wave_coadd <= ln_ctr + 25)] # window of 50 A around the line center
                wave_coadd = wave_coadd[(wave_coadd >= ln_ctr - 25) & (wave_coadd <= ln_ctr + 25)]

            N = 2 # smoothing window size for peak detection, in pixels. Minimum necessary to avoid trapping in noise
            flux_coadd = np.convolve(flux_coadd, np.ones(N)/N, mode='same')
            inverted_flux = -flux_coadd

            # find peaks in the inverted flux
            peaks_indices = find_peaks(inverted_flux, width=1)[0]
            # measure widths at half maximum for all peaks
            widths_indices = peak_widths(inverted_flux, peaks_indices, rel_height=0.5)[0]

            # extract wavelengths and FWHM
            lines_center = wave_coadd[peaks_indices]
            lines_fwhm = widths_indices * np.median(np.diff(wave_coadd)) # convert width from pixels to wavelength units

            # find the line closest to ln_ctr
            ln_fwhm = None
            ln_ctr_new = None
            ctr_flux = None
            for wavelength, fwhm, flux in zip(lines_center, lines_fwhm, flux_coadd[peaks_indices]):
                if np.isclose(wavelength, ln_ctr, atol=0.1): # tolerance of 0.1 A around the line center
                    ln_fwhm = np.around(fwhm, 5)
                    ln_ctr_new = np.around(wavelength, 5)
                    ctr_flux = np.around(flux, 5)
                    break

            if ln_fwhm is None:
                raise Exception("Could not find a line near to ln_ctr input.")
        
            ln_win = np.around(automatic_windows_mult[0] * ln_fwhm, 1)
            interp_win = np.around(automatic_windows_mult[1] * ln_win, 1)

            mask_interp = (wave_coadd >= ln_ctr - interp_win/2) & (wave_coadd <= ln_ctr + interp_win/2)
            flux_c = np.array(flux_coadd, dtype=float)[mask_interp] 
            flux_c = flux_c[~np.isnan(flux_c)] # remove NaNs
            wave_c = np.array(wave_coadd, dtype=float)[mask_interp]

            if plot_line == "simple":
                self.plot_simple(index, wave_c, flux_c, ln_ctr, ln_win, percentile_cont)
            elif plot_line == "complete":
                self.plot_complete(star, index, wave_coadd, flux_coadd, peaks_indices, wave_c, flux_c, ln_ctr, ln_fwhm, ln_ctr_new, ctr_flux, interp_win, ln_win, folder)
            
        results = {"ln_ctr": ln_ctr, "ln_win": ln_win, "interp_win": interp_win, "spectra_obs": spectra_obs}

        self.results = results
        

    def _wavelength_2D_grid(self, spectra_obs):
        """Given an array of 2D spectra, finds the common wavelength grid for all spectra.

        Parameters
        ----------
        spectra_obs : numpy array
            array of spectral data with format (N_spectra, N_axis, N_orders, N_pixels). 
        
        Returns
        -------
        wave_grid : numpy array
            array with the step size, minimum starting wavelength and maximum ending wavelength for the common wavelength grid.
        """
        wave_grid = np.zeros((3))

        step_values = []
        lambda_min_values = []
        lambda_max_values = []

        for i in range(spectra_obs.shape[0]):

            wave_order = spectra_obs[i,0,:]
            step_values.append(np.median(np.diff(wave_order)))
            lambda_min_values.append(np.min(wave_order))
            lambda_max_values.append(np.max(wave_order))

        wave_grid[0] = np.min(step_values)  # minimum step size
        wave_grid[1] = np.max(lambda_min_values)  # maximum starting wavelength
        wave_grid[2] = np.min(lambda_max_values)  # minimum ending wavelength

        return wave_grid
    

    def _coadd_spectra(self, spectra_obs:np.ndarray, wave_grid:np.ndarray=None):
        """Coadd 1D or 2D spectra, using a common wavelength grid.

        Parameters
        ----------
        spectra_obs : numpy array
            array of spectral data with format (N_spectra, N_axis, N_orders, N_pixels). 
        wave_grid : numpy array
            common wavelength grid.
        
        Returns
        -------
        wave_coadd : numpy array
            wavelength grid of the coadded spectrum.
        flux_coadd : numpy array
            flux values of the coadded spectrum.
        """
        if len(spectra_obs.shape) > 3:
            spectra_type = "2D" 
        else:
            spectra_type = "1D"

        if spectra_type == "1D":
            #using the first spectrum as the common reference grid
            wave_0 = spectra_obs[0,0,:]
            delta_lambda = np.median(np.diff(wave_0))
            new_disp_grid = np.arange(np.min(wave_0)+10, np.max(wave_0)-10, delta_lambda) * u.AA
            new_spec = Spectrum(spectral_axis=new_disp_grid, flux=np.zeros_like(new_disp_grid / u.AA) * u.dimensionless_unscaled)

        else:
            new_disp_grid = np.arange(wave_grid[1], wave_grid[2], wave_grid[0]) * u.AA
            new_spec = Spectrum(spectral_axis=new_disp_grid, flux=np.zeros_like(new_disp_grid / u.AA) * u.electron)

        for i in range(spectra_obs.shape[0]):

            wave = np.array(spectra_obs[i,0,:], dtype=float)
            flux = np.array(spectra_obs[i,1,:], dtype=float)

            flux[np.isnan(flux)] = 0
            flux[flux < 0] = 0  # replace negative flux with 0

            wave = wave * u.AA
            if spectra_type == "2D":
                flux = flux * u.electron
            else:
                flux = flux * u.dimensionless_unscaled

            input_spec = Spectrum(spectral_axis=wave, flux=flux)
            spline = manipulation.SplineInterpolatedResampler() # cubic spline interpolation

            new_spec_sp = spline(input_spec, new_disp_grid)
            new_spec += new_spec_sp

        wave_coadd = new_spec.wavelength.value
        flux_coadd = new_spec.flux.value

        return wave_coadd, flux_coadd


    def plot_simple(self, index:str, wave_c:np.ndarray, flux_c:np.ndarray, ln_ctr:float, ln_win:float, percentile_cont:float):
        """Simple plot coadded and smoothed spectral line.
        
        Parameters
        ----------
        index : str
            identifier of index / spectral line.
        wave_c : numpy array
            wavelength grid of the coadded spectrum.
        flux_c : numpy array
            flux values of the coadded spectrum.
        ln_ctr : float
            line center in wavelength units.
        ln_win : float
            line window in wavelength units.
        percentile_cont : float
            percentile to be plotted as a horizontal line in the plot.
        """
        plt.figure(figsize=(9, 3.5))

        plt.title(f"{index} coadded line")
        plt.xlabel(r"$\lambda$ [Å]")
        plt.ylabel("Flux")

        plt.plot(wave_c, flux_c, color="black", label="Coadded spectrum")
        plt.axvspan(ln_ctr - ln_win / 2, ln_ctr + ln_win / 2, alpha=0.1, color='yellow', ec = "black", lw = 2)

        plt.axhline(np.percentile(flux_c, percentile_cont), color='red', linestyle='--', linewidth=1, label=f'{percentile_cont}th percentile')

        plt.legend()
        plt.show()
        gc.collect()


    def plot_complete(self, star:str, index:str, wave_coadd:np.ndarray, flux_coadd:np.ndarray, peaks_indices:np.ndarray, wave_c:np.ndarray, flux_c:np.ndarray, ln_ctr:float, ln_fwhm:float, ln_ctr_new:float, ctr_flux:float, interp_win:float, ln_win:float, folder:str):
        """Complete plot coadded and smoothed spectral line with detected peaks and defined windows.
        
        Parameters
        ----------
        star : str
            star name.
        index : str
            identifier of index / spectral line.
        wave_coadd : numpy array
            wavelength grid of the coadded spectrum.
        flux_coadd : numpy array
            flux values of the coadded spectrum.
        peaks_indices : numpy array
            indices of the detected peaks in the coadded spectrum.
        wave_c : numpy array
            wavelength grid of the coadded spectrum in the interpolation window.
        flux_c : numpy array
            flux values of the coadded spectrum in the interpolation window.
        ln_ctr : float
            line center in wavelength units.
        ln_fwhm : float
            full width at half maximum of the detected line in wavelength units.
        ln_ctr_new : float
            line center of the detected line in wavelength units.
        ctr_flux : float
            flux value at the line center of the detected line.
        interp_win : float
            interpolation window in wavelength units.
        ln_win : float
            line window in wavelength units.
        folder : str
            folder path to save the plot. If None, the plot will not be saved.
        """
        fig, axes = plt.subplots(ncols=2, figsize=(13, 3.6), width_ratios=[1.7, 1])

        mask_plot = (wave_coadd >= ln_ctr - 35) & (wave_coadd <= ln_ctr + 35)
        mask_plot_peaks = (wave_coadd[peaks_indices] >= ln_ctr - 35) & (wave_coadd[peaks_indices] <= ln_ctr + 35)

        axes[0].plot(wave_coadd[mask_plot], flux_coadd[mask_plot], color="black")
        axes[0].plot(wave_coadd[peaks_indices][mask_plot_peaks], flux_coadd[peaks_indices][mask_plot_peaks], 
                     "x", color="red", label='Detected absorption spectral lines')
        
        axes[0].axvline(ln_ctr,ls="--",color="red")
        axes[0].annotate(f'FWHM: {ln_fwhm:.2f}\nCenter: {ln_ctr_new:.2f}', xy=(ln_ctr_new, ctr_flux), xytext=(11, 9), textcoords='offset points', fontsize=12)
        
        axes[0].axvline(ln_ctr_new-ln_win/2,lw=1,ls="--", color="blue")
        axes[0].axvline(ln_ctr_new+ln_win/2,lw=1,ls="--", color="blue")
        axes[0].axvline(ln_ctr_new-interp_win/2,lw=1,ls="--", color="black")
        axes[0].axvline(ln_ctr_new+interp_win/2,lw=1,ls="--", color="black")

        axes[0].set_xlabel(f"Wavelength $\lambda$ [$\AA$]", fontsize=12)
        axes[0].set_ylabel(r"Flux [e$^-$]", fontsize=12)
        axes[0].text(0.79, 1.03, f'{star} - {index} line', transform=axes[0].transAxes, ha='center', fontsize=13)
        axes[0].margins(x=0)
        axes[0].legend()

        axes[1].plot(wave_c, flux_c, color="black")
        axes[1].plot(ln_ctr_new, ctr_flux, "x", color="red", label='Target spectral line')

        axes[1].axvline(ln_ctr_new-ln_win/2,lw=1,ls="--")
        axes[1].axvline(ln_ctr_new+ln_win/2,lw=1,ls="--")

        axes[1].axhline(np.percentile(flux_c, 80), ls="--", color="red", label="Pseudo-continuum")
        
        axes[1].set_xlabel(f"Wavelength $\lambda$ [$\AA$]", fontsize=12)
        axes[1].set_ylabel(r"Flux [e$^-$]", fontsize=12)
        axes[1].text(0.61, 1.03, f'{star} - {index} line', transform=axes[1].transAxes, ha='center', fontsize=13)
        axes[1].legend(loc="lower right")
        axes[1].margins(x=0)
        
        axes[1].axvspan(ln_ctr_new-0.1/2, ln_ctr_new+0.1/2, color='orange', alpha=0.4)
        axes[1].axvspan(ln_ctr_new-ln_win/2, ln_ctr_new+ln_win/2, color='yellow', alpha=0.2)

        plt.tight_layout()

        if folder is not None:
            plt.savefig(f"{folder}amaterasu_auto_wins_{index}_{star}.pdf",dpi=400,bbox_inches="tight")
        
        plt.show()