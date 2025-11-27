import pandas as pd
import matplotlib.pyplot as plt
import gc
import numpy as np
from specutils import Spectrum, manipulation 
import astropy.units as u
from scipy.signal import find_peaks

class SpectraTreat:
    """Given an array of 1D or 2D spectra, finds the spectral order of the line (in the case of 2D spectra) and defines the bandpass and interpolation windows.
    Args:
        spectra_observations (numpy array): array of spectral data with format (N_spectra, N_axis, N_orders, N_pixels). N_axis = 3 (wavelength, flux and flux error) and N_orders = 1 in case of 1D spectra.
        indice (str): identifier of indice / spectral line.
        indice_info (dict or str): Dictionary containing the line identifier ``ln_id``, the line center ``ln_ctr``, the maximum bandpass ``ln_win`` and the interpolation window ``interp_win``. If it's a string, it interprets it as a path for an .csv file, fetches it and converts into a dict.
        automatic_windows (dict): if window definition is automatic, the multiplier of the FWHM for each window.
        plot_line (bool): plot coadded and smoothed spectra line or not.

    Attributes:
        resuls (tuple): line center, bandpass window, interpolation window, compressed array of spectral data 
    """
    def __init__(self, spectra_observations, indice, indice_info=None, automatic_windows={"ln_win_mult":5,"interp_win_mult":5},plot_line=False):
        
        if type(indice_info) == str:
            ind_table = pd.read_csv(indice_info)
            ind_table = ind_table.to_dict(orient='records')
            indice_info = {d['ln_id']: {k:v for k,v in d.items() if k != 'ln_id'} for d in ind_table}

        indice_info = indice_info[indice]
        ln_ctr = indice_info["ln_ctr"]

        if len(spectra_observations.shape) > 3:

            spectra_obs = []

            for n in range(spectra_observations.shape[0]): #N_observations

                orders = []
                min_dist = []

                for i in range(spectra_observations.shape[2]): #N_orders

                    wave = spectra_observations[n,0,i,:]

                    if ln_ctr > wave[0] and ln_ctr < wave[-1]:

                        orders.append(i)
                        dist = ((ln_ctr - wave[0], wave[-1] - ln_ctr))
                        min_dist.append(np.min(dist))

                spec_order = orders[np.argmax(min_dist)]
                spectra_obs.append(spectra_observations[n,:,spec_order,:])

            spectra_obs = np.array(spectra_obs, dtype=object) #N_obs, spectrum

        else:
            spectra_obs = np.array(spectra_observations, dtype=object)

        if "ln_win" in list(indice_info.keys()) and "interp_win" in list(indice_info.keys()):
            ln_win, interp_win = indice_info["ln_win"], indice_info["interp_win"]

        else:
            #coadd, find_peaks, define windows
            if len(spectra_observations.shape) > 3:
                wave_grid = self._wavelength_2D_grid(spectra_obs)
                wave_coadd, flux_coadd = self._coadd_spectra_s2d(spectra_obs, wave_grid)
            else:
                wave_coadd, flux_coadd = self._coadd_spectra_s1d(spectra_obs)
                flux_coadd = flux_coadd[(wave_coadd >= ln_ctr - 25) & (wave_coadd <= ln_ctr + 25)]
                wave_coadd = wave_coadd[(wave_coadd >= ln_ctr - 25) & (wave_coadd <= ln_ctr + 25)]

            N = 2
            flux_coadd = np.convolve(flux_coadd, np.ones(N)/N, mode='same')
            inverted_flux = -flux_coadd

            #find peaks in the inverted flux
            peaks_indices, properties = find_peaks(inverted_flux, rel_height=0.5, width=1)

            #extract wavelengths and widths of the detected lines
            lines_center = wave_coadd[peaks_indices]
            lines_fwhm = properties['widths'] * np.median(np.diff(wave_coadd))
 
            for i, (wavelength, fwhm) in enumerate(zip(lines_center, lines_fwhm)):
                if np.isclose(wavelength,ln_ctr,atol=0.1):
                    ln_fwhm = np.around(fwhm,5)

            try:
                ln_fwhm
            except NameError:
                raise Exception("Could not find a line near to ln_ctr input.")

            ln_win = np.around(automatic_windows["ln_win_mult"]*ln_fwhm,1)
            interp_win = np.around(automatic_windows["interp_win_mult"]*ln_win,1)

            if plot_line:

                plt.figure(figsize=(9, 3.5))
                plt.title(f"{indice} coadded line")
                plt.xlabel(r"$\lambda$ [Å]")
                plt.ylabel("Flux")
                mask_interp = (wave_coadd >= ln_ctr - interp_win/2) & (wave_coadd <= ln_ctr + interp_win/2)
                plt.plot(wave_coadd[mask_interp], flux_coadd[mask_interp])
                plt.axvspan(ln_ctr - ln_win / 2, ln_ctr + ln_win / 2, alpha=0.1, color='yellow', ec = "black", lw = 2)
                flux_c = np.percentile(np.array(flux_coadd[mask_interp], dtype=float)[(np.isnan(np.array(flux_coadd[mask_interp], dtype=float))==False)], 85)
                plt.axhline(flux_c)
                plt.show()
                gc.collect()
            
        SpectraTreat.results = ln_ctr, ln_win, interp_win, spectra_obs


    def _wavelength_2D_grid(self, spectra_obs):
        # Initialize the wavelength grid
        wave_grid = np.zeros((3))

        step_values = []
        lambda_min_values = []
        lambda_max_values = []

        for i in range(spectra_obs.shape[0]):

            wave_order = spectra_obs[i,0,:]
            step_values.append(np.median(np.diff(wave_order)))
            lambda_min_values.append(np.min(wave_order))
            lambda_max_values.append(np.max(wave_order))

        wave_grid[0] = np.min(step_values)  #minimum step size
        wave_grid[1] = np.max(lambda_min_values)  #maximum starting wavelength
        wave_grid[2] = np.min(lambda_max_values)  #minimum ending wavelength

        return wave_grid
    

    def _coadd_spectra_s2d(self, spectra_obs, wave_grid):

        new_disp_grid = np.arange(wave_grid[1], wave_grid[2], wave_grid[0]) * u.AA
        new_spec = Spectrum(spectral_axis=new_disp_grid, flux=np.zeros_like(new_disp_grid / u.AA) * u.electron)

        for i in range(spectra_obs.shape[0]):

            wave = np.array(spectra_obs[i,0,:], dtype=float)
            flux = np.array(spectra_obs[i,1,:], dtype=float)

            flux[np.isnan(flux)] = 0
            flux[flux <= 0] = 1  # Replace non-positive flux with 1

            wave = wave * u.AA
            flux = flux * u.electron

            input_spec = Spectrum(spectral_axis=wave, flux=flux)
            spline = manipulation.SplineInterpolatedResampler()

            new_spec_sp = spline(input_spec, new_disp_grid)
            new_spec += new_spec_sp

        return new_spec.wavelength.value, new_spec.flux.value
    

    def _coadd_spectra_s1d(self, spectra_obs):
        #using the first spectrum as the common reference grid
        wave_0 = spectra_obs[0,0,:]
        delta_lambda = np.median(np.diff(wave_0))

        new_disp_grid = np.arange(np.min(wave_0)+10, np.max(wave_0)-10, delta_lambda) * u.AA
        new_spec = Spectrum(spectral_axis=new_disp_grid, flux=np.zeros_like(new_disp_grid / u.AA) * u.dimensionless_unscaled)

        for i in range(spectra_obs.shape[0]):

            wave = np.array(spectra_obs[i,0,:], dtype=float)
            flux = np.array(spectra_obs[i,1,:], dtype=float)
            flux[np.isnan(flux)] = 0
            flux[flux <= 0] = 1  #replace non-positive flux with 1

            wave = wave * u.AA
            flux = flux * u.dimensionless_unscaled

            input_spec = Spectrum(spectral_axis=wave, flux=flux)
            spline = manipulation.SplineInterpolatedResampler()

            new_spec_sp = spline(input_spec, new_disp_grid)
            new_spec += new_spec_sp

        return new_spec.wavelength.value, new_spec.flux.value