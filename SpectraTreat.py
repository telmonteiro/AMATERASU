import pandas as pd, numpy as np # type: ignore
from specutils.manipulation import SplineInterpolatedResampler #type: ignore
from specutils import Spectrum1D #type: ignore
import astropy.units as u #type: ignore
from scipy.signal import find_peaks

class SpectraTreat:
    """Given an array of 1D or 2D spectra, finds the spectral order of the line (in the case of 2D spectra) and defines the bandpass and interpolation windows.

    Args:
        spectra_observations (numpy array): array of spectral data with format (N_spectra, N_axis, N_orders, N_pixels). N_axis = 3 (wavelength, flux and flux error) and N_orders = 1 in case of 1D spectra.
        indice (str): identifier of indice / spectral line.
        indice_info (dict): Dictionary containing the line identifier ``ln_id``, the line center ``ln_ctr``, the maximum bandpass ``ln_win`` and the interpolation window ``total_win``. If None, fetches ind_table.csv.
        automatic_windows (dict): if window definition is automatic, the multiplier of the FWHM for each window.

    Attributes:
        resuls (tuple): line center, bandpass window, interpolation window, compressed array of spectral data 
    """
    def __init__(self, spectra_observations, indice, indice_info=None, automatic_windows={"ln_win_mult":4,"interp_win_mult":2}):
        
        if indice_info == None:
            ind_table = pd.read_csv("ind_table.csv")
            indice_info = ind_table[ind_table["ln_id"]==indice].to_dict(orient='records')[0]
        
        ln_ctr = indice_info["ln_ctr"]

        if len(spectra_observations.shape) == 4:
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

        if automatic_windows:
            #coadd, find_peaks, define windows
            if len(spectra_observations.shape) == 4:
                wave_grid = self._wavelength_2D_grid(spectra_obs)
                wave_coadd, flux_coadd = self._coadd_spectra_s2d(spectra_obs, wave_grid)
            else:
                wave_coadd, flux_coadd = self._coadd_spectra_s1d(spectra_obs)

            inverted_flux = -flux_coadd
            # Find peaks in the inverted flux
            peaks_indices, properties = find_peaks(inverted_flux, rel_height=0.5, width=1)
            # Extract the wavelengths and widths of the detected absorption lines
            lines_center = wave_coadd[peaks_indices]
            lines_fwhm = properties['widths'] * np.median(np.diff(wave_coadd))
 
            for i, (wavelength, fwhm) in enumerate(zip(lines_center, lines_fwhm)):
                if np.isclose(wavelength,ln_ctr,atol=0.1):
                    ln_fwhm = np.around(fwhm,5)

            ln_win = np.around(automatic_windows["ln_win_mult"]*ln_fwhm,1)
            interp_win = automatic_windows["interp_win_mult"]*ln_win

        else:
            ln_win, interp_win = indice_info["ln_win"], indice_info["interp_win"]
            
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

        wave_grid[0] = np.min(step_values)  # Minimum step size
        wave_grid[1] = np.max(lambda_min_values)  # Maximum starting wavelength
        wave_grid[2] = np.min(lambda_max_values)  # Minimum ending wavelength

        return wave_grid
    
    def _coadd_spectra_s2d(self, spectra_obs, wave_grid):
        new_disp_grid = np.arange(wave_grid[1], wave_grid[2], wave_grid[0]) * u.AA
        new_spec = Spectrum1D(spectral_axis=new_disp_grid, flux=np.zeros_like(new_disp_grid / u.AA) * u.electron)

        for i in range(spectra_obs.shape[0]):

            wave = np.array(spectra_obs[i,0,:], dtype=float)
            flux = np.array(spectra_obs[i,1,:], dtype=float)
            flux[np.isnan(flux)] = 0
            flux[flux <= 0] = 1  # Replace non-positive flux with 1

            wave = wave * u.AA
            flux = flux * u.electron

            input_spec = Spectrum1D(spectral_axis=wave, flux=flux)
            spline = SplineInterpolatedResampler()
            new_spec_sp = spline(input_spec, new_disp_grid)
            new_spec += new_spec_sp

        return new_spec.wavelength.value, new_spec.flux.value
    
    def _coadd_spectra_s1d(self, spectra_obs):
        #using the first spectrum has the common reference grid
        wave_0 = spectra_obs[0,0,:]
        delta_lambda = np.median(np.diff(wave_0))
        new_disp_grid = np.arange(np.min(wave_0)+10, np.max(wave_0)-10, delta_lambda) * u.AA
        new_spec = Spectrum1D(spectral_axis=new_disp_grid, flux=np.zeros_like(new_disp_grid / u.AA) * u.dimensionless_unscaled)

        for i in range(spectra_obs.shape[0]):

            wave = np.array(spectra_obs[i,0,:], dtype=float)
            flux = np.array(spectra_obs[i,1,:], dtype=float)
            flux[np.isnan(flux)] = 0
            flux[flux <= 0] = 1  # Replace non-positive flux with 1

            wave = wave * u.AA
            flux = flux * u.dimensionless_unscaled

            input_spec = Spectrum1D(spectral_axis=wave, flux=flux)
            spline = SplineInterpolatedResampler()
            new_spec_sp = spline(input_spec, new_disp_grid)
            new_spec += new_spec_sp

        return new_spec.wavelength.value, new_spec.flux.value