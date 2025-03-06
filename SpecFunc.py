import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
from useful_funcs import read_headers

class SpecFunc:
   """
    Extracts and corrects data from spectrograph fits files.

    Args:
        file (str): Fits file path containing the spectral data.
        spec_class_in (str, None): Spectrograph class identification. Useful if new spectrographs are added.
        verb (bool): Turn verbose on/off.
        **spec_kw (dict): Additional keyword arguments to be passed to the spectrograph class.

    *Class attributes:*

    Attributes:
        spectrum (dict) : Dictionary containing wavelength and flux
        hdrs (dict) : Dictionary containing selected headers from 
            the fits file
        spec (object): The spectrograph class.
    """
   def __init__(self):
        self.hdrs = dict(
            obj         = 'OBJECT',
            instr       = 'INSTRUME',
            date_obs    = 'DATE-OBS',
            bjd         = f'HIERARCH ESO QC BJD',
            exptime     = f'EXPTIME',
            ra          = 'RA',
            dec         = 'DEC',
            snr7        = f'HIERARCH ESO QC ORDER{7} SNR',
            snr50       = f'HIERARCH ESO QC ORDER{50} SNR',
            snr66       = f'HIERARCH ESO QC ORDER{66} SNR', #Halpha for HARPS
            prog_id     = f'HIERARCH ESO OBS PROG ID',
            pi_coi      = f'HIERARCH ESO OBS PI-COI NAME',
            berv        = f"HIERARCH ESO QC BERV",         # [km/s]
            spec_rv     = f'HIERARCH ESO TEL TARG RADVEL',  # [km/s]
            rv          = f"HIERARCH ESO QC CCF RV",      # [km/s]
            rv_err      = f"HIERARCH ESO QC CCF RV ERROR",        # [km/s]
            fwhm        = f'HIERARCH ESO QC CCF FWHM',     # [km/s]
            fwhm_err    = f'HIERARCH ESO QC CCF FWHM ERROR', # [km/s]
            contrast    = f'HIERARCH ESO QC CCF CONTRAST', # [%]
            contrast_err = f'HIERARCH ESO QC CCF CONTRAST ERROR', # [%] 
            ccf_mask    = f"HIERARCH ESO QC CCF MASK",
            bis         = f'HIERARCH ESO QC CCF BIS SPAN', # [km/s]
            bis_err     = f'HIERARCH ESO QC CCF BIS SPAN ERROR', # [km/s]
            drs_version = f'HIERARCH ESO PRO REC1 PIPE ID',
            )
        
   def _Read(self,file,mode="vac"):

        hdul = fits.open(file)

        if "S1D" in file:
            hdr = hdul[0].header
            if mode == "air":
                wave = hdul[1].data[f"wavelength_{mode}"]
            else:
                wave = hdul[1].data["wavelength"]
            flux = hdul[1].data["flux"]
            flux_error = hdul[1].data["error"]

        elif "S2D" in file:
            hdr = hdul['PRIMARY'].header
            wave = hdul[f'WAVEDATA_{mode.upper()}_BARY'].data #AIR or VAC. 
            flux = hdul['SCIDATA'].data
            try:
                flux_error = hdul['ERRDATA'].data
            except:
                np.seterr(divide='ignore', invalid='ignore')
                flux_rel_err = np.sqrt(abs(flux))/abs(flux)
                flux_error = flux_rel_err * flux

        hdul.close()

        spectrum = {'wave':wave, 'flux':flux, 'flux_error':flux_error}

        headers = read_headers(hdr, self.hdrs, verbose=False)
        if "berv" in list(headers.keys()):
            headers["berv_err"] = 0

        snr_med, snr_orders = self._get_snr(hdr)
        headers['snr_med'] = snr_med
        for i,snr in enumerate(snr_orders):
            headers[f"snr_order{i}"] = snr

        return spectrum, headers
        
   def _RV_correction(self, spectrum_raw, header):
        
        spectrum = spectrum_raw.copy()
   
        rv = header["rv"]

        c = 299792.458 #km/s

        wave = spectrum_raw["wave"]
        spectrum["wave"] = wave / (1+rv/c)

        header["RV corrected"] = "Yes"
        
        return spectrum, header
   
   def _get_snr(self, hdr):
        
        i = 1
        snr_orders = []
        while True:
            try:
                keyword = f'HIERARCH ESO QC ORDER{i} SNR'
                snr_orders.append(hdr[keyword])
                i += 1
            except:
                try:
                    keyword = f'HIERARCH ESO DRS SPE EXT SN{i}' #old pipeline HARPS 3.8eggs
                    snr_orders.append(hdr[keyword])
                    i += 1
                except:
                    break

        snr_orders = np.array(snr_orders)
        snr_med = np.median(snr_orders)
        
        return snr_med, snr_orders
   
   @staticmethod
   def _spec_order(wave_2d, ln_ctr, ln_win):

        wmin = ln_ctr - ln_win/2
        wmax = ln_ctr + ln_win/2

        orders = []
        min_dist = []
        for i, wave in enumerate(wave_2d):
            if wmin > wave[0] and wmax < wave[-1]:
                orders.append(i)
                dist = ((wmin - wave_2d[i][0], wave_2d[i][-1] - wmax))
                min_dist.append(np.min(dist))

        # Selects the order with the higher distance to the wave edges from the bandpass limits:

        return orders[np.argmax(min_dist)]