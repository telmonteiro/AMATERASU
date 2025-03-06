# This file contains several useful functions that may be changed or rearranged later.

import numpy as np, os, re
import matplotlib.pylab as plt
from collections import defaultdict

def extract_date(file_path):
    match = re.search(r"/(\d{4}-\d{2}-\d{2})/", file_path)
    if match:
        return match.group(1)
    return None

def gather_spectra(star_name, instrument, type="2d"):
    '''
    Extracts the paths to the spectra files for each instrument, as well as the dates of the observations (name of the folder).

    Args:
        star_name (str): identifier of the star
        instrument (str): identifier of the instrument

    Returns:
        observations_dates (dict): dates of the observations
        files_instrument (list): paths of the spectra to be used
    '''
    base_dir = f'/mnt/e/2024_NIRPS_project/{star_name}/{instrument}_DRS/'
    date_folders = []; spectra_files = []

    for root, dirs, files in os.walk(base_dir): #traverse the directory tree
        for dir_name in dirs: #if the directory contains date folders (assuming they contain year-month-day format)
            date_folders.append(os.path.join(root, dir_name))
        
        for file_name in files: #collect the spectra files within each date folder
            if file_name.endswith(('.fits')):
                spectra_files.append(os.path.join(root, file_name))

    if instrument == "HARPS":
        if type == "1d": type_list = ["S1D","s1d"]
        else: type_list = ["S2D", "e2ds"]

        date_grouped_files = defaultdict(list)
        # Group files by date
        for spec in spectra_files:
            date = extract_date(spec)
            if date and "._" not in spec:
                date_grouped_files[date].append(spec)

        files_instrument = []
        for date, files in date_grouped_files.items():
            # Prioritize S2D_A over e2ds_A
            s2d_file = next((f for f in files if f.endswith(f'_{type_list[0]}_A.fits')), None)
            e2ds_file = next((f for f in files if f.endswith(f'_{type_list[1]}_A.fits')), None)

            if s2d_file:
                files_instrument.append(s2d_file)
            elif e2ds_file:
                files_instrument.append(e2ds_file)

    else:
        if type == "1d": type_list = ["S1D"]
        else: type_list = ["S2D"]
        files_instrument = [spec for spec in spectra_files if spec.endswith(f'_{type_list[0]}_TELL_CORR_A.fits') and "_r." not in spec]

    observations_dates = {extract_date(file) for file in files_instrument if extract_date(file)}

    return observations_dates, files_instrument

####################################################################

def read_headers(hdr, headers, data=None, verb=False, verbose=True):
    """Read fits header data using 'headers' dictionary.
    Result is included in 'data' dictionary if not 'None'. If 'None' a new dictionary is returned"""
    if not data:
        data = {}

    for key, hdr_id in zip(headers.keys(), headers.values()):
        try:
            if key == "rv_err" and hdr["OBJECT"] in ["GJ406","GJ643","GJ3737"]:
                data[key] = hdr[hdr_id]/1e3
            else:
                data[key] = hdr[hdr_id]
        except KeyError:
            if verbose == True:
                print(f"Header {key} not in fits file", verb)
            data[key] = None
    return data

####################################################################

def bin_data(x, y, err=None, bsize=1, stats="median", n_vals=2, estats="quad", show_plot=False):
    """Bin time series data.

    Args:
        x (list, array): Time domain values.
        y (list, array): Y-coordinate values.
        err (list, array): Uncertainties of the y-coordinate values. If no errors set to 'None'.
        bsize (int): Size of the bins in terms of time values. If time is in
            days, bsize = 1 will bin per day.
        stats (str): The statistic to apply to y-coordinate inside each bin.
            Options are 'mean', 'median', 'std' for standard deviation, 'min' for minimum and 
            'max' for maximum.
        n_vals (int): The number of points to be included in the 'min' or 'max' stats option.
        estats (int): The statistic to apply to the uncertainties, 'err'. Options are 
            'quad' for quadratically added errors and 'SEM' for standard error on the
            mean.
        show_plot (bool): If 'True' shows a plot of the procedure (for diagnostic).
    
    Returns:
        (array): Binned time domain
        (array): Binned y-coordinate
        (array): Errors of the bin statistic
    """
    # need to order the two arrays by ascending x

    x, y = zip(*sorted(zip(x, y)))

    buffer = 10e20

    x = np.asarray(x) * buffer
    y = np.asarray(y)
    bsize = bsize * buffer

    # Set errors:
    if isinstance(err, type(None)):
        err = np.zeros_like(x)
    else:
        err = np.asarray(err)

    # Create grid of bins:
    start = np.nanmin(x)
    end = np.nanmax(x) + bsize
    bins = np.arange(start, end, bsize)

    # Initialize lists:
    x_stat = np.zeros_like(bins)
    y_stat = np.zeros_like(bins)
    y_stat_err = np.zeros_like(bins)

    def calc_mean(x, err):
        if err.any():
            x_stat = np.average(x, weights=1/err**2)
        else:
            x_stat = np.average(x)
        return x_stat

    for i in range(bins.size):
        mask = (x >= bins[i]) & (x < bins[i] + bsize)

        # Dealing with no data in bin:
        if not x[mask].size:
            x_stat[i] = np.nan
            y_stat[i] = np.nan
            y_stat_err[i] = np.nan
            continue

        if stats == "mean":
            x_stat[i] = calc_mean(x[mask], err[mask])
            y_stat[i] = calc_mean(y[mask], err[mask])
        elif stats == "median":
            x_stat[i] = np.median(x[mask])
            y_stat[i] = np.median(y[mask])
        elif stats == "std":
            x_stat[i] = np.average(x[mask])
            y_stat[i] = np.std(y[mask])
        elif stats == "min":
            indices = y[mask].argsort()
            x_stat[i] = np.average(x[mask][indices][:n_vals])
            sorted_y = y[mask][y[mask].argsort()][:n_vals]
            y_stat[i] = np.average(sorted_y)
        elif stats == "max":
            indices = y[mask].argsort()
            x_stat[i] = np.average(x[mask][indices][-n_vals:])
            sorted_y = y[mask][y[mask].argsort()][-n_vals:]
            y_stat[i] = np.average(sorted_y)
        else:
            raise Exception("*** Error: 'stats' needs to be one of: 'mean', 'median', 'std', 'min', 'max'.")

        if estats == "quad":
            # quadratically added errors
            y_stat_err[i] = np.sqrt(np.sum(err[mask]**2))/y[mask].size
        elif estats == "SEM":
            # standard error on the mean
            y_stat_err[i] = np.std(y[mask])/np.sqrt(y[mask].size)
        else:
            raise Exception("*** Error: 'estats' needs to be one of: 'quad', 'SEM'.")

    remove_nan = (~np.isnan(x_stat))
    x_stat = x_stat[remove_nan]
    y_stat = y_stat[remove_nan]
    y_stat_err = y_stat_err[remove_nan]

    x_stat = np.asarray(x_stat)
    y_stat = np.asarray(y_stat)
    y_stat_err = np.asarray(y_stat_err)

    if show_plot:
        plt.figure("bin_data diagnostic")
        plt.title(f"Binning result: bsize = {bsize/buffer}")
        plt.errorbar(x/buffer, y, err, fmt='.r')
        plt.errorbar(x_stat/buffer, y_stat, y_stat_err, color='none', ecolor='k', markeredgecolor='k', marker='o', ls='')
        for step in bins:
            plt.axvline(step/buffer, color='k', ls=':', lw=0.7)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.show()

    return x_stat/buffer, y_stat, y_stat_err

####################################################################

def seq_sigma_clip(df, key, sigma=3, show_plot=False):
    """Sequencial sigma clip of given 'key' for a full DataFrame, 'df'.

    The time series of 'key' will be sequencially sigma clipped until no more values are above the 'sigma' from the median.

    Args:
        df (pandas DataFrame): DataFrame with time series to be clipped
        key (str): DataFrame column on which to apply sigma clip
        sigma (int): Sigma (number of standard deviations) value
        show_plot (bool): Show diagnostic plot

    Returns:
        DataFrame: Sigma clipped DataFrame
    """
    mask = (df[key] >= np.nanmedian(df[key]) - sigma*np.nanstd(df[key]))
    mask &= (df[key] <= np.nanmedian(df[key]) + sigma*np.nanstd(df[key]))

    if show_plot:
        plt.figure()
        plt.plot(df.bjd, df[key], 'k.', label=key)
        plt.axhline(df[key].median(), color='k', ls='--', lw=0.7)
        plt.axhline(df[key].median() + sigma*df[key].std(), color='k', ls=':', lw=0.7)
        plt.axhline(df[key].median() - sigma*df[key].std(), color='k', ls=':', lw=0.7)

    while len(df[key].dropna()[~mask]) != 0:
        df.loc[~mask, key] = np.nan

        mask = (df[key] >= np.nanmedian(df[key]) - sigma*df[key].std())
        mask &= (df[key] <= np.nanmedian(df[key]) + sigma*df[key].std())

    if show_plot:  
        plt.plot(df.bjd, df[key], color='none', marker='o', mec='r', ls='')
        plt.legend()
        plt.show()
        plt.close()

    return df