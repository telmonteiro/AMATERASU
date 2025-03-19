import numpy as np
import matplotlib.pylab as plt

def bin_data(x, y, err=None, bsize=1, show_plot=False):
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

    for i in range(bins.size):
        mask = (x >= bins[i]) & (x < bins[i] + bsize)

        # Dealing with no data in bin:
        if not x[mask].size:
            x_stat[i] = np.nan
            y_stat[i] = np.nan
            y_stat_err[i] = np.nan
            continue

        x_stat[i] = np.median(x[mask])
        y_stat[i] = np.median(y[mask])

        # quadratically added errors
        y_stat_err[i] = np.sqrt(np.sum(err[mask]**2))/y[mask].size
    
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