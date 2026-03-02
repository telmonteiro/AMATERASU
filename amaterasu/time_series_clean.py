import numpy as np
import pandas as pd


class time_series_clean:
    """Class to clean time series data, with options for sequential sigma clipping and binning by night.
    
    Parameters
    ----------
    df_raw : pandas DataFrame
        Raw DataFrame containing the time series data to be cleaned.
    indices_lists : list of list of str
        Two lists of column names in df_raw that contain the time series data to be cleaned and their corresponding errors.
    bin_night : bool
        Whether to bin the time series data by night (default: True).
    sigma_clip : bool
        Whether to apply sequential sigma clipping to the time series data (default: True).
    sigma : int
        Sigma (number of standard deviations) value for sequential sigma clipping (default: 3).

    Attributes
    -------
    df : pandas DataFrame
        Cleaned DataFrame containing the cleaned time series data, with the same columns as df_raw for the 
        indices and their errors.

    Methods
    -------
    _bin_data(x, y, err=None, bsize=1)
        Bin time series data with specified bin size and error handling.
    _seq_sigma_clip(df, key, sigma=3)
        Apply sequential sigma clipping to a specified column until no more outliers are detected.
    """
    def __init__(self, df_raw:pd.DataFrame, indices_lists:list, bin_night:bool=True, sigma_clip:bool=True, sigma:int=3):

        indices = [x[0] for x in indices_lists]
        indices_errors = [x[1] for x in indices_lists]

        if sigma_clip:

            df_clean = df_raw.copy()
            # sequential sigma clip
            for j in range(len(indices)):

                df_ind = self._seq_sigma_clip(df_clean, indices[j], sigma=sigma)
                df_err = self._seq_sigma_clip(df_clean, indices_errors[j], sigma=sigma)

                # combine both and set to NaN only for that pair
                mask = df_ind[indices[j]].isna() | df_err[indices_errors[j]].isna()
                df_clean.loc[mask, [indices[j], indices_errors[j]]] = np.nan

        if bin_night:
            # binning the data per night of observations
            table = pd.DataFrame()
            bin_bjd = None

            for i in range(len(indices)):

                if pd.api.types.is_numeric_dtype(df_clean[indices[i]]):
                    # if the column contains numeric values, apply binning
                    bin_bjd, bin_column_values, bin_column_values_err = self._bin_data(df_clean["BJD"], df_clean[indices[i]], df_clean[indices_errors[i]])
                    temp_df = pd.DataFrame({f"{indices[i]}":bin_column_values, f"{indices_errors[i]}":bin_column_values_err})
                    table = pd.concat([table, temp_df], axis=1)

                    if bin_bjd is not None and i == 0:
                        table["BJD"] = bin_bjd  # add BJD column only once

                else:
                    temp_df = pd.DataFrame({f"{indices[i]}":df_clean[indices[i]], f"{indices_errors[i]}":df_clean[indices_errors[i]]})
                    table = pd.concat([table, temp_df], axis=1)

                mask = table[indices[i]].isna() | table[indices_errors[i]].isna()
                table.loc[mask, [indices[i], indices_errors[i]]] = np.nan

            df_clean = table.apply(pd.to_numeric, errors='coerce')

        if sigma_clip == False and bin_night == False:
            df = df_raw.copy()
            if "flux_continuum" in df_raw.columns:
                df["flux_continuum"] = df_raw["flux_continuum"]

        else:
            df = df_clean.copy()
            if "flux_continuum" in df_clean.columns:
                df["flux_continuum"] = df_clean["flux_continuum"]

        self.df = df


    def _bin_data(self, x:np.ndarray, y:np.ndarray, err:np.ndarray=None, bsize:int=1):
        """Bin time series data.

        Parameters
        ----------
        x : list, array 
            Time domain values.
        y : list, array
            Y-coordinate values.
        err : list, array
            Uncertainties of the y-coordinate values. If no errors set to 'None'.
        bsize : int 
            Size of the bins in terms of time values. If time is in days, bsize = 1 will bin per day.
        
        Returns
        -------
        binned_t : numpy array
            Binned time values (bin centers).
        binned_y : numpy array
            Binned y-coordinate values (bin medians).
        binned_y_err : numpy array
            Uncertainties of the binned y-coordinate values (quadratically added errors divided by the number of points in the bin).
        """
        # need to order the two arrays by ascending x
        x, y = zip(*sorted(zip(x, y)))
        buffer = 10e20

        x = np.asarray(x) * buffer
        y = np.asarray(y)
        bsize = bsize * buffer

        # set errors
        if isinstance(err, type(None)): 
            err = np.zeros_like(x)
        else: 
            err = np.asarray(err)

        # grid of bins
        start = np.nanmin(x)
        end = np.nanmax(x) + bsize
        bins = np.arange(start, end, bsize)

        x_stat = np.zeros_like(bins)
        y_stat = np.zeros_like(bins)
        y_stat_err = np.zeros_like(bins)

        for i in range(bins.size):
            mask = (x >= bins[i]) & (x < bins[i] + bsize)

            # no data in bin
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

        binned_t = x_stat / buffer
        binned_y = y_stat
        binned_y_err = y_stat_err

        return binned_t, binned_y, binned_y_err
    

    def _seq_sigma_clip(self, df:pd.DataFrame, key:str, sigma:int=3):
        """Sequencial sigma clip of given 'key' for a full DataFrame, 'df'.
        The time series of 'key' will be sequencially sigma clipped until no more values are above the 'sigma' from the median.

        Parameters
        ----------
        df : pandas DataFrame
            DataFrame with time series to be clipped
        key : str 
            DataFrame column on which to apply sigma clip
        sigma : int 
            Sigma (number of standard deviations) value

        Returns
        -------
        df : pandas DataFrame
            Sigma clipped DataFrame.
        """
        mask = (df[key] >= np.nanmedian(df[key]) - sigma*np.nanstd(df[key]))
        mask &= (df[key] <= np.nanmedian(df[key]) + sigma*np.nanstd(df[key]))

        while len(df[key].dropna()[~mask]) != 0:
            df.loc[~mask, key] = np.nan
            mask = (df[key] >= np.nanmedian(df[key]) - sigma*df[key].std())
            mask &= (df[key] <= np.nanmedian(df[key]) + sigma*df[key].std())

        return df