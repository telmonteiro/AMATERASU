import numpy as np
import pandas as pd

class time_series_clean:

    def __init__(self, df_raw, inds, inds_errors, sigma_clip, bin_night, sigma):

        if sigma_clip:

            df_clean = df_raw.copy()
            #sequential sigma clip
            for j in range(len(inds)):

                df_ind = self._seq_sigma_clip(df_clean, inds[j], sigma=sigma)
                df_err = self._seq_sigma_clip(df_clean, inds_errors[j], sigma=sigma)

                #combine both and set to NaN only for that pair
                mask = df_ind[inds[j]].isna() | df_err[inds_errors[j]].isna()
                df_clean.loc[mask, [inds[j], inds_errors[j]]] = np.nan

        if bin_night:
            #binning the data to days
            table = pd.DataFrame()
            bin_bjd = None

            for i in range(len(inds)):

                if pd.api.types.is_numeric_dtype(df_clean[inds[i]]):
                    #if the column contains numeric values, apply binning
                    bin_bjd, bin_column_values, bin_column_values_err = self._bin_data(df_clean["BJD"], df_clean[inds[i]], df_clean[inds_errors[i]])
                    temp_df = pd.DataFrame({f"{inds[i]}":bin_column_values, f"{inds_errors[i]}":bin_column_values_err})
                    table = pd.concat([table, temp_df], axis=1)

                    if bin_bjd is not None and i == 0:
                        table["BJD"] = bin_bjd  #add BJD column only once

                else:
                    temp_df = pd.DataFrame({f"{inds[i]}":df_clean[inds[i]], f"{inds_errors[i]}":df_clean[inds_errors[i]]})
                    table = pd.concat([table, temp_df], axis=1)

                mask = table[inds[i]].isna() | table[inds_errors[i]].isna()
                table.loc[mask, [inds[i], inds_errors[i]]] = np.nan

            df_clean = table.apply(pd.to_numeric, errors='coerce')

        if sigma_clip == False and bin_night == False:
            self.df = df_raw
            if "flux_continuum" in df_raw.columns:
                self.df["flux_continuum"] = df_raw["flux_continuum"]
        else:
            self.df = df_clean
            if "flux_continuum" in df_clean.columns:
                self.df["flux_continuum"] = df_clean["flux_continuum"]


    def _bin_data(self, x, y, err=None, bsize=1):
        """Bin time series data.

        Args:
            x (list, array): Time domain values.
            y (list, array): Y-coordinate values.
            err (list, array): Uncertainties of the y-coordinate values. If no errors set to 'None'.
            bsize (int): Size of the bins in terms of time values. If time is in
                days, bsize = 1 will bin per day.
        
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
        if isinstance(err, type(None)): err = np.zeros_like(x)
        else: err = np.asarray(err)

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

        return x_stat/buffer, y_stat, y_stat_err

    def _seq_sigma_clip(self, df, key, sigma=3):
        """Sequencial sigma clip of given 'key' for a full DataFrame, 'df'.
        The time series of 'key' will be sequencially sigma clipped until no more values are above the 'sigma' from the median.

        Args:
            df (pandas DataFrame): DataFrame with time series to be clipped
            key (str): DataFrame column on which to apply sigma clip
            sigma (int): Sigma (number of standard deviations) value

        Returns:
            DataFrame: Sigma clipped DataFrame
        """
        mask = (df[key] >= np.nanmedian(df[key]) - sigma*np.nanstd(df[key]))
        mask &= (df[key] <= np.nanmedian(df[key]) + sigma*np.nanstd(df[key]))

        while len(df[key].dropna()[~mask]) != 0:
            df.loc[~mask, key] = np.nan
            mask = (df[key] >= np.nanmedian(df[key]) - sigma*df[key].std())
            mask &= (df[key] <= np.nanmedian(df[key]) + sigma*df[key].std())

        return df