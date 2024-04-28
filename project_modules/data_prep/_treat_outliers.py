from scipy import stats
import pandas as pd
import numpy as np
# from numpy import NDArray, bool_
from pathlib import Path

from typing import Optional

#==============================================================================
def _get_zscores(df: pd.DataFrame):
    """
    Calculate z-scores for each feature in the dataframe.
    
    Args:
    df: pd.Dataframe
        Dataframe to calculate z-scores for.
        
    Returns:
    pd.Dataframe
        Dataframe with z-scores for each feature.
    """
#==============================================================================

    # calculate z-score for each feature
    z_scores = stats.zscore(df, nan_policy="omit", axis=0)
    
    return z_scores



#==============================================================================
def _get_outlier_mask(df: pd.DataFrame,
                      z_threshold: float = 3.0):
#==============================================================================
    """
    Get a mask for the dataframe to identify outliers.
    
    Args:
    df: pd.DataFrame
        Dataframe to get the mask for.
    z_threshold: float
        Threshold for z-scores to identify outliers.
        
    Returns:
    pd.DataFrame
        Mask for the dataframe to identify outliers.
    """
    
    # get z-scores for the dataframe
    z_scores = _get_zscores(df)
    
    # print(f">>> Getting outlier mask for z-threshold = {z_threshold:0.2f}")

    # create mask for outliers
    # outlier_mask = (z_scores > z_threshold) | (z_scores < -z_threshold)
    outlier_mask = np.abs(z_scores) > z_threshold

    # calculate some summary values
    total_items = df.shape[0]
    total_outliers = np.sum(outlier_mask).sum()

    # calculate fraction of outliers per feature
    fract_outliers = pd.DataFrame(np.sum(outlier_mask, axis=0) / df.shape[0], columns=["Outlier Fraction"])
    
    print(f">>> There are {total_outliers} outliers in the data for abs(z) ≥ {z_threshold:0.2f}.")
    print(f">>> Outliers are {total_outliers/total_items*100:0.2f}% of the total data.")

    return outlier_mask


#==============================================================================
def _replace_outliers(df: pd.DataFrame, 
                      outlier_mask: pd.DataFrame,
                      method: str = "nan") -> pd.DataFrame:
#==============================================================================


    new_df = pd.DataFrame(index = df.index, columns = df.columns)

    print(f">>> Replacing outliers with {method}.")

    if method == "nan":
        # replace with nan
        replacement_value = np.nan
        df = df.mask(outlier_mask, replacement_value)

        new_df = df.mask(outlier_mask, replacement_value)


    elif method == "median":
        # calculate the median of each column
        median = df.median()
        means = df.mean()

        print(median)

        # first, fill the new_df with the original values
        new_df = df.copy()

        for col in df.columns:
            # replace the outliers with the median
            # df[col] = df[col].mask(outlier_mask[col], median[col])

            for row in df.index:
                if outlier_mask.loc[row, col]:
                    # print(f"Replacing outlier in {col} at row {row} with median = {median[col]:0.2f} (mean = {means[col]:0.2f})")

                    new_df.loc[row, col] = median[col]



    else:
        raise ValueError(f"Method {method} not supported.")

    return new_df


#==============================================================================
def _summarize_outliers(df: pd.DataFrame,
                      outlier_mask: pd.DataFrame,
                        z_threshold: float,
                        odir: Path):
#==============================================================================
    
    # calculate fraction of outliers per feature
    fract_outliers = pd.DataFrame(np.sum(outlier_mask, axis=0) / df.shape[0], columns=["Outlier Fraction"])
    
    # calculate some summary values
    total_items = df.shape[0]
    total_outliers = np.sum(outlier_mask).sum()

    fn = odir / "tab-outliers-fraction.tex"

    # make a styler and save to latex
    s = fract_outliers.style
    s.format(lambda s: '{:.2f}\%'.format(s*100))
    s.to_latex(fn, 
               caption="Fraction of outliers per feature", label="tab:fract-outliers", 
               hrules = True,
               position_float = "centering")

    ltx_zscore = f"\\newcommand{{\\zthresh}}{{{z_threshold:0.2f}}}"
    ltx_total_items = f"\\newcommand{{\\totalitems}}{{{total_items}}}"
    ltx_total_outliers = f"\\newcommand{{\\totaloutliers}}{{{total_outliers}}}"
    ltx_outliers_fraction = f"\\newcommand{{\\outliersfraction}}{{{total_outliers/total_items*100:0.2f}}}"

    with open(odir / "tex-parms-outliers.tex", "w") as f:
        # write a timestamp as a comment
        f.write(f"% Generated on {pd.Timestamp.now()}\n")
        f.write(ltx_zscore + "\n")
        f.write(ltx_total_items + "\n")
        f.write(ltx_total_outliers + "\n")
        f.write(ltx_outliers_fraction + "\n")



#==============================================================================
def treat_outliers(df: pd.DataFrame,
                   z_threshold: float = 3.0,
                   odir: Optional[Path] = None,
                   method: str = "nan",
                   ) -> pd.DataFrame:
#==============================================================================
    """
    Replace outliers in the dataframe with NaN.
    
    Args:
    df: pd.DataFrame
        Dataframe to replace outliers in.
    z_threshold: float
        Threshold for z-scores to identify outliers.
        
    Returns:
    pd.DataFrame
        Dataframe with outliers replaced with NaN.
    """

    print("="*80)
    print(f">>> Using z-threshold = {z_threshold:0.2f} to identify outliers.")
    print("="*80)

    # get mask for outliers
    outlier_mask = _get_outlier_mask(df, z_threshold)
    
    if odir is not None:
        # summarize the outliers in a latex table
        _summarize_outliers(df, outlier_mask, z_threshold, odir)

    # replace outliers
    new_df = _replace_outliers(df, outlier_mask = outlier_mask, method = method)
    
    # print(new_df.head)

    return new_df