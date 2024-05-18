from pandas.core.groupby import GroupBy
import numpy as np
import pandas
import pandas

import pandas as pd
import re
import matplotlib.pyplot as plt
import colorcet as cc

import seaborn as sns


FONTSIZE  = 24
DEBUG     = False
POINTSIZE = 100

# CORRECTION_METHOD = "bonferroni"
CORRECTION_METHOD = "fdr_bh"

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ranksums

from ._stats import iqr

#==============================================================================
def ts1(
                            df_grouped: GroupBy, 
                            comparator: str,
                            features: list,
                            cat_order: list,
                            ):
#==============================================================================
    
    baseline = df_grouped.get_group(comparator)
    
    # make a new dataframe to store the baseline
    qq = pd.DataFrame(baseline)

    # a list to store the results; each group will have a dataframe in this list
    result_list = []

    for g in df_grouped.groups:
        if g == comparator:
            continue
        else:
            print(g)
            group_records = []
            data = df_grouped.get_group(g)
            data = data.drop("bin", axis=1)
            comp = baseline.drop("bin", axis=1)

            for col in data.columns:

                # ct = pd.crosstab(data[col], comp[col])

                mean   = data[col].mean()
                median = data[col].median()
                std    = data[col].std()
                iqr_ = iqr(data[col])

                v1 = baseline[col].dropna()
                v2 = data[col].dropna()

                stat, p = mannwhitneyu(v1, v2)

                # wilcoxon rank sum
                from scipy.stats import ranksums
                stat, p = ranksums(v1, v2)

                # from scipy.stats import chi2_contingency
                # stat, p = chi2_contingency(ct)


                record = {
                    "bin": g,
                    "Feature": col,
                    "Mean": mean,
                    "Median": median,
                    "Std": std,
                    "IQR": iqr_,
                    "Median (IQR)": f"{median:0.2f} ({iqr_:0.2f})",
                    "statistic": stat,
                    "p-value": p,
                }
                group_records.append(record)

            df_stats = pd.DataFrame.from_records(group_records)
            # print(df_stats.head())
            result_list.append(df_stats)
            # qq = pd.concat([qq, df_stats], axis=0)

    # now handle the comparator target
    baseline = baseline.drop("bin", axis=1)
    bl_records = []

    for col in data.columns:
        mean   = baseline[col].mean()
        median = baseline[col].median()
        std    = baseline[col].std()
        iqr_ = iqr(data[col])

        v1 = baseline[col].dropna()
        v2 = data[col].dropna()

        record = {
            "bin": g,
            "Feature": col,
            "Mean": mean,
            "Median": median,
            "Std": std,
            "IQR": iqr_,
            "Median (IQR)": f"{median:0.2f} ({iqr_:0.2f})",
            "statistic": np.nan,
            "p-value": np.nan,
        }
        bl_records.append(record)
    
    bl_df = pd.DataFrame.from_records(bl_records)
    dt_df = pd.concat(result_list, axis=0)

    print(dt_df.columns)

    # apply fdr p-value correction
    from statsmodels.stats.multitest import multipletests
    dt_df["p-value"] = multipletests(dt_df["p-value"], method=CORRECTION_METHOD)[1]

    # join the baseline and the test dataframes
    qq = pd.concat([bl_df, dt_df], axis=0)


    # pibot
    q2 = qq.pivot(index = ["Feature"], 
                               values = ["Median (IQR)", "p-value", ], 
                               columns = ["bin"]).swaplevel(0,1, axis=1)\
                                .sort_index(axis=1)

    # # reset the index to match the order of neuro_features, withiout "bins"
    q2 = q2.reindex(features[:-1])

    # Drop Baseline p-value
    # ts_bio_summary = ts_bio_summary.drop("Baseline", axis=1, level=1)

    # drop columns with nan
    q2 = q2.dropna(axis=1, how="all")

    # sort the level 0 column names by cat_order
    q2 = q2.reindex(cat_order, axis=1, level=0)

    return dt_df, bl_df, q2
