from pandas.core.groupby import GroupBy
from scipy.stats import mannwhitneyu, kruskal
import numpy as np
import pandas
import pandas as pd
import re
import matplotlib.pyplot as plt
import colorcet as cc

from scipy.stats import chi2_contingency
# from ._chi2_test import calculate_chi2_stats



FONTSIZE = 24
DEBUG = False

arrowprops = dict(facecolor='gray', 
                    edgecolor = "gray",
                    # arrowstyle=f"-[",
                    shrink=0.05, 
                    width=0.125, 
                    headwidth=0,
                    )


#==============================================================================
def cohend(d1, d2) -> float:
#==============================================================================
    """
    Calculate the Cohen's d effect size between two samples.

    Parameters:
    d1 (array-like): The first sample.
    d2 (array-like): The second sample.

    Returns:
    float: The Cohen's d effect size.

    """

    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    # cd = (u1 - u2) / s


    # recalculate the effect size so that the sign makes sense
    # if d2 > d1, then we want a positive value for the effect size
    # because, if there is a before/after effect, we want to 
    # be able to say "the value increased" as a result of the event
    cd = (u2 - u1) / s

    return cd


#==============================================================================
def cohend_grouped(x: GroupBy):
#==============================================================================
    d1 = x.get_group(0).dropna()
    d2 = x.get_group(1).dropna()

    n1, n2 = len(d1), len(d2)
    s1, s2 = d1.var(), d2.var()
    m1, m2 = d1.mean(), d2.mean()

    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

    cd = (m2 - m1) / s
    return cd


# #==============================================================================
# def summarize_cts_data(
#                             df: pandas.DataFrame, 
#                             factor: str, 
#                             factor_map: dict = None,
#                             use_mean: bool = False,
#                             use_kruskal: bool = False,
#                             use_counts: bool = False,
#                             pval_correction: bool = False,
#                            ) -> pandas.DataFrame:
# #==============================================================================
    
#     # make an empty dataframe for results
#     results = pd.DataFrame(index=df.columns)

#     # what index corresponds to the factor?
#     factor_index = df.columns.get_loc(factor)

#     # group the dataframe by the factor
#     grouped = df.groupby(factor)

#     x = grouped.get_group(0)
#     y = grouped.get_group(1)

#     # labels
#     if factor_map is not None:
#         f0 = factor_map[0]
#         f1 = factor_map[1]
#     else:
#         f0 = 'Factor = 0'
#         f1 = 'Factor = 1'

#     if use_mean == True:
#         results[f"Mean {f0}"] = x.mean().values
#         results[f"Mean {f1}"] = y.mean().values

#         results[f"IQR {f0}"] = x.quantile(0.75) - x.quantile(0.25)
#         results[f"IQR {f1}"] = y.quantile(0.75) - y.quantile(0.25)

#         results[f"{f0}"] = results[f"Mean {f0}"].map(lambda x: f"{x:0.2f}") + " (" + results[f"IQR {f0}"].map(lambda x: f"{x:0.2f}") + ")"
#         results[f"{f1}"] = results[f"Mean {f1}"].map(lambda x: f"{x:0.2f}") + " (" + results[f"IQR {f1}"].map(lambda x: f"{x:0.2f}") + ")"

#     elif use_counts == True:
#         results[f"Count {f0}"] = x.sum().values.astype(int)
#         results[f"Count {f1}"] = y.sum().values.astype(int)

#         results[f"Total {f0}"] = len(df)
#         results[f"Total {f1}"] = len(df)

#         results[f"Percent {f0}"] = x.sum().values.astype(int) / len(df)
#         results[f"Percent {f1}"] = y.sum().values.astype(int) / len(df)

#         results[f"{f0}"] = results[f"Count {f0}"].map(lambda x: f"{x:0d}") + " (" + results[f"Percent {f0}"].map(lambda x: f"{x*100:0.1f}") + ")"
#         results[f"{f1}"] = results[f"Count {f1}"].map(lambda x: f"{x:0d}") + " (" + results[f"Percent {f1}"].map(lambda x: f"{x*100:0.1f}") + ")"


#     else:
#         results[f"Median {f0}"] = x.median().values
#         results[f"Median {f1}"] = y.median().values

#         results[f"IQR {f0}"] = x.quantile(0.75) - x.quantile(0.25)
#         results[f"IQR {f1}"] = y.quantile(0.75) - y.quantile(0.25)

#         results[f"{f0}"] = results[f"Median {f0}"].map(lambda x: f"{x:0.2f}") + " (" + results[f"IQR {f0}"].map(lambda x: f"{x:0.2f}") + ")"
#         results[f"{f1}"] = results[f"Median {f1}"].map(lambda x: f"{x:0.2f}") + " (" + results[f"IQR {f1}"].map(lambda x: f"{x:0.2f}") + ")"
        

#     if use_counts == True:
#         from ._chi2_test import calculate_chi2_stats

#         # the features are the columns of the dataframe
#         features   = df.columns
#         features   = features.drop(factor)
#         # get the results and select the corrected column
#         ggg        = calculate_chi2_stats(y,x, features, n_bins = 7)
#         chi2_pvals = ggg['p-value'].values
#         print(chi2_pvals)


#     # loop through the columns of the dataframe
#     for col in df.columns:

#         xx = x[col].dropna()
#         yy = y[col].dropna()

#         mwu_stat, mwu_p = mannwhitneyu(xx,yy, alternative='two-sided')
#         cohend_d = cohend(x[col], y[col])

#         results.loc[col, 'MWU p-value'] = mwu_p
#         results.loc[col, "Cohen's d"]   = cohend_d

#         if use_kruskal == True:
#             try:
#                 kw_stat, kw_p = kruskal(xx,yy,)
#             except:
#                 kw_stat, kw_p = np.nan, np.nan
#             # print(col, kw_stat, kw_p)

#             results.loc[col, 'KW p-value'] = kw_p

#     # rename 
#     results.rename(columns={'mwu_p': 'MWU p-value'}, inplace=True)
#     results.rename(columns={'cohen_d': "Cohen's d"}, inplace=True)

#     # drop the row with the factor name
#     results = results.drop(index=factor)

#     # drop the columns with the factor name

#     if use_mean == True:
#         results = results.drop([f"Mean {f0}", f"Mean {f1}", f"IQR {f0}", f"IQR {f1}"], axis=1)

#     elif use_counts == True:
#         results = results.drop([f"Count {f0}", f"Count {f1}", f"Total {f0}", f"Total {f1}", f"Percent {f0}", f"Percent {f1}"], axis=1)
#         results["p-value"] = chi2_pvals

#         # drop the MWU p-value and Cohen's d columns
#         results = results.drop(['MWU p-value', "Cohen's d"], axis=1)

#     else:
#         results = results.drop([f"Median {f0}", f"Median {f1}", f"IQR {f0}", f"IQR {f1}"], axis=1)

#     if pval_correction == True:
#         from statsmodels.stats.multitest import multipletests
#         # apply fdr to the p-values

#         print("Applying p-value correction in place.")

#         for col in results.columns:
#             if 'p-value' in col:
#                 results[col] = multipletests(results[col], method='fdr_bh')[1]

#     return results


# #==============================================================================
# def summarize_cat_data(
#                                 df: pandas.DataFrame, 
#                                 factor: str, 
#                                 factor_map: dict = None,
#                                 n_bins: int = 7,
#                                 pval_correction: bool = False,
#                                 use_counts: bool = False,
#                            ) -> pandas.DataFrame:
# #==============================================================================
#     # what index corresponds to the factor?
#     factor_index = df.columns.get_loc(factor)

#     # group the dataframe by the factor
#     grouped = df.groupby(factor)

#     # how many values does factor have?
#     n_values = len(grouped.groups.keys())
#     assert n_values == 2, "This function is only for binary factors."

#     # extract the two groups
#     x = grouped.get_group(0)
#     y = grouped.get_group(1)

#     # drop the factor index
#     x = x.drop(columns=factor)
#     y = y.drop(columns=factor)

#     # make an empty dataframe for results
#     results = pd.DataFrame(index=x.columns)

#     # labels
#     if factor_map is not None:
#         f0 = factor_map[0]
#         f1 = factor_map[1]
#     else:
#         f0 = 'Factor = 0'
#         f1 = 'Factor = 1'

#     results['med0'] = x.median().values
#     results['med1'] = y.median().values

#     results['iqr0'] = x.quantile(0.75) - x.quantile(0.25)
#     results['iqr1'] = y.quantile(0.75) - y.quantile(0.25)

#     # make nice labels
#     results[f"{f0}"] = results['med0'].map(lambda x: f"{x:0.2f}") \
#         + " (" + results['iqr0'].map(lambda x: f"{x:0.2f}") + ")"
#     results[f"{f1}"] = results['med1'].map(lambda x: f"{x:0.2f}") \
#         + " (" + results['iqr1'].map(lambda x: f"{x:0.2f}") + ")"
        

#     if use_counts == True:
#         # if it's a binary feature, get the counts of 1s and 0s
#         counts0 = x.sum().values
#         counts1 = y.sum().values

#         # get the total number of samples
#         total0 = len(x)
#         total1 = len(y)

#         # get the percentage of 1s and 0s
#         percent0 = counts0 / total0
#         percent1 = counts1 / total1

#         # make nice labels
#         results[f"Count {f0}"] = counts0.astype(int)
#         results[f"Count {f1}"] = counts1.astype(int)

#         results[f"Total {f0}"] = total0
#         results[f"Total {f1}"] = total1

#         results[f"Percent {f0}"] = percent0
#         results[f"Percent {f1}"] = percent1

#         results[f"{f0}"] = results[f"Count {f0}"].map(lambda x: f"{x:0d}") + " (" + results[f"Percent {f0}"].map(lambda x: f"{x*100:0.1f}") + ")"
#         results[f"{f1}"] = results[f"Count {f1}"].map(lambda x: f"{x:0d}") + " (" + results[f"Percent {f1}"].map(lambda x: f"{x*100:0.1f}") + ")"


#         contingency = pd.crosstab(x, y)
#         print(contingency)

#     # get the results and select the corrected column
#     from ._chi2_test import calculate_chi2_stats

#     # the features are the columns of the dataframe
#     features   = x.columns



#     # get the results and select the corrected column
#     ggg        = calculate_chi2_stats(x,y, features, n_bins = n_bins)

#     # set the pvalues
#     results["p-value"] = ggg['p-value'].values


#     for col in x.columns:
            
#         xx = x[col].dropna()
#         yy = y[col].dropna()

#         chi, p, dof, ex = chi2_contingency([xx.value_counts().values, yy.value_counts().values])

#         print(col, chi, p)

#     # drop intermediate columns
#     # drop columns that start wtih med or iqr
#     results = results.drop(results.filter(regex='med').columns, axis=1)
#     results = results.drop(results.filter(regex='iqr').columns, axis=1)
#     results = results.drop(results.filter(regex='Count').columns, axis=1)
#     results = results.drop(results.filter(regex='Total').columns, axis=1)
#     results = results.drop(results.filter(regex='Percent').columns, axis=1)
    
#     if pval_correction == True:
#         from statsmodels.stats.multitest import multipletests
#         # apply fdr to the p-values

#         print("Applying p-value correction in place.")

#         for col in results.columns:
#             if 'p-value' in col:
#                 results[col] = multipletests(results[col], method='fdr_bh')[1]



#     return results


#==============================================================================
def iqr(x) -> float:
#==============================================================================
    """
    calculate the interquartile range of a series

    Args:
        x (array-like): the series to calculate the IQR for

    Returns:
        float: the interquartile range of the series
    """

    return x.quantile(0.75) - x.quantile(0.25)


# #==============================================================================
# def summarize_cts_timeseries(df, 
#                             factor: str, 
#                             cat_order: list,
#                             factor_map: dict = None,
#                             pvalue_correction: bool = False,
#                             categorical: bool = False,
#                            ) -> pandas.DataFrame:
# #==============================================================================
    
#     result_dict = {}
#     result_list = []

#     # # make an empty dataframe for results
#     # results = pd.DataFrame(index=df.columns)

#     # group the dataframe by the factor
#     grouped = df.groupby(factor)

#     # make a list of the keys in the grouping
#     key_list = list(grouped.groups.keys())

#     # the comparator is the last group
#     # make sure that this is the case!
#     comparator = key_list[-1]

#     # pop the comparator from the list
#     key_list.pop()

#     # the comparator (baseline) group
#     baseline_group = grouped.get_group(comparator)

#     # drop the comparator column from baseline_group
#     baseline_group = baseline_group.drop(columns=factor)

#     # the other groups
#     for key in key_list:

#         print(key)

#         x = grouped.get_group(key)
#         x = x.drop(columns=factor)

#         # labels
#         if factor_map is not None:
#             f0 = factor_map[0]
#         else:
#             f0 = f'{key}'
#             f1 = f'{comparator}'

#         # if use_mean == True:
#         #     results[f"{f0} Mean"] = x.mean().values
#         #     results[f"{f1} Mean"] = y.mean().values
#         # else:
#         #     results[f"{f0} Median"] = x.median().values
#         #     results[f"{f1} Median"] = y.median().values

#             # loop through the columns of the dataframe
#         for col in df.columns:

#             # print(col)

#             if col == factor:
#                 continue

#             xx = x[col].dropna()
#             yy = baseline_group[col].dropna()

#             mwu_stat, mwu_p = mannwhitneyu(xx,yy, alternative='two-sided')
#             cohend_d = cohend(x[col], baseline_group[col])

#             iqr = xx.quantile(0.75) - xx.quantile(0.25)

#             record = {'bin': key, 
#                       'col': col, 
#                       "Mean": x[col].mean(), 
#                       "Median": x[col].median(), 
#                       "Median (IQR)": f"{x[col].median():0.2f} ({iqr:0.2f})",
#                       'p-value':   mwu_p, 
#                       "Cohen's d":   cohend_d, 
#                       "iqr": iqr,}

#             # add the record to the result_list
#             result_list.append(record)

#     # if use_mean == True:
#     #     results[f"{f1} Mean"] = y.mean().values
#     # else:
#     #     results[f"{f1} Median"] = y.median().values


#         # # get the results and select the corrected column
#         # ggg = calculate_chi2_stats(x,baseline_group, x.columns)
#         # print(ggg['p-value-corr'].values)


#     # for each column
#     for col in baseline_group.columns:

#         # get IQR
#         iqr = baseline_group[col].quantile(0.75) - baseline_group[col].quantile(0.25)

#         # make a record
#         record = {'bin': comparator, 
#                   'col': col, 
#                   "Mean":   baseline_group[col].mean(), 
#                   "Median": baseline_group[col].median(), 
#                   "Median (IQR)": f"{baseline_group[col].median():0.2f} ({iqr:0.2f})",
#                   'p-value': 1.0, 
#                   "Cohen's d":     0.0, 
#                   "iqr":   iqr,}

#         result_list.append(record)

#     # make a dataframe from the result_list
#     ddd = pd.DataFrame.from_records(result_list)

#     # reorder the bins
#     from pandas.api.types import CategoricalDtype
#     cdt = CategoricalDtype(categories=cat_order, ordered=True)
#     ddd["bin"] = ddd["bin"].astype(cdt)

#     if pvalue_correction == True:
#         # apply pvalue correction
#         from statsmodels.stats.multitest import multipletests

#         print("Applying p-value correction in place.")

#         for col in ddd.columns:
#             if 'p-value' in col:
#                 ddd[col] = multipletests(ddd[col], method='fdr_bh')[1]

#     # where bin == compartor, set the p-value to NaN
#     ddd.loc[ddd['bin'] == comparator, 'p-value'] = np.nan

#     return ddd



# #==============================================================================
# def summarize_cat_timeseries(df, 
#                             factor: str, 
#                             cat_order: list,
#                             factor_map: dict = None,
#                             pvalue_correction: bool = False,
#                             categorical: bool = False,
#                            ) -> pandas.DataFrame:
# #==============================================================================
    
#     result_list = []

#     # # make an empty dataframe for results
#     # results = pd.DataFrame(index=df.columns)

#     # group the dataframe by the factor
#     grouped = df.groupby(factor)

#     # make a list of the keys in the grouping
#     key_list = list(grouped.groups.keys())

#     # the comparator is the last group
#     # make sure that this is the case!
#     comparator = key_list[-1]

#     # pop the comparator from the list
#     key_list.pop()

#     # the comparator (baseline) group
#     baseline_group = grouped.get_group(comparator)

#     # drop the comparator column from baseline_group
#     baseline_group = baseline_group.drop(columns=factor)

#     # the other groups
#     for key in key_list:

#         print(key)

#         x = grouped.get_group(key)
#         x = x.drop(columns=factor)

#         # labels
#         if factor_map is not None:
#             f0 = factor_map[0]
#         else:
#             f0 = f'{key}'
#             f1 = f'{comparator}'


#         # loop through the columns of the dataframe
#         for col in df.columns:

#             if col == factor:
#                 continue

#             xx = x[col].dropna()
#             yy = baseline_group[col].dropna()

#             # chi2 test
#             vv1, bb = pd.cut(xx, bins=7, retbins=True, labels = False)
#             vv2 = pd.cut(yy, bins=bb, labels = False)

#             # counts
#             vpc = vv1.value_counts().values
#             vnc = vv2.value_counts().values

#             # check that we pad arrays if necessary
#             if len(vnc) < len(vpc):
#                 vnc = np.append(vnc, np.zeros(len(vpc) - len(vnc)))

#             if len(vpc) < len(vnc):
#                 vpc = np.append(vpc, np.zeros(len(vnc) - len(vpc)))

#             chi, p, dof, ex = chi2_contingency([vpc, vnc])

#             mwu_stat, mwu_p = mannwhitneyu(xx,yy, alternative='two-sided')
#             cohend_d = cohend(x[col], baseline_group[col])

#             iqr = xx.quantile(0.75) - xx.quantile(0.25)

#             record = {'bin': key, 
#                       'col': col, 
#                       "Mean": x[col].mean(), 
#                       "Median": x[col].median(), 
#                       "Median (IQR)": f"{x[col].median():0.2f} ({iqr:0.2f})",
#                       'stat': chi,
#                       'p-value':   p, 
#                       'mwu-p-value':   mwu_p, 
#                       "Cohen's d":   cohend_d, 
#                       "iqr": iqr,}

#             # add the record to the result_list
#             result_list.append(record)

#     # for each column
#     for col in baseline_group.columns:

#         # get IQR
#         iqr = baseline_group[col].quantile(0.75) - baseline_group[col].quantile(0.25)

#         # make a record
#         record = {'bin': comparator, 
#                   'col': col, 
#                   "Mean":   baseline_group[col].mean(), 
#                   "Median": baseline_group[col].median(), 
#                   "Median (IQR)": f"{baseline_group[col].median():0.2f} ({iqr:0.2f})",
#                   'p-value': 1.0, 
#                   'mwu-p-value': 1.0, 
#                   "Cohen's d":     0.0, 
#                   "iqr":   iqr,}

#         result_list.append(record)

#     # make a dataframe from the result_list
#     ddd = pd.DataFrame.from_records(result_list)

#     # reorder the bins
#     from pandas.api.types import CategoricalDtype
#     cdt = CategoricalDtype(categories=cat_order, ordered=True)
#     ddd["bin"] = ddd["bin"].astype(cdt)

#     if pvalue_correction == True:
#         # apply pvalue correction
#         from statsmodels.stats.multitest import multipletests

#         print("Applying p-value correction in place.")

#         for col in ddd.columns:
#             if 'p-value' in col:
#                 ddd[col] = multipletests(ddd[col], method='fdr_bh')[1]

#     # where bin == compartor, set the p-value to NaN
#     ddd.loc[ddd['bin'] == comparator, 'p-value'] = np.nan

#     return ddd


#==============================================================================
def pandas_to_latex(df_table, 
                    latex_file, 
                    vertical_bars=False, 
                    right_align_first_column=True, 
                    header=True, 
                    index=False,
                    escape=False, 
                    multicolumn=False, 
                    rotate = False,
                    **kwargs) -> None:
#==============================================================================
    # from
    # https://gist.githubusercontent.com/flutefreak7/50ffd291eaa348ead35c9794587006df/raw/9b97e7ab97105fc70f68487dbfdac75400c2bd96/pandas_to_latex.py
    """
    Function that augments pandas DataFrame.to_latex() capability.

    :param df_table: dataframe
    :param latex_file: filename to write latex table code to
    :param vertical_bars: Add vertical bars to the table (note that latex's booktabs table format that pandas uses is
                          incompatible with vertical bars, so the top/mid/bottom rules are changed to hlines.
    :param right_align_first_column: Allows option to turn off right-aligned first column
    :param header: Whether or not to display the header
    :param index: Whether or not to display the index labels
    :param escape: Whether or not to escape latex commands. Set to false to pass deliberate latex commands yourself
    :param multicolumn: Enable better handling for multi-index column headers - adds midrules
    :param kwargs: additional arguments to pass through to DataFrame.to_latex()
    :return: None
    """
    n = len(df_table.columns) + int(index)

    cols = 'l' + 'r' * (2*n - 1)

    import pandas

    if isinstance(df_table, pandas.core.frame.DataFrame):
        latex = df_table.to_latex(escape=escape, index=index, column_format=cols, header=header, multicolumn=multicolumn,
                              **kwargs)
        


    elif isinstance(df_table, pandas.io.formats.style.Styler):
        latex = df_table.to_latex(
                                    hrules         = True,
                                    convert_css    = True,
                                    column_format  = cols,
                                    multicol_align = "c",
                                    **kwargs,
                                    )

    # Multicolumn improvements - center level 1 headers and add midrules
    if multicolumn:
        latex = latex.replace(r'{l}', r'{c}')

        offset = int(index+1)
        midrule_str = ''
        for i, col in enumerate(df_table.columns.levels[0]):
            indices = np.nonzero(np.array(df_table.columns.codes[0]) == i)[0]
            hstart = 1 + offset + indices[0]
            hend = 1 + offset + indices[-1]
            midrule_str += rf'\cmidrule(lr){{{hstart}-{hend}}} '

        # Ensure that headers don't get colored by row highlighting
        midrule_str += r'\rowcolor{white}'

        latex_lines = latex.splitlines()
        latex_lines.insert(6, midrule_str)
        latex = '\n'.join(latex_lines)

    if rotate == True:
        latex = re.sub(r'\\begin{table}', r'\\begin{sidewaystable}', latex)
        latex = re.sub(r'\\end{table}', r'\\end{sidewaystable}', latex)


    # print(latex)
    with open(latex_file, 'w') as f:
        f.write(latex)


#==============================================================================
def format_tabular_data(
                        df, 
                        precision: int = 2,
                        ) :
#==============================================================================
    """
    return a formatted styler object
    """

    # get the list of columns in the df and break them into groups we anticipate

    cols = df.columns

    pvals = [col for col in cols if 'p-value' in col]
    # cohen = [col for col in cols if "Cohen's d" in col]
    cohen = [col for col in cols if col in ["Cohen's d", "d", "d-value"]]
    means = [col for col in cols if '-mean' in col]
    numerical_subset = []

    # make a styler object
    styler = df.style

    # apply numeric formatting
    format_string =  "{:0." + str(precision) + "f}"
    styler.format(format_string, subset = pvals + cohen + means, na_rep = "-")

    # colour formatting for p-vals
    styler.applymap(lambda x: "background-color: yellow" if float(x) < 0.10 else "", subset=pvals)
    styler.applymap(lambda x: "background-color: green" if float(x) < 0.05 else "", subset=pvals)

    # colour formatting for cohen's d
    styler.applymap(lambda x: "background-color: pink" if abs(float(x)) >0.20 else "", subset=cohen)
    styler.applymap(lambda x: "background-color: magenta" if abs(float(x)) >0.50 else "", subset=cohen)

    return styler