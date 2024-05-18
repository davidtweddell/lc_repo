from pandas.core.groupby import GroupBy
from scipy.stats import mannwhitneyu, kruskal
import numpy as np
import pandas
import pandas as pd
import re
import matplotlib.pyplot as plt
import colorcet as cc

FONTSIZE = 24
DEBUG = True

arrowprops = dict(
                    facecolor='gray', 
                    edgecolor = "gray",
                    # arrowstyle=f"-[",
                    shrink=0.05, 
                    width=0.125, 
                    headwidth=0,
                    )

bbox_props = dict(
                    boxstyle="round,pad=0.3", 
                    fc="white", 
                    ec="black", 
                    lw=0.5,
                    alpha=0.5
            )

#==============================================================================
def calculate_chi2_stats(cpos: pandas.DataFrame, 
                         cneg: pandas.DataFrame,
                         features: list,
                         n_bins: int = 7) -> pandas.DataFrame:
#==============================================================================
    """
    Calculate the chi-squared statistics for a given feature
    """

    data = []

    from scipy.stats import chi2_contingency

    # calculate the number of bins

    for col in features:

        # calculate the positive and negative values
        v_pos, bb = pd.cut(cpos[col].dropna(), 
                        bins=n_bins, 
                        labels = False, 
                        retbins = True)

        #  use the same bins for the negative values
        v_neg = pd.cut(cneg[col].dropna(), 
                    bins=bb, 
                    labels = False)

        # counts
        vpc = v_pos.value_counts().values
        vnc = v_neg.value_counts().values

        # check that we pad arrays if necessary
        if len(vnc) < len(vpc):
            vnc = np.append(vnc, np.zeros(len(vpc) - len(vnc)))

        if len(vpc) < len(vnc):
            vpc = np.append(vpc, np.zeros(len(vnc) - len(vpc)))

        if DEBUG:
            print(vpc, vnc)

        # contingency = pd.crosstab(v_pos, v_neg)
        # print(contingency)

        # now do a chi-squared test
        chi2, p, dof, ex = chi2_contingency([vpc, vnc])

        # assemble a record to add to the list
        record = {"feature": col, "Chi2": chi2, "p-value": p}
        data.append(record)

    # make a dataframe
    gh = pd.DataFrame(data, index = [t["feature"] for t in data])

    # apply FDR correction
    print("Applying FDR correction to create p-value-corr column.")

    from statsmodels.stats.multitest import multipletests
    gh["p-value-corr"]       = multipletests(gh["p-value"], method="fdr_bh")[1]

    # take the negative log10 of the p-values
    gh["p-value-log10"]      = -np.log10(gh["p-value"])
    gh["p-value-corr-log10"] = -np.log10(gh["p-value-corr"])

    return gh



#==============================================================================
def plot_chi2(
                df: pandas.DataFrame,
                effect: str   = "effect",
                fontsize: int = FONTSIZE,
                inset_range: tuple = ((0,15), (0,5)),
                palette       = None,
                zoom_in: bool = False,
                figsize = (16,16),
                ):
#==============================================================================

    import seaborn as sns

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if palette is None:
        hue   = None
        color = "blue"
    else:
        hue   = "feature"
        color = None

    sns.scatterplot(data      = df, 
                    x         = "Chi2", 
                    y         = "p-value-corr-log10", 
                    hue       = hue,
                    ax        = ax, 
                    palette   = palette, 
                    s         = 100, 
                    color     = color,
                    edgecolor = "black",
                    )

    # FDR Threshold
    _fdr_threshold(ax)

    # y-axis label
    ax.set_ylabel(r"-log$_{10}$(FDR-adjusted p-value)")
    ax.set_xlabel(r"$\chi^{2}$ statistic for time effect")

    # add labels
    # sort the dataframe by chi2
    df = df.sort_values("Chi2", ascending=True, )

    # label the points

    if zoom_in == True:
        clipx = inset_range[0][1]
        clipy = inset_range[1][1]
    else:
        clipx = None
        clipy = None

    # annotations
    # TODO: come up with a good way to do this
    _annotate_plot(ax, df, fontsize=fontsize, clipx = clipx, clipy = clipy)
    # _annotate_plot_2(ax, fig, df, fontsize=fontsize)
    # from adjustText import adjust_text
    # adjust_text(texts, ax=ax, arrow_props = dict(arrowstyle='-', color='gray', alpha=.5), force_text = (0.5,0.5))

    # enlarge axis labels
    ax.set_xlabel(fr"$\chi^{2}$ statistic for {effect}", fontsize=fontsize)
    ax.set_ylabel(fr"$-log_{{10}}$ (FDR-adjusted p-value)", fontsize=fontsize)

    # enlarge ticklabels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)


    # if a zoomed-in plot is requested
    if zoom_in:

        # inset axis
        ins_ax = ax.inset_axes([0.05, 0.60, 0.35, 0.35]) 

        ins_ax.set_xlim(inset_range[0])
        ins_ax.set_ylim(inset_range[1])

        # plot again
        sns.scatterplot(data          = df, 
                            x         = "statistic", 
                            y         = "logp", 
                            hue       = hue,
                            ax        = ins_ax, 
                            palette   = palette, 
                            s         = 100, 
                            # color     = color,
                            edgecolor = "black",
                            )

        # add fdr threshold line
        _fdr_threshold(ins_ax)

        # label the plot
        _annotate_plot(ins_ax, df, fontsize=fontsize)
        # _annotate_plot_2(ins_ax, fig, df, fontsize=fontsize)

        # turn off axis labels
        ins_ax.set_xlabel("")
        ins_ax.set_ylabel("")

        # turn off the legend in the inset
        if palette is not None:
            ins_ax.get_legend().remove()

    if zoom_in:
        ax.indicate_inset_zoom(ins_ax, edgecolor="blue")
    ax.get_legend().remove()


#==============================================================================
def _fdr_threshold(ax: plt.Axes, 
                   val: float = 0.05
                   ) -> None:
#==============================================================================

    # FDR Threshold
    ax.axhline(-np.log10(val), color="red", linestyle=":", zorder=0)

    # get the x-axis limits
    xmin, xmax = ax.get_xlim()

    # option 1 - center the text
    xloc = (xmin+xmax)/2
    # option 2 - place the text at the right edge
    xloc = xmax - 0.05*(xmax-xmin)

    # label the line
    ax.text(xloc, -np.log10(val), 
            f"FDR = {val}", 
            color="red", 
            fontsize=FONTSIZE/2, 
            # ha="center", 
            ha="right", 
            va="center", 
            bbox=dict(facecolor='w', 
                      alpha=1.0, 
                      edgecolor = 'red'),
            zorder = 0)
    

#============================================================================== 
def _annotate_plot(ax: plt.Axes, 
                   df: pandas.DataFrame, 
                   fontsize: int = FONTSIZE,
                   to_uppercase: bool = False,
                   to_strip: bool = False,
                   clipx = None,
                     clipy = None,
                   ):
#============================================================================== 
    
    texts = []
    xs = df["Chi2"]
    ys = df["p-value-corr-log10"]
    nn = df.index

    # the adjustText way
    # for x, y, n in zip(xs, ys, nn):
        # texts.append(ax.text(x,y,n, bbox=bbox_props, fontsize=12, zorder=0, ha="center", va="center"))
        # texts.append(ax.annotate(n, (x,y), zorder=0, bbox=bbox_props, ha = "center", va = "center", fontsize = 12, arrowprops = arrowprops))
    # from adjustText import adjust_text
    # adjust_text(texts, ax=ax, arrow_props = dict(arrowstyle='-', color='gray', alpha=.5), force_text = (0.5,0.5))


    # my way
    for i, txt in enumerate(df.index):
        # print(i,txt)

        xx = df["Chi2"][i]
        yy = df["p-value-corr-log10"][i]

        if clipx is not None:
            if xx < clipx:
                continue

        # print(distx, disty)

        if i % 2 == 0:
            dx = 2
            dy = -2
            ha = 'left'

        else:
            dx = -2
            dy = 2
            ha = 'right'

        if to_uppercase:
            txt = str.upper(txt)

        if to_strip:
            txt = re.sub(r'\(.*?\)', '', txt)

        q = ax.annotate(txt, 
                    (xx, yy),   
                    textcoords = 'offset fontsize', 
                    xytext     = (dx,dy), 
                    ha         = ha, 
                    va         = 'center',
                    arrowprops = arrowprops, 
                    rotation   = 0, 
                    fontsize   = 12, 
                    zorder     = 0,
                    bbox       = bbox_props,
                    )

        texts.append(q)

    # from adjustText import adjust_text
    # adjust_text(texts, ax=ax, arrow_props = dict(arrowstyle='-', color='gray', alpha=.5), )

    # return texts


#============================================================================== 
def _annotate_plot_2(ax: plt.Axes, 
                     fig: plt.Figure,
                    df: pandas.DataFrame, 
                    fontsize: int = FONTSIZE,
                    to_uppercase: bool = False,
                    to_strip: bool = False,
                    clipx = None,
                     clipy = None,
                   ):
#============================================================================== 
    import textalloc as ta

    texts = []
    xs = df["Chi2"]
    ys = df["p-value-corr-log10"]
    nn = df.index

    # for x, y, n in zip(xs, ys, nn):
    #     # texts.append(ax.text(x,y,n, bbox=bbox_props, fontsize=12, zorder=0, ha="center", va="center"))
    #     # texts.append(ax.text(x,y,n))
    #     texts.append(ax.annotate(n, (x,y), zorder=0, bbox=bbox_props, ha = "center", va = "center", fontsize = 12, arrowprops = arrowprops))

    # from adjustText import adjust_text
    # adjust_text(texts, ax=ax, arrow_props = dict(arrowstyle='-', color='gray', alpha=.5), force_text = (0.5,0.5))

    text_list = [t for t in df.index]

    x = df["Chi2"]
    y = df["p-value-corr-log10"]

    if clipx is not None:
        if x < clipx:
            pass

    ta.allocate_text(fig,
                        ax,
                        x, 
                        y, 
                        text_list,
                        x_scatter = x,
                        y_scatter = y,
                        linecolor = "gray",
                        linewidth = 1,
                        textsize = 16
                )
     
