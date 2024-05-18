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

#==============================================================================
# configurations
#==============================================================================

arrowprops = dict(
                    facecolor = 'gray', 
                    edgecolor = "gray",
                    arrowstyle = "->",
                    # shrink    = 0.05, 
                    # width     = 0.125, 
                    # headwidth = 0.25,
                    )

bbox_props = dict(
                    boxstyle="round,pad=0.3", 
                    fc="white", 
                    ec="black", 
                    lw=0.5,
                    alpha=0.5
            )


#==============================================================================
def chi_p_plot(df,
               title = None,
               xlabel = None,
               ylabel = None,
               zoom_in = False,
               ):
#==============================================================================

    fig, ax = plt.subplots(figsize=(12, 12))

    df["feature"] = df.index

    # if palette is not None:
    #     c = 'k'
    #     hue = "feature"
    # else:
    c       = 'k'
    hue     = None
    palette = None

    # points are black unless they are below the p=0.05 threshold
    color = ['k' if x < 0.05 else 'r' for x in df["p-value"]]

    # plot points
    sns.scatterplot(df, 
                    x = "statistic", 
                    y = "logp", 
                    hue = hue,
                    palette = palette,
                    s = POINTSIZE,
                    ax = ax,
                    edgecolor = "black",
                    linewidth = 1,
                    c = color

                    )

    # for i, txt in enumerate(df.index):
    #     pass
        # ax.annotate(txt, (df["statistic"][i], df["logp"][i]),
                    # bbox = bbox_props,)
  
  
    # _annotate_plot(ax, df)

    # turn off legend
    # ax.get_legend().remove()

    # set y-axis name
    ax.set_ylabel(r"$-log_{10}$ (FDR-corrected p-value)", fontsize=FONTSIZE)
    ax.set_xlabel(r"$\chi^2$", fontsize=FONTSIZE)

    # set axis ticks larger
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-4)

    # title
    ax.set_title(title, fontsize=FONTSIZE)

    # draw the p=0.05 line
    _fdr_line(ax)
    # _fdr_line(ax, value = 0.10)

    # TODO: improve annotations
    # _annotate_plot(ax, df)
    import textalloc as ta
    ta.allocate_text(fig, 
                        ax, 
                        df["statistic"], 
                        df["logp"], 
                        df.index, 
                        linecolor = "gray", 
                        textsize = 12
                        )

    # if a zoomed-in plot is requested
    if zoom_in:

        # inset axis
        ins_ax = ax.inset_axes([0.05, 0.60, 0.35, 0.35]) 

        inset_range = ((0,25),(0,15))

        # get the max x value
        xmax = df["statistic"].max()
        ymax = df["logp"].max()

        inset_range = ((0,0.1*xmax),(0,0.1*ymax))

        ins_ax.set_xlim(inset_range[0])
        ins_ax.set_ylim(inset_range[1])

        # plot again
        sns.scatterplot(data          = df, 
                            x         = "statistic", 
                            y         = "logp", 
                            hue       = hue,
                            ax        = ins_ax, 
                            palette   = palette, 
                            s         = POINTSIZE/2,
                            # color     = color,
                            c         = color,
                            edgecolor = "black",
                            )

        # add fdr threshold line
        _fdr_line(ins_ax)
        # _fdr_line(ax)

        # label the plot
        # _annotate_plot(ins_ax, df, )
        # _annotate_plot_2(ins_ax, fig, df, fontsize=fontsize)

        # find the points that are within the inset range
        df_inset = df[(df["statistic"] > inset_range[0][0]) & (df["statistic"] < inset_range[0][1]) & (df["logp"] > inset_range[1][0]) & (df["logp"] < inset_range[1][1])]

        ta.allocate_text(fig, 
                    ins_ax, 
                    df_inset["statistic"], 
                    df_inset["logp"], 
                    df_inset.index, 
                    linecolor = "gray", 
                    textsize = 12,
                    # direction = "northwest",
                    # margin = 0.25,
                    )

        # turn off axis labels
        ins_ax.set_xlabel("")
        ins_ax.set_ylabel("")

        # turn off the legend in the inset
        if palette is not None:
            ins_ax.get_legend().remove()

    # turn on the zoom inset lines
    if zoom_in == True:
        ax.indicate_inset_zoom(ins_ax, facecolor="gray", edgecolor="gray", clip_on = False)


#==============================================================================
def _fdr_line(ax: plt.Axes,
                value: float = 0.05
                ):
#==============================================================================
    # draw a line at p = 0.05

    ax.axhline(-np.log(value), color="red", linestyle="--", zorder=-1)
    # get the x-axis limits
    xmin, xmax = ax.get_xlim()
    # option 1 - center the text
    xloc = (xmin+xmax)/2
    # option 2 - place the text at the right edge
    xloc = xmax - 0.05*(xmax-xmin)
    ax.annotate(
                f"p = {value}", 
                (xloc, -np.log(value)), 
                ha       = "right",
                va       = "center",
                fontsize = FONTSIZE-8,
                color    = "red",
                zorder   = 1,
                bbox     = dict(facecolor='w', 
                                alpha=1.0, 
                                edgecolor = 'red'),
                )


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
    xs = df["statistic"]
    ys = df["logp"]
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

        xx = df["statistic"][i]
        yy = df["logp"][i]

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
                    fontsize   = 8, 
                    zorder     = 0,
                    bbox       = bbox_props,
                    )

        texts.append(q)