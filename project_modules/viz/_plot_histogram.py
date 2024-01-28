from typing import List, Tuple, Union, Dict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns
import numpy as np
import colorcet as cc

# figure defaults
figsize = (6,6)

#========================================
def plot_histogram(
                    df:     pd.DataFrame,
                    line1:  str,
                    line2:  Union[str, None] = None,
                    xlabel: str = 'X label',
                    ylabel: str = 'Y label',
                    legend: Union[str, None] = 'Legend',
                    palette = cc.glasbey_hv, 
                    ) -> Figure:
#========================================
    fig, ax = plt.subplots(figsize=figsize)

    # plot
    sns.histplot(
                data    = df, 
                ax      = ax, 
                palette = palette, 
                element = "step", 
                kde     = True,
                )

    # axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # titles
    if line2 is None:
        ax.set_title(line1)
    else:
        ax.set_title(f"{line1}\n{line2}")

    # # legend title
    # if legend is not None:
    #     ax.legend(title = legend)
    # else:
    #     ax.legend().set_visible(False)

    return fig
