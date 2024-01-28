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
def plot_scatter(        
                    df:     pd.DataFrame,
                    line1:  str,
                    line2:  Union[str, None] = None,
                    xlabel: str = 'X label',
                    ylabel: str = 'Y label',
                    palette = cc.glasbey_hv, 
                    hue_variable: Union[pd.Series, None] = None,
                    ) -> Figure:
#========================================
    fig, ax = plt.subplots(figsize=figsize)

    print(type(palette))

    # plot
    sns.scatterplot(
                data    = df,
                x       = df.columns[0],
                y       = df.columns[1], 
                ax      = ax, 
                palette = palette, 
                hue = hue_variable,
                )

    # axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # titles
    if line2 is None:
        ax.set_title(line1)
    else:
        ax.set_title(f"{line1}\n{line2}")

    # legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    # set legend title
    leg = ax.get_legend()
    leg.set_title(hue_variable.name)
    # set legend labels
    # if y is not None:
    #     leg_labels = y.unique()
    #     leg.set_labels(leg_labels)



    return fig


