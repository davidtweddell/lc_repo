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
def plot_bar(
                df: pd.DataFrame, 
                line1: str,
                line2: str = None,
                palette: Union[List[str], Dict[str, str]] = cc.glasbey_hv,
                xlabel: str = 'X label',
                ylabel: str = 'Y label',
                ) -> Figure:
#========================================

    fig, ax = plt.subplots(figsize=figsize)

    # plot
    sns.barplot(
                data    = df, 
                ax      = ax, 
                palette = palette
                )

    # axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # titles
    if line2 is None:
        ax.set_title(line1)
    else:
        ax.set_title(f"{line1}\n{line2}")

    return fig
