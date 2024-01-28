from typing import List, Tuple, Union, Dict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns
import numpy as np

# figure defaults
figsize = (6,6)

#========================================
def plot_heatmap(
                    m:      Union[pd.DataFrame, np.ndarray],
                    line1:  str, 
                    line2:  Union[str, None] = None,
                    labels: Union[List[str], None] = None,
                    ) -> Figure:
#========================================

    heatmap_kwargs = {
                        "annot"       : True,
                        "fmt"         : "0.2f",
                        "cbar"        : False,
                        "square"      : True, 
                        "cmap"        : 'Blues',
                        "linewidth"   : 1,
                        "linecolor"   : "k"                    
                        }


    fig, ax = plt.subplots(1,1, figsize = figsize)

    sns.heatmap(m, **heatmap_kwargs)
    
    # titles
    if line2 is None:
        ax.set_title(line1)
    else:
        ax.set_title(f"{line1}\n{line2}")

    # labels
    if labels is None:
        labels = [f"Line {i}" for i in range(m.shape[0])]
    # else:
    #     labels = [f"{label}" for label in labels]

    # tick label positions    
    ax.set_xticklabels(labels, rotation = 90)
    ax.set_yticklabels(labels, rotation = 0)

    return fig