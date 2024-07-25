# TODO: add frequently used viz functions
# ideas:
# - [x] generic heatmap
# - [x] generic scatterplot
# - [x] generic barplot

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd
# import colorcet as cc

from ._plot_bar import plot_bar
from ._plot_heatmap import plot_heatmap
from ._plot_histogram import plot_histogram
from ._plot_scatter import plot_scatter
from ._excursion_plot import excursion_plot

# # heatmap parameters
# heatmap_parms = {
#                 'cbar': False,
#                 'square': False,
#                 'linewidths': 0.5,
#                 'linecolor': 'black',
#                 'center': 0.0,
#                 'vmin': -1,
#                 'vmax': 1,
#                 # 'annot': False,
#                 'fmt': '.2f',
# }

__all__ = [ 
                        "plot_bar",
                        "plot_heatmap", 
                        "plot_histogram"
                        "plot_scatter",
                        "excursion_plot"
                ]
# all = [ 
#         "plot_bar",
#         "plot_heatmap", 
#         "plot_histogram"
#         "plot_scatter",
#         "excursion_plot"
#        ]