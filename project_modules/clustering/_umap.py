import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from typing import Union, List, Dict, Tuple, Optional

# globals
figsize = (6,6)

DEBUG = True

#========================================
def _reduce(
                X: pd.DataFrame, 
                y: Optional[pd.Series] = None,
                min_dist: float   = 0.15, 
                n_components: int = 2, 
                verbose: bool     = False
            ) -> np.array:  # type: ignore
#========================================

    import umap

    # instantiate the reducer
    um   = umap.UMAP(
                        metric       = 'euclidean', 
                        min_dist     = min_dist,  
                        random_state = 42, 
                        verbose      = verbose,
                    )

    # fit and transform the data
    rr = um.fit_transform(np.array(X), y = y)

    # return the transformed coordinates
    return rr


#========================================
def _cluster(X: np.array,
             verbose: bool = False
             ): # type: ignore
#========================================
    import hdbscan

    hdb = hdbscan.HDBSCAN(
                            min_cluster_size         = 20, 
                            metric                   = 'euclidean', 
                            cluster_selection_method = 'eom', 
                            prediction_data          = True
                        )
    
    # fit the clusterer
    hdb.fit(X)

    # assing clusters to the most probable class
    cluster_vec = hdbscan.all_points_membership_vectors(hdb)
    cluster_id = [np.argmax(c) for c in cluster_vec] #type: ignore

    if DEBUG:
        print(cluster_vec[:5])

    # convert the cluster vector to a list
    cluster_vec = [list(c) for c in cluster_vec] #type: ignore

    return cluster_vec, cluster_id


#================================================================
def _style_correct(pred, true, verbose=False) -> pd.Series:
#================================================================
    
    # the labels to apply to the mask
    the_labels = {1:'Correct', 0:'Incorrect'}
    
    pred = pd.Series(pred, index = true.index)
    true = pd.Series(true)

    # make a true/false mask of the predictions versus the true values
    # if they are the same (True), the prediction is correct
    mask = (pred == true)

    # apply the labels to the mask and make a series
    correct = list(map(the_labels.get, mask))
    correct_df = pd.Series(correct, index = pred.index).rename('Correct/Incorrect')

    # print the value counts if requested
    if verbose:
        print(correct_df.value_counts())
    else:
        pass

    return correct_df


#========================================
def prep_umap(
                X: Union[pd.DataFrame, List],
                y: Union[pd.Series, List, None] = None,
                verbose: bool = False
            ) -> Tuple[np.array, np.array, np.array]: # type: ignore
#========================================

    if isinstance(X, pd.DataFrame):
        pass
    elif isinstance(X, list):
        X = pd.DataFrame(X)

    print(f">>> [umap] ... reducing")
    r = _reduce(X, verbose = verbose)

    print(f">>> [umap] ... clustering")
    c_vec, c_id = _cluster(r, verbose = verbose)

    return r, c_vec, c_id


#========================================
def plot_umap(
                coords:      np.array,   # type: ignore
                hue:         pd.Series, 
                style:       pd.Series, 
                hue_order:   list, 
                style_order: list, 
                palette:     Union[List, Dict], 
                stitle:      str, 
                line2:       str,
              ) -> Figure:
#========================================

    fig, ax = plt.subplots(figsize=figsize)

    f = sns.scatterplot(    
                        x            = coords[:,0], 
                        y            = coords[:,1],
                        hue          = hue, 
                        hue_order    = hue_order,
                        style        = style,
                        style_order  = style_order,
                        palette      = palette,
                        edgecolor    = 'k',
                        linewidth    = 1,
                        markers      = ['o', 'v', 's'],  # type: ignore
                        s            = 50, 
                        alpha        = 0.75,
                        ax           = ax,
                            )   

    # set the title
    ax.set_title(f"Separation - {stitle}\n{line2}")

    # remove the axis ticks
    f.set_xticks([])
    f.set_yticks([])

    return fig