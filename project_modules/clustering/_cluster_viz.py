import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Union, Optional

# globals
figsize = (6,6)

DEBUG    = True
FONTSIZE = 24



#===============================================================================
def plot_feature_importances(
                            feature_set: pd.DataFrame,
                            feature_map: dict,
                            feature_colour_map: dict,
                            ax,
                            FONTSIZE: Union[int, float] = 24,
                            ):
#===============================================================================
    
    # second, plot the feature importances in the "bars" window
    try:
        sns.barplot(
                        y       = feature_set["Feature"], 
                        x       = feature_set["Importance"],
                        palette = feature_colour_map,
                        hue     = feature_set["Feature"],
                        # orient = "h",
                        ax = ax,
                        legend = False,
                        )

    except ValueError as e:
        print(e)
        
        sns.barplot(
                y       = feature_set["Feature"], 
                x       = feature_set["Importance"],
                palette = 'viridis',
                hue     = feature_set["Feature"],
                # orient = "h",
                ax = ax,
                legend = False,
                ) 


    # make the x ticks and label larger
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # no legend
    # ax.get_legend().remove()

    # turn off legend
#===============================================================================


#===============================================================================
def plot_clusters(
                    df: pd.DataFrame, 
                    site_name_dict: dict,
                    palette_dict: dict,
                    centroids: Optional[bool]    = False, 
                    title: Optional[str]         = None,
                    most_likely: Optional[bool]  = False,
                    size_by_prob: Optional[bool] = False,
                    ax: Optional[Axes]           = None,
                    FONTSIZE: Union[int, float]                = 24,
                    style:str                    = "Site",
                    centroid_kws: dict           = {},
                      ):
#===============================================================================

    # plot the embeddings
    # fig, ax = plt.subplots(figsize=(10, 10))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        # fig = None
        ax = ax
        pass

    # should we ensure that ax is not None?

    # which cluster to use
    if most_likely:
        hue = "Most Likely Cluster"
    else:
        hue = "Cluster"

    if size_by_prob:
        size = "Cluster Confidence"
    else:
        size = None

    sns.scatterplot(data = df, 
                    x           = "x",
                    y           = "y",
                    s           = 150,
                    edgecolor   = 'black', 
                    hue         = hue,
                    linewidth   = 0.5, 
                    palette     = palette_dict,
                    style       = style,
                    style_order = site_name_dict.values(),
                    markers     = ['o','D', 'P', 'X', 's'],
                    size        = size,
                    ax          = ax
                        )

    # plot the centroids
    if centroids:
        centroid_df = _make_centroids(df[["x", "y"]].values, df["Cluster"])
        _plot_centroids(centroid_df, ax, **centroid_kws)

    # set title
    if title:
        plt.title(title)

    # place the legend outside to the right of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # turn off axis labels and ticks
    if ax is None:
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")

        # make labels larger
        plt.tick_params(axis='both', which='major', labelsize=16)

        # make the legend larger
        plt.setp(ax.get_legend().get_texts(), fontsize='24') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='24') # for legend title

    # turn off x and y axis labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # turn off grid
    ax.grid(False)
    
    # axis labels
    ax.set_xlabel("Dimension 1", fontsize = FONTSIZE)
    ax.set_ylabel("Dimension 2", fontsize = FONTSIZE)


    # make the legend larger
    plt.setp(ax.get_legend().get_texts(), fontsize=FONTSIZE-2) # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize=FONTSIZE-2) # for legend title
#===============================================================================


#===============================================================================
def plot_multiple_features(
                            d: pd.DataFrame,
                            X: pd.DataFrame,
                            plot_df: pd.DataFrame,
                            feature_map: dict,
                            site_name_dict: dict,
                            site_style_dict: dict,
                            highlight_colour: str = "red",
                            ax             = None,
                            subfig_label   = None,
                            show_centroids = True,
                            fontsize: int = 24,
            ):
#===============================================================================
    
    
    sympt_dict = {0:"Absent", 1:"Present"}

    max_feature_length = min(20, len(d))


    int_feat = d["Feature"][:max_feature_length]

    n_rows = max_feature_length // 5

    if ax is not None:
        pass
    else:
        fig, ax = plt.subplots(n_rows, 5, 
                                figsize=(20, 20), 
                                sharex = True, 
                                sharey = True, 
                                layout = "constrained")

    ax  = ax.flatten()

    centroids = _make_centroids(plot_df[["x", "y"]].values, 
                                plot_df["Cluster"])

# plot the features by cluster
    for i, f in enumerate(int_feat):
        the_ax = ax[i]

        if i == 0: 
            # add an annotation
            the_ax.annotate(subfig_label,
                        xy=(0, 1.01), 
                        xycoords='axes fraction',
                        xytext=(0.0,0.0), 
                        textcoords = 'offset points',
                        fontsize = FONTSIZE*2,
                        horizontalalignment='center', 
                        verticalalignment='bottom',
                        fontweight = "bold",

                        )



        if f == "age":
            # TODO: fix this for better colours
            plot_df[f] = X[f]
            the_palette = 'magma'
            hue_order = None

        elif f == "sex":
            plot_df[f] = X[f].map({0: "Male", 1:"Female"})
            the_palette = ["#ababab", "#5577aa"]
            hue_order = ["Male", "Female"]

        else:
            plot_df[f] = X[f].map(sympt_dict)
            the_palette = ['white', highlight_colour]
            hue_order = ["Absent", "Present"]


        sns.scatterplot(data = plot_df, 
                    x         = "x", 
                    y         = "y", 
                    hue       = f,
                    hue_order = hue_order,
                    palette   = the_palette, 
                    style     = "Site", 
                    style_order = site_name_dict.values(), 
                    markers   = list(site_style_dict.values()),
                    edgecolor = 'k', 
                    linewidth = 0.125, 
                    s         = 20, 
                    # alpha     = 0.75,
                    ax        = the_ax
                    )

        if show_centroids:
            # add centroids for reference
            _plot_centroids(centroids, 
                            the_ax, 
                            alpha     = 0.25, 
                            show_edge = True)

        # turn off the ticks on the ax
        the_ax.set_xticks([])
        the_ax.set_yticks([])
        the_ax.set_xlabel("")
        the_ax.set_ylabel("")

        # if the feature_map[f] contains a slash, split it and use the first part

        if "/" in feature_map[f]:
            the_ax.set_title(feature_map[f].split("/")[0], fontsize = fontsize)
        else:
            # set axis title
            the_ax.set_title(feature_map[f], fontsize = fontsize)

        # turn off the legnd on the ax
        the_ax.legend().remove()

#===============================================================================



#===============================================================================
def _plot_centroids(centroids: pd.DataFrame, 
                   ax: Axes, 
                   alpha: Optional[float]       = 1.0, 
                   palette_dict: Optional[bool] = None, 
                   fontsize: int                = 24,
                   show_edge: Optional[bool]    = False,
                   ):
#===============================================================================
    # plot the centroids

    the_shape = "round"

    for centroid_id, centroid in centroids.iterrows():

        # don't plot the -1 cluster
        if centroid_id != -1:

            # plot in colour, if passed in
            if palette_dict is not None:
                # ec = palette_dict[centroid_id]
                ec = 'k'
                fc = palette_dict[centroid_id]
                color = 'k'
            else:
                ec = (0,0,0,alpha)
                fc = (0,0,0,alpha)
                color = 'w'

            # if show_edge:
            #     ec = 'black'

            ax.annotate(
                        str(centroid_id),
                        (centroid.x, centroid.y), 
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle=f"{the_shape},pad=0.3", 
                                    fc    = fc, 
                                    ec    = ec,
                                    lw    = 3, 
                                    # alpha = alpha,
                                    ),
                        fontsize   = fontsize,
                        color      = color,
                        fontweight = "bold",
                        )
#===============================================================================


#===============================================================================
def _make_centroids(e, lbls):
#===============================================================================
    """
    Get the centroids of the umap coordinates by cluster
    """

    # assume that lbls is a pandas series
    if isinstance(lbls, pd.Series):
        lbls = lbls.values
    
    # get the centroids of the umap coordinates by cluster
    centroids = pd.DataFrame(e, 
                                columns = ["x", "y"]).assign(cluster = lbls).groupby("cluster").mean()
    return centroids
#===============================================================================












#===============================================================================
# def plot_cluster_by_site(src,
#                          data,
#                          labels,
#                         site_name_dict: dict,

#                          title: str,
#                          tag: str,
#                          ):
# #===============================================================================

#     # make a copy of the source data
#     site_g = src.copy()

#     # augment with cluster labels
#     site_g.loc[data.index, "cluster"] = labels["Cluster"]

#     # group and count
#     cluster_by_site = site_g.groupby("cluster")["SITE"].value_counts().unstack().T

#     # fill missing values
#     cluster_by_site = cluster_by_site.fillna(0).astype(int)

#     # map SITE values using site-name_dict

#     cluster_by_site.index = cluster_by_site.index.map(site_name_dict)



#     # rename columns
#     cluster_by_site.columns = cluster_by_site.columns.astype(int)
#     # cluster_by_site = cluster_by_site.loc[site_name_dict.values()]

#     # sort index alphabetically
    

#     fig, ax = plt.subplots(figsize=(8,8), layout = "constrained")

#     sns.heatmap(cluster_by_site, 
#                 robust     = True, 
#                 ax         = ax, 
#                 annot      = True, 
#                 fmt        = "d", 
#                 cmap       = "viridis", 
#                 square     = True, 
#                 linewidths = 0.5, 
#                 linecolor  = "w", 
#                 cbar       = False)

#     # rename axis labels
#     plt.xlabel("Cluster")
#     plt.ylabel("Site")
#     plt.title(title)

#     # fn = f"cluster-membership-umap-{tag}.png"
#     # p = f"{figdir}/{fn}"

#     # plt.savefig(p, dpi = 300, bbox_inches = 'tight')
#===============================================================================