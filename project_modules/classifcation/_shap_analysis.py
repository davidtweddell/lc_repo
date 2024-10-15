from pathlib import Path
from turtle import title
from typing import Dict, List, Tuple, Union

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Axes
from rich.pretty import pprint
from sklearn.preprocessing import MinMaxScaler

# clustering
import hdbscan
import umap    

# import from viz
from project_modules.viz import red_blue

VERBOSE = False
DEBUG   = False

import matplotlib.colors as mcolors
reduced_rb = red_blue.resampled(2)
binary_red_blue  = [mcolors.rgb2hex(reduced_rb(i)) for i in range(2)]


#==============================================================================
def shap_fit(
            model, 
             X: pd.DataFrame, 
             verbose: bool = False
             ) -> Tuple[shap.Explainer, shap.Explanation]:
#==============================================================================

    print("="*50)
    print(f">>> SHAP fit")
    print("-"*50)

    
    if verbose | DEBUG:
        # print the name of the classifier type
        print(f">>> ... Classifier: {model.__class__.__name__:<34s}")

        # prettyprint the parameters
        print(f">>> ... Parameters:")
        pprint(model.get_params())

    # create the explainer
    explainer = shap.Explainer(model)

    # calculate the shap values annd store in an Explanation object
    explanation = explainer(X)

    return explainer, explanation


#==============================================================================
def shap_get_feature_importance(
                                explainer: shap.Explainer, 
                                explanation: shap.Explanation, 
                                X: pd.DataFrame, 
                                plot: bool      = False, 
                                verbose: bool   = False, 
                                top_N: Union[int, None] = None,
                              ) -> pd.DataFrame:
#==============================================================================

    print("="*50)
    print(f">>> SHAP feature importance")
    print("-"*50)

    if top_N is None:
        top_N = X.shape[1]
    else:
        pass

    if verbose | DEBUG:
        # prettyprint the parameters
        print(f">>> ... Parameters:")
        pprint(explainer.__dict__)

    if plot == True:
        plot_values = explainer.shap_values(X) # type: ignore
        shap.summary_plot(plot_values, 
                          X, 
                          class_inds = 'original',
                          plot_type = 'dot',
                          )

    # get the mean absolute shap values
    shap_mean = np.abs(explanation.values).mean(axis=0)

    if len(shap_mean.shape) > 1:
        shap_sum  = shap_mean.sum(axis=1)
        # shap_sum = np.sum(shap_mean)
    else:
        shap_sum = shap_mean

    # normalize the values by the max value
    shap_sum = shap_sum / shap_sum.max()

    # get the indices of the top n features
    idx = np.argsort(shap_sum)
    idx = idx[::-1]

    # make a dataframe of the feature names and the values
    shap_df = pd.DataFrame({'Feature':X.columns, 'Importance':shap_sum}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print(f">>> ... Top {top_N} features: {shap_df['Feature'].values[:top_N]}")
    # print(f">>> ... Important features: {shap_df['Feature'].values}")

    # return the top n features
    return shap_df


#==============================================================================
def shap_plot_points(
                        ax: Axes,
                        explanation: shap.Explanation, 
                        X: pd.DataFrame, 
                        target_class: int,
                        #
                        label_map: Union[Dict[int, str], None] = None,
                        plot_n_features = None, 
                        verbose: bool   = False, 
                        logscale: bool = False,
                        suppress_yticks: bool = False,
                        normalize_values: bool = False,
                        fontsize: int = 16,
                        edgecolor: str = 'w',
                     ) -> Axes:
#==============================================================================
    
    # this is the original SHAP colour map
    cmap = red_blue

    # reshape the explanation.values from m x n x p 
    # to "p" m x n matrices
    # where 
    # - m is the number of samples
    # - n is the number of features
    # - p is the number of classes

    # get the number of features
    n_features = explanation.values.shape[1] # type: ignore

    # get the number of samples
    n_samples = explanation.values.shape[0] # type: ignore

    # the number of features to plot
    if plot_n_features is None:
        plot_n_features = n_features

    if verbose | DEBUG:
        print(f">>> ... {explanation.values.shape}")
        print(f">>> ... n_features: {n_features}")
        print(f">>> ... n_samples: {n_samples}")

    # get the SHAP values for the target class
    if len(explanation.values.shape) > 2:
        sv = explanation.values[:, :, target_class].reshape(n_samples, n_features) # type: ignore
    else:
        sv = explanation.values.reshape(n_samples, n_features)

    if verbose | DEBUG:
        print(f">>> ... sv.shape: {sv.shape}")

    # make a dataframe with sv
    sv_df = pd.DataFrame(sv, columns = X.columns)

    # sort the dataframe by the most influential features
    sv_df = sv_df.reindex(sv_df.abs().mean().sort_values(ascending=False).index, axis=1).iloc[:, :plot_n_features]


    # scale the data for colour presentation
    X_scaled = _map_colours(X)
    # select only the items with the same columns as sv_df
    X_subset = X_scaled[sv_df.columns]
    
    # melt the dataframe to long form
    X_df_long = X_subset.melt(var_name='Feature', value_name='Feature Value')

    # melt the shap values dataframe
    sv_df_long = sv_df.melt(var_name='Feature', value_name='SHAP Value')
    
    # add the feature value
    sv_df_long['Feature Value'] = X_df_long['Feature Value']

    if normalize_values == True:
        # scale the shap values to run between -1 and 1
        sv_df_long['SHAP Value'] = sv_df_long['SHAP Value'] / sv_df_long['SHAP Value'].abs().max()
    else:
        pass


    sns.stripplot(
                    data      = sv_df_long, 
                    x         = 'SHAP Value', 
                    y         = 'Feature', 
                    orient    = 'h', 
                    hue       = 'Feature Value',
                    palette   = cmap,
                    legend    = False,
                    size      = 4,
                    ax        = ax,
                    edgecolor = edgecolor,
                    linewidth = 0.5,
                    jitter    = 0.25, # type: ignore
                    )

    # # set the title
    # if label_map is not None:
    #     ax.set_title(f"SHAP Values for Class {label_map[target_class]}")
    # else:
    #     ax.set_title(f"SHAP Values for Class {target_class}")
    # ax.set_title(f"Impact on Model Output by Feature")

    # set the x-axis label

    # set the x-range to be between -1 and 1
    if normalize_values == True:
        ax.set_xlabel("Normalized SHAP Value", fontsize=fontsize)
        ax.set_xlim(-1, 1)

    else:
        ax.set_xlabel("SHAP Value", fontsize=fontsize)
        # ax.set_xlim(-6, 6)

    # scale the x axis to be the same as the y axis
    # ax.set_aspect('equal')

    # set the y-axis label
    ax.set_ylabel("Feature", fontsize=fontsize)
    ax.set_ylim(-1, plot_n_features)

    # invert the y-axis
    ax.invert_yaxis()

    # draw a line at x = 0
    ax.axvline(x=0, color='#4a4a4b', linestyle=':',linewidth=2)

    # symmetric log scale
    if logscale == True:
        ax.set_xscale('symlog', linthresh=0.001)

    # format the y tick labels to take up a fixed amount of space
    # ax.set_yticklabels([f"{x.get_text():>10s}" for x in ax.get_yticklabels()])

    if suppress_yticks == True:
        ax.set_yticks([])
    else:
        pass

    # set x-axis ticklabel size
    # ax.tick_params(axis='x', labelsize=fontsize)
    # ax.tick_params(axis='y', labelsize=fontsize)

    # set ticklabels to fontsize
    ax.tick_params(axis='both', labelsize=fontsize)
    
    # #----------------------------------------------------------
    # # colorbar
    # #----------------------------------------------------------
    # # create a custom colour bar
    # if logscale == True:
    #     # get the symmetric log norm
    #     norm = SymLogNorm(linthresh=0.01)
    # else:
    #     norm = plt.Normalize(0,1) # type: ignore
    
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])


    # # turn off the ticks on the colorbar
    # if logscale == True:
    #     cb = plt.colorbar(sm, ax=plt.gca(), ticks=[-1, 1], aspect=50, )
    # else:
    #     cb = plt.colorbar(sm, ax=plt.gca(), ticks=[0, 1], aspect=50, )

    # cb.set_ticklabels(['Low', 'High'])

    # # cb.set_label('Feature Values', size=16, labelpad=-2)
    # cb.ax.tick_params(labelsize=16, length=0)
    # cb.set_alpha(1)
    # cb.outline.set_visible(False) # type: ignore

    # # make the colourbar half as long
    # cb.ax.set_aspect(50)



    # place colourbar in the bottom of the plot
    # cb.ax.yaxis.set_label_position('left')
    # cb.ax.yaxis.set_ticks_position('left')

    # plt.colorbar(sm, ax=ax, aspect=50, orientation='horizontal')

    # set x-axis ticklabel size
    ax.tick_params(axis='x', labelsize=fontsize)

    return ax


#==============================================================================
def shap_plot_bar(
                    ax: Axes,
                     explanation: shap.Explanation, 
                     X: pd.DataFrame, 
                     target_class: int,
                     feature_palette: Dict[str,List], 
                     #
                     label_map: Union[Dict[int, str], None] = None,
                     plot_n_features = None, 
                     verbose: bool   = False, 
                     fontsize        = 16,
                     ) -> Axes:
#==============================================================================
    
    # needs a dict of feature name and colour passed in


    # get the mean absolute values
    mean_abs_sv = np.abs(explanation.values).mean(0)

    # get the number of features
    n_features = explanation.values.shape[1] # type: ignore

    # get the number of samples
    n_samples = explanation.values.shape[0] # type: ignore

    if isinstance(explanation.values, np.ndarray) and len(explanation.values.shape) > 2:
        vv = mean_abs_sv[:, target_class]
    else:
        vv = mean_abs_sv

    # normalize the values by the max value
    vv = vv / vv.max()

    df = pd.DataFrame({'Feature':X.columns, 
                       'Mean Abs SHAP Value':vv}).sort_values(by='Mean Abs SHAP Value', ascending=False).reset_index(drop=True)

    # select the first n features
    df = df.iloc[:plot_n_features, :]

    # plot the points
    sns.barplot(
                    data      = df,
                    x         = "Mean Abs SHAP Value",
                    y         = "Feature", 
                    orient    = 'h', 
                    legend    = False,
                    ax        = ax,
                    hue       = "Feature",
                    palette   = feature_palette,
                    )

    # set the title


    # set the x-axis label
    ax.set_xlabel("Normalized mean(|SHAP Value|)", fontsize=fontsize)

    # set the y-axis label
    ax.set_ylabel("Feature", fontsize=fontsize)
    if plot_n_features is not None:
        ax.set_ylim(-1, plot_n_features)
        # ax.set_ylim(-1, plot_n_features)
    else:
        pass

    # # format the y tick labels to take up a fixed amount of space
    # ax.set_yticklabels([f"{x.get_text():>20s}" for x in ax.get_yticklabels()])

    # set x-axis ticklabel size
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)

    # set x-axis ticklabel size
    ax.tick_params(axis='x', labelsize=fontsize)

    # invert the y-axis
    ax.invert_yaxis()

    # # make axes equal
    # ax.set_aspect('equal')

    return ax


# #==============================================================================
# def shap_plot_bar_all(
#                     ax: Axes,
#                      explanation: shap.Explanation, 
#                      X: pd.DataFrame, 
#                      labels: Dict[int, str],
#                      plot_n_features = None, 
#                      verbose: bool   = False, 
#                      **kwargs
#                      ) -> Axes:
    

#     # sum the explanation values per class
#     # this is the mean absolute value of the shap values for each feature
#     if len(explanation.values.shape) > 2:
#         dd = np.abs(explanation.values).mean(axis = 0)
#     else:
#         dd = np.abs(explanation.values)

#     # make a dataframe
#     dd_df = pd.DataFrame(dd, index = X.columns, columns =labels.values()) # type:ignore

#     # sort by the sum of the means
#     dd_df = dd_df.reindex(dd_df.sum(axis=1).sort_values(ascending=False).index)

#     # select the top n items
#     dd_df = dd_df[:plot_n_features]

#     # map the state labels to colours
#     # given a dict, make a dict that maps colours to dict values
#     # def make_colour_dict(d, palette = cc.glasbey_hv):
#     #     return {key: palette[idx] for idx, key in enumerate(d.values())}

#     # cd = make_colour_dict(labels)

#     cd = {key: cc.glasbey_hv[idx] for idx, key in enumerate(labels.values())}

#     # make the plot
#     fig = dd_df.plot.barh(stacked=True, linewidth =0, color = cd, ax = ax) # type: ignore

#     # invert the y axis
#     plt.gca().invert_yaxis()

#     # set the title
#     plt.title("Overall Influence: \nMean Absolute SHAP Value by Feature")

#     # # set the title
#     # ax.set_title(f"mean(|SHAP Value|) for Class {target_class}")

#     # set the x-axis label
#     # add an x-axis label
#     plt.xlabel("mean(|SHAP Value|)")


#     # turn off grid
#     plt.grid(False)

#     return ax


# #==============================================================================
# def shap_plot_figpair(
#                      explanation: shap.Explanation, 
#                      X: pd.DataFrame, 
#                      target_class: int,
#                      feature_palette, 
#                      figdir: Path,
#                      tag: str,
#                      #
#                      labels: Union[Dict[int, str], None] = None,
#                      plot_n_features: int = 10, 
#                      verbose: bool        = False, 
#                      **kwargs
#                      ):
# #==============================================================================
#     # a frame for 2 plots
#     fig, axs = plt.subplots(1, 2, figsize=(20,10), sharey=True)

#     axs[0] = shap_plot_bar(
#                             axs[0],
#                             explanation, 
#                             X, 
#                             target_class    = target_class, 
#                             feature_palette = feature_palette, 
#                             plot_n_features = plot_n_features, 
#                             verbose         = True,
#                             label_map       = labels,
#                         )
    
#     axs[1] = shap_plot_points(
#                                 axs[1],
#                                 explanation, 
#                                 X, 
#                                 target_class    = target_class, 
#                                 plot_n_features = plot_n_features, 
#                                 logscale        = False,
#                                 label_map       = labels,
#                         )
    
#     # save the figure
#     # replace spaces
#     ftag = tag.replace(" ", "-").lower()

#     # set supertitle
#     # fig.suptitle(f"SHAP Analysis: {str.upper(tag)}", fontsize=24)
    
#     fn = f"shap-fi-{ftag}.png"
#     print(f">>> Saving figure to {figdir / fn}")
#     plt.savefig(figdir / fn,  dpi=300, bbox_inches="tight")


#==============================================================================
def shap_plot_cluster(
                        explainer: shap.Explainer,
                        X: pd.DataFrame,
                        y: pd.Series,
                        figdir: Path,
                        tag: str,
                        hue_variable: Union[str, None] = None,
                        guided: bool = False,
                        label_map: Union[Dict[int, str], None] = None,
                    ):
#==============================================================================

    # get the shap values
    shap_values = explainer(X)
    sv = shap_values.values

    if len(sv.shape) > 2:
        sv = sv[:, :, 1].reshape(X.shape[0], X.shape[1])
    else:
        pass


    # # FIXME: consider scaling the shap values to range 0-1
    svdf = pd.DataFrame(sv, columns = X.columns)
    scaler = MinMaxScaler().set_output(transform="pandas")
    sv = scaler.fit_transform(svdf)

    # print(f">>> ... shap values shape: {sv.shape}")
    # print(f">>> ... shap values type: {type(sv)}")
    print(f">>> ... SHAP values were scaled!")

    # reduce the dimensionality of the shap values
    shap_reducer = umap.UMAP(
                            n_components  = 2, 
                             random_state = 42,
                             metric       = 'euclidean',  
                             n_neighbors  = 25, 
                             verbose      = False, 
                            #  min_dist     = 0.4, 
                            #  spread       = 1.0,  
                             )

    if guided == True:
        guide_value = y
    else:
        guide_value = None
    shap_embed = shap_reducer.fit_transform(sv, y = guide_value)

    # cluster the shap values
    shap_clusterer = hdbscan.HDBSCAN(min_cluster_size=15, prediction_data=True)
    shap_clusterer.fit(shap_embed)
    shap_labels = pd.Series(shap_clusterer.labels_)
    shap_labels.index = X.index
    
    # make a df for plotting
    to_plot_df = pd.DataFrame(shap_embed, 
                              columns=["UMAP1", "UMAP2"], 
                              index=X.index)


    # use the label map if it exists
    if label_map is not None:
        to_plot_df["CLASS"] = y.map(label_map)
    else:
        to_plot_df["CLASS"] = y

    # assign the cluster number
    to_plot_df["cluster"] = shap_labels

    # # summarize the predicted labels
    # print(f">>> Cluster labels:")
    # print(to_plot_df["cluster"].value_counts())

    # handle the hue variable by scaling the data
    if hue_variable is not None:

        # how many unique values are there in the hue variable?
        n_unique = X[hue_variable].nunique()

        if n_unique > 2:
            # we want to trim the data for presentation, 
            # but retain the original values for the hue variable
            X_scaled = _map_colours(X, unit_scale = False)
            hue      = X_scaled[hue_variable]
            palette  = red_blue
        else:
            # it's a binary variable, so we can use the original values
            hue     = X[hue_variable]
            # the palette should be a list of two colours: 
            # the extrema from the red-blue palette
            palette = binary_red_blue

    else:
        hue     = "CLASS"
        palette = binary_red_blue

    fig, ax = plt.subplots(figsize=(8,8))

    sns.scatterplot(data=to_plot_df, 
                    x        = "UMAP1", 
                    y        = "UMAP2", 
                    style    = "CLASS",
                    hue      = hue,
                    palette  = palette,
                    s        = 100,
                    ax       = ax,
                )

    # turn off axes ticks
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel("UMAP1", fontsize=16)
    # plt.ylabel("UMAP2", fontsize=16)

    # # place xticks at 0 and 1
    # plt.xticks([0, 1], fontsize=16)
    # plt.yticks([0, 1], fontsize=16)

    # title
    # plt.title(f"UMAP Visualisation: {str.upper(tag)}")

    # replace spaces and set to lower case
    ftag = tag.replace(" ", "-")
    ftag = ftag.lower()

    # make legend labels larger
    plt.legend(fontsize=16)

    # save the figure
    fn = f"shap-umap-{ftag}.png"
    print(f">>> Saving figure to {figdir / fn}")
    plt.savefig(figdir / fn, dpi=300, bbox_inches="tight")

    plt.close()

#==============================================================================
def shap_analysis(
                    model, 
                    X: pd.DataFrame, 
                    y: pd.Series,
                    target_class: int,
                    figdir: Path,
                    tag: str,
                    palette: Dict[str,List], 
                    guided: bool = False,
                    label_map: Union[Dict[int, str], None] = None,
                    n_features: int = 10, 
                    umap_plot_hue: Union[str, None] = None,
                    tabdir: Union[Path, None] = None,
                    verbose: bool   = False, 
                    plot_shap: bool = False,
                    plot_umap: bool = False,
                    ):
#==============================================================================

    # explainers    
    explainer, explanation = shap_fit(model, X, verbose = verbose)

    # feature importances
    shap_fi = shap_get_feature_importance(explainer, 
                                      explanation, 
                                      X, 
                                      top_N=n_features, 
                                      plot = False, 
                                      verbose=verbose)

    ff = list(shap_fi["Feature"].values)
    fi = list(shap_fi["Importance"].values)

    # save the feature importances to a file
    if tabdir is not None:

        # in the Features column, replace the underscores with spaces
        shap_fi["Feature"] = shap_fi["Feature"].str.replace("_", "\\_")

        styler = shap_fi.style
        # ste formatter for the float values
        styler.format("{:.2f}", subset = pd.IndexSlice[:, "Importance"])

        fn = tabdir / f"tab-shap-fi-{tag}.tex"

        styler.to_latex(fn, 
                caption="Feature Importance", 
                label=f"tab:shap-fi-{tag}", 
                hrules = True,
                position_float = "centering")

    # remove spaces from tag
    tag = tag.replace(" ", "-").lower()

    if plot_shap == True:
        # bar plot
        fig, ax = plt.subplots(figsize=(8,8))
        shap_plot_bar(ax, explanation, X, target_class = 1, feature_palette = palette, verbose = verbose, plot_n_features = n_features)
        plt.savefig(figdir / f"T-shap-bar-{tag}.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        # points plot
        fig, ax = plt.subplots(figsize=(8,8))
        shap_plot_points(ax, explanation, X, target_class = 1, verbose = verbose, plot_n_features = n_features)
        plt.savefig(figdir / f"T-shap-points-{tag}.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()


    # get the explainer values
    sv = explainer(X)
    print(sv.shape)

    if model.__class__.__name__ == "XGBClassifier":
        sv = sv
    else:
        sv = explainer(X)[:, :, 1]

    # remove spaces from tag
    tag = tag.replace(" ", "-").lower()

    # clustering
    if plot_umap == True:

        fig, ax = plt.subplots(figsize=(8,8))

        # shap_plot_cluster(explainer, 
        #                 X, 
        #                 y, 
        #                 figdir, 
        #                 tag, 
        #                 umap_plot_hue, 
        #                 guided = guided,
        #                 label_map = label_map,
        #                 )
        aa = shap_cluster_2(ax, explainer, X, y, guided = guided, label_map = label_map)
        
        plt.savefig(figdir / f"T-umap-shap-{tag}.png", dpi=300, bbox_inches="tight")

        plt.close()

    record = {'feature': ff,
            'importance': fi,}

    return explainer, ff, fi


#==============================================================================
def _map_colours(X: pd.DataFrame, 
                 unit_scale: bool = True) -> pd.DataFrame:
#==============================================================================

    scaler = MinMaxScaler().set_output(transform="pandas")

    # get the number of unique values in each column
    n_unique = X.nunique()

    # get the 5th and 95th percentile
    p05 = X.astype(np.float32).quantile(0.05)
    p95 = X.astype(np.float32).quantile(0.95)

    # for each column, set values above the 95th percentile to the 95th percentile
    # and values below the 5th percentile to the 5th percentile
    X_thresh = X.copy()
    for col in X.columns:

        if n_unique[col] > 2:
            X_thresh[col] = X_thresh[col].map(lambda x: p95[col] if x > p95[col] else x)
            X_thresh[col] = X_thresh[col].map(lambda x: p05[col] if x < p05[col] else x)
        else:
            # don't do anything with binary values
            pass

    if unit_scale == True:
        # scale the thresholded data
        X_thresh = scaler.fit_transform(X_thresh)
    else:
        pass

    return X_thresh


#==============================================================================
def shap_cluster_2(
                    ax: Axes,
                    explainer: shap.Explainer,
                    # explanation: shap.Explanation,
                    X: pd.DataFrame,
                    y: pd.Series,
                    hue_variable: Union[str, None] = None,
                    guided: bool = False,
                    label_map: Union[Dict[int, str], None] = None,
                    fontsize: int = 16,
                    scale_values: bool = False,
                ) -> Axes:
#==============================================================================

    # get the shap values
    shap_values = explainer(X)
    # shap_values = explanation
    sv = shap_values.values

    if len(sv.shape) > 2:
        sv = sv[:, :, 1].reshape(X.shape[0], X.shape[1])
    else:
        pass

    print(f">>> ... Clustering: shap values shape: {sv.shape}")

    # # FIXME: consider scaling the shap values to range 0-1
    sv = pd.DataFrame(sv, columns = X.columns)

    if scale_values == True:
        scaler = MinMaxScaler().set_output(transform="pandas")
        sv = scaler.fit_transform(sv)
        print(f">>> ... SHAP values were scaled!")
    else:   
        pass

    # reduce the dimensionality of the shap values
    shap_reducer = umap.UMAP(
                            n_components  = 2, 
                             random_state = 42,
                            #  metric       = 'euclidean',  
                            #  n_neighbors  = 25, 
                            #  verbose      = False, 
                            #  min_dist     = 0.1, 
                            #  spread       = 5.0,  
                             )

    if guided == True:
        guide_value = y
    else:
        guide_value = None
    shap_embed = shap_reducer.fit_transform(sv, y = guide_value)

    # cluster the shap values
    shap_clusterer = hdbscan.HDBSCAN(min_cluster_size=15, prediction_data=True)
    shap_clusterer.fit(shap_embed)
    shap_labels = pd.Series(shap_clusterer.labels_)
    shap_labels.index = X.index
    
    # make a df for plotting
    to_plot_df = pd.DataFrame(shap_embed, 
                              columns=["UMAP1", "UMAP2"], 
                              index=X.index)


    # use the label map if it exists
    if label_map is not None:
        to_plot_df["CLASS"] = y.map(label_map)
    else:
        to_plot_df["CLASS"] = y

    # assign the cluster number
    to_plot_df["cluster"] = shap_labels

    # # summarize the predicted labels
    # print(f">>> Cluster labels:")
    # print(to_plot_df["cluster"].value_counts())

    # handle the hue variable by scaling the data
    if hue_variable is not None:

        # how many unique values are there in the hue variable?
        n_unique = X[hue_variable].nunique()

        if n_unique > 2:
            # we want to trim the data for presentation, 
            # but retain the original values for the hue variable
            X_scaled = _map_colours(X, unit_scale = False)
            hue      = X_scaled[hue_variable]
            palette  = red_blue
        else:
            # it's a binary variable, so we can use the original values
            hue     = X[hue_variable]
            # the palette should be a list of two colours: 
            # the extrema from the red-blue palette
            palette = binary_red_blue

    else:
        hue     = "CLASS"
        palette = binary_red_blue

    # fig, ax = plt.subplots(figsize=(8,8))

    # sns.scatterplot(data=to_plot_df, 
    #                 x        = "UMAP1", 
    #                 y        = "UMAP2", 
    #                 style    = "CLASS",
    #                 markers   = ["o", "^"],
    #                 hue      = hue,
    #                 palette  = cc.glasbey_hv,
    #                 hue_order = ["No Concussion", "Concussion"],
    #                 # palette  = ["black", "green"],
    #                 # palette  = palette,
    #                 s        = 150,
    #                 # alpha    = 0.5,
    #                 ax       = ax,
    #             )
    dfn = to_plot_df[to_plot_df["CLASS"] == "No Concussion"]
    dfc = to_plot_df[to_plot_df["CLASS"] == "Concussion"]

    sns.scatterplot(data=dfn, 
                    x        = "UMAP1", 
                    y        = "UMAP2", 
                    style    = "CLASS",
                    markers   = ["o", "^"],
                    hue      = hue,
                    c  = cc.glasbey_hv[0],
                    # palette  = cc.glasbey_hv,
                    hue_order = ["No Concussion", "Concussion"],
                    # palette  = ["black", "green"],
                    # palette  = palette,
                    s        = 90,
                    # alpha    = 0.5,
                    ax       = ax,
                )
    sns.scatterplot(data=dfc, 
                    x        = "UMAP1", 
                    y        = "UMAP2", 
                    # style    = "CLASS",
                    markers   = ["x"],
                    # hue      = hue,
                    c    = cc.glasbey_hv[1],
                    s        = 150,
                    # alpha    = 0.5,
                    ax       = ax,
                )

    # turn off axes ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # set axis labels
    ax.set_xlabel("UMAP1", fontsize=fontsize)
    ax.set_ylabel("UMAP2", fontsize=fontsize)

    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)

    # set legend fontsize
    ax.legend(fontsize=fontsize)

    # make legend labels larger
    # plt.legend(fontsize=16)

    # make legend dots larger
    ax.legend(markerscale=1.2)

    # set each of the legend labels to be larger
    for l in ax.legend_.texts:
        l.set_fontsize(fontsize-4)


    # put legend in top right corner
    # ax.legend(loc='lower left', fontsize=fontsize)
    return ax



# def new_viz(shap_values, 
#             feature_names=None, 
#             method="pca", 
#             show=True, 
#             alpha=1, 
#             cmap=colors.red_blue, 
#             figsize=(7,5), 
#             **kwargs):
    
#     from sklearn.decomposition import PCA
#     from sklearn.manifold import TSNE

#     if feature_names is None:
#         feature_names = [labels['FEATURE'] % str(i) for i in range(shap_values.shape[1])]

#     ind = convert_name(ind, shap_values, feature_names)
#     if ind == "sum()":
#         cvals = shap_values.sum(1)
#         fname = "sum(SHAP values)"
#     else:
#         cvals = shap_values[:,ind]
#         fname = feature_names[ind]

#     # see if we need to compute the embedding
#     if isinstance(method, str) and method == "pca":
#         pca = sklearn.decomposition.PCA(2)
#         embedding_values = pca.fit_transform(shap_values)
#     elif isinstance(method, str) and method == "tsne":
#         pca = sklearn.manifold.TSNE(2)
#         embedding_values = pca.fit_transform(shap_values)
#     elif hasattr(method, "shape") and method.shape[1] == 2:
#         embedding_values = method
#     else:
#         print("Unsupported embedding method:", method)

#     plt.scatter(
#         embedding_values[:,0], embedding_values[:,1], c=cvals,
#         cmap=colors.red_blue, alpha=alpha, linewidth=0
#     )
#     plt.axis("off")
#     #pl.title(feature_names[ind])


#     cb = plt.colorbar()
#     cb.set_label("SHAP value for\n"+fname, size=13)
#     cb.outline.set_visible(False)
#     cb.set_alpha(1)

#==============================================================================
def new_viz(shap_values, 
            ax,
            feature_values=None,
            target_values = None,
            feature_names=None, 
            colour_by = None,
            method="pca", 
            alpha=1, 
            fontsize = 16,
            cmap=red_blue, 
            **kwargs):
#==============================================================================
    
    import matplotlib.colors as mcolors
    reduced_rb = red_blue.resampled(2)
    binary_red_blue  = [mcolors.rgb2hex(reduced_rb(i)) for i in range(2)]

    from project_modules.analysis._shap_analysis import _map_colours


    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(shap_values.shape[1])]

    # default!
    ind = 0

    if feature_values is not None:
        fname = feature_names[ind]
        cvals = feature_values[fname]

    if colour_by is not None and feature_values is not None:
        fname = colour_by
        cvals = feature_values[fname]

    cbar_label = f"Feature {fname}"

    # see if we need to compute the embedding
    if isinstance(method, str) and method == "pca":
        pca = PCA(2)
        embedding_values = pca.fit_transform(shap_values)

    elif isinstance(method, str) and method == "tsne":
        pca = TSNE(2)
        embedding_values = pca.fit_transform(shap_values)

    elif hasattr(method, "shape") and method.shape[1] == 2:
        embedding_values = method
    else:
        print("Unsupported embedding method:", method)

    # scale the embeddings to run from -1 to 1 for consistent plotting
    embedding_values = MinMaxScaler(feature_range=(-1, 1)).fit_transform(embedding_values)

    # make a dataframe for plotting
    plot_df = pd.DataFrame(embedding_values, columns=["x", "y"])
    plot_df["cvals"] = cvals
    if target_values is not None:
        plot_df["style"] = target_values
        style = target_values
    else:
        style = None
        # plot_df["style"] = [None for i in range(len(plot_df))]




    # how many unique values are there in the hue variable?
    n_unique = cvals.nunique()


    if n_unique > 2:
        # we want to trim the data for presentation, 
        # but retain the original values for the hue variable
        X_scaled = _map_colours(feature_values, unit_scale = True)
        hue      = X_scaled[fname]
        # hues = "cvals"
        palette  = red_blue
    else:
        # it's a binary variable, so we can use the original values
        hue     = "cvals"
        # the palette should be a list of two colours: 
        # the extrema from the red-blue palette
        palette = binary_red_blue

    # plt.scatter(
    #             embedding_values[:,0], 
    #             embedding_values[:,1], 
    #             c         = cvals,
    #             cmap      = cmap, 
    #             alpha     = alpha, 
    #             linewidth = 1,
    #             edgecolor = 'k',
    # )

    # set the shape of th epoints
    # plt.scatter(embedding_values[:,0], embedding_values[:,1], c=cvals, cmap=cmap, alpha=alpha, linewidth=1, edgecolor='k')

    # make a colorbar for the feature values

    # fig, ax = plt.subplots(figsize=(8,8))

    sns.scatterplot(
                        data=plot_df,
                        x="x", 
                        y="y", 
                        hue=hue, 
                        style=target_values,
                        palette=palette, 
                        alpha=alpha, 
                        linewidth=1, 
                        edgecolor="w",
                        # size=100,
                        ax = ax
                     )

    # # make a colorbar for the feature values
    # norm = plt.Normalize(min(cvals), max(cvals))
    # sm = plt.cm.ScalarMappable(cmap=red_blue, norm=norm)
    # ax.figure.colorbar(sm, ax=ax).set_label(cbar_label, size=16)

    
    # cb = plt.colorbar()
    # cb.set_label(cbar_label)
    # # cb.outline.set_visible(False)
    # cb.set_alpha(1)
    # # make cb horizontal across bottom
    # cb.ax.yaxis.set_tick_params(color='black')
    
    # make the colorbar horizontal

    # turn off axis ticks and labels
    # plt.gca().set_xticks([])
    # plt.gca().set_yticks([])
    # plt.gca().set_xlabel("")
    # plt.gca().set_ylabel("")

    # turn off legend
    # plt.gca().get_legend().remove()

    # set to square
    # plt.gca().set_aspect('equal', )

    ax.set_xlabel(f"DIM 1", fontsize=fontsize)
    ax.set_ylabel(f"DIM 2", fontsize=fontsize)
    # ax.set_xlabel(f"{str.upper(method)} 1", fontsize=fontsize)
    # ax.set_ylabel(f"{str.upper(method)} 2", fontsize=fontsize)

    # set the aspect ratio to 1
    # ax.set_aspect('equal')
