import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from hdbscan import all_points_membership_vectors


from typing import Dict

# globals
figsize = (6,6)

DEBUG    = True
FONTSIZE = 24


#===========================================================================
def make_plot_df(p: Pipeline,
                #  LC_pos_: pd.DataFrame,
                #  X: pd.DataFrame,
                 sites: pd.Series,
                 site_name_dict: Dict[int, str]):

    # TODO: move this to a function
    points   = p["embed"].embedding_
    clusters = p["cluster"].labels_ # type: ignore
    probs    = p["cluster"].probabilities_ # type: ignore
    # sites    = LC_pos_.loc[X.index]["SITE"].map(site_name_dict)

    # try:
    # # # cluster labels are the maximally probable cluster
    #     cluster_labels = np.argmax(all_points_membership_vectors(p["cluster"]), axis = 1) # type: ignore

    # # # add two types of errors
    # except Exception as e:
    #     print(e)
    #     cluster_labels = clusters

    # print(cluster_labels)

    # reassing the most likely cluster
    cluster_labels = np.argmax(all_points_membership_vectors(p["cluster"]), axis = 1) # type: ignore
    clusters = cluster_labels

    # plot_df = pd.DataFrame(points, columns = ["x", "y"], index = X.index)
    plot_df = pd.DataFrame(points, columns = ["x", "y"], index = sites.index)
    plot_df["Cluster"]            = [c+1 for c in clusters]
    plot_df["Cluster Confidence"] = probs
    plot_df["Site"]               = sites

    return plot_df, cluster_labels
#===========================================================================



#===========================================================================
def make_feature_importance_df(classifier, 
                                  important_features) -> pd.DataFrame:


    # make a table of features and their importances from the classifier
    fi_df = pd.DataFrame(classifier.feature_importances_, 
                        index = important_features, 
                        columns = ["Importance"])
    
    # sort by importance
    fi_df = fi_df.sort_values("Importance", ascending = False)

    fi_df.reset_index(inplace = True)
    fi_df.rename(columns = {"index": "Feature"}, inplace = True)

    return fi_df
#===========================================================================



#===========================================================================
def make_topN_features(fi_df: pd.DataFrame, 
                        top_N: int,
                        feature_map: dict, 
                        feature_colour_map: dict) -> pd.DataFrame:

    # select only the top20 features
    feature_set_df = fi_df[:top_N].copy()

    # old names
    feature_set_df["OLD Feature"] = feature_set_df["Feature"]

    # rename the features
    feature_set_df["Feature"] = feature_set_df["Feature"].map(feature_map)

    # normalize the importances
    feature_set_df["Importance"] = feature_set_df["Importance"] / feature_set_df["Importance"].max()

    # assign a colour
    feature_set_df["Colour"] = feature_set_df["Feature"].map(feature_colour_map)


    return feature_set_df
#===========================================================================