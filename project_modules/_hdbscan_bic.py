import pandas as pd
import numpy as np

import hdbscan
from hdbscan.prediction import all_points_membership_vectors
from sklearn.mixture import GaussianMixture


import seaborn as sns
import matplotlib.pyplot as plt


# ======================================
def _compute_bic_hdbscan(
                            data: pd.DataFrame,
                            c_min: int = 10,
                            eps: float = 0.1,
                            method: str = "leaf",
                            prediction_data: bool = True,
):
    # ======================================
    # Step 1: Run HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
                                min_cluster_size=c_min,
                                min_samples=c_min,
                                cluster_selection_method=method,
                                cluster_selection_epsilon=eps,
                                prediction_data=prediction_data,
    )

    labels = clusterer.fit_predict(data.values)
    # labels = np.argmax(all_points_membership_vectors(clusterer), axis = 1)

    # if we were clustering , we would then assign the noise points (-1) to their most likely cluster.
    # however, we are not clustering, so we will ignore the noise points
    unique_clusters = set(labels)
    unique_clusters.discard(-1)

    # print(unique_clusters)

    # Step 2: Fit a Gaussian Mixture Model (GMM) to each cluster
    gmm_models = {}
    unique_clusters = set(labels)
    unique_clusters.discard(-1)  # Ignore noise points

    for cluster_label in unique_clusters:
        cluster_points = data[labels == cluster_label]
        gmm = GaussianMixture(n_components=1, covariance_type="full")
        gmm.fit(cluster_points)
        gmm_models[cluster_label] = gmm

    # Step 3: Compute total log-likelihood (L)
    total_log_likelihood = sum(
        gmm.score_samples(data[labels == cluster]).sum()
        for cluster, gmm in gmm_models.items()
    )

    # Step 4: Compute the number of parameters (k)
    d = data.shape[1]  # Number of features
    total_parameters = sum(
        d + (d * (d + 1)) // 2  # Mean + covariance matrix parameters
        for _ in gmm_models
    )

    # Step 5: Compute BIC
    N = sum(len(data[labels == cluster]) for cluster in gmm_models)  # Exclude noise
    aic = total_parameters * 2 - 2 * total_log_likelihood
    bic = total_parameters * np.log(N) - 2 * total_log_likelihood

    return aic, bic, len(unique_clusters)


# ======================================
def _make_plot_df(embedding: np.ndarray, c_min: int, c_max: int, c_step: int):
    # ======================================

    results = {}

    for mcs in range(c_min, c_max, c_step):
        r = _compute_bic_hdbscan(embedding, mcs)
        results[mcs] = r

    # make a dataframe for this
    plot_df = pd.DataFrame(results).T

    # name the columns aic, bic, N
    plot_df.columns = ["aic", "bic", "N"]

    # reset the index and make the old index a column called mcs
    plot_df.reset_index(inplace=True)

    plot_df.rename(columns={"index": "mcs"}, inplace=True)

    # add a column x_star which is the normalized x value
    plot_df["x_star"] = plot_df.index / plot_df.index.max()

    # add a column N_star which runs from 0 to 1
    plot_df["N_star"] = 1 - (plot_df["N"] - plot_df["N"].min()) / (
        plot_df["N"].max() - plot_df["N"].min()
    )

    return plot_df


# ======================================
def _make_kneeplot(
    df: pd.DataFrame,
    x_col: str = "mcs",
    y_col: str = "N",
    direction: str = "decreasing",
    curve: str = "convex",
):
    # ======================================
    # find the elbow point
    from kneed import KneeLocator

    kl = KneeLocator(
        df["mcs"], df["N"], curve=curve, direction=direction, online=False, S=1
    )

    print(kl.all_knees, kl.all_elbows_y)

    f1 = kl.plot_knee_normalized()
    f2 = kl.plot_knee()

    # add axis labels
    plt.xlabel("Minimum Cluster Size")
    plt.ylabel("Number of Clusters")

    # get the first item in the knees list
    # knee = kl.all_knees[0]

    # extract an item from a set object
    knee = list(kl.all_knees)[0]

    # get the x value for the knee
    knee_y = kl.all_elbows_y[0]

    return f1, f2, (knee, knee_y)


# ======================================
def _plot_xstar_nstar(df: pd.DataFrame, 
                      c_min: int, 
                      c_max: int, 
                      pts = None
                    #   mcs_sqrt = None
                      ):
    # ======================================
    # plot N_star vs x_star

    # figure canvas
    fig, ax = plt.subplots(figsize=(8,8))

    # plots
    sns.lineplot(x="x_star", y="N_star", data=df, ax=ax)
    sns.scatterplot(x="x_star", y="N_star", data=df, ax=ax, s=100)

    # add horizontal line at y = 0.9
    ax.axhline(0.9, color="gray", linestyle="--")

    # axis labels
    ax.set_xlabel(r"Normalized Minimum Cluster Size: $M^{*}$", fontsize=18)
    # ax.set_xlabel(r"Normalized Minimum Cluster Size: MCS/MCS$_{max}$", fontsize=18)
    ax.set_ylabel(r"Normalized Number of Clusters: $N^*$", fontsize=18)

    # set ticklabels larger
    ax.tick_params(axis="both", labelsize=18)

    # mark the point closest to y = 0.9
    x_star = df["x_star"].values

    # find the index of the value closest to 0.9
    # idx = (np.abs(df["N_star"] - 0.9)).idxmin() + 1

    # find the index of the value greater than 0.9
    idx = df[df["N_star"] > 0.9].index[0]

    # annotation
    plt.annotate(
        f"Above Threshold at\nN = {int(df['N'][idx]):d}, MCS = {df['mcs'][idx]}",
        (x_star[idx], df["N_star"][idx]),
        (x_star[idx] + 0.1, 0.9 - 0.075),
        arrowprops=dict(facecolor="black", shrink=0.05),
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="black"),
        fontsize=18,
        verticalalignment="bottom",
        horizontalalignment="left",
    )

    # check that pts is a tuple
    if pts is not None:
        print(pts)
        if isinstance(pts, tuple):
            idx = df[df["N"] == pts[1]].index[0]
            plt.annotate(
                f"Knee at\nN = {int(df['N'][idx]):d}, MCS = {df['mcs'][idx]}",
                (x_star[idx], df["N_star"][idx]),
                (x_star[idx] + 0.1, df["N_star"][idx] - 0.075),
                arrowprops=dict(facecolor="black", shrink=0.05),
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="black"),
                fontsize=18,
                verticalalignment="bottom",
                horizontalalignment="left",
            )
    # title

    # if mcs_sqrt is None:
    #     plt.title(
    #         f"Elbow Method for HDBSCAN\nMin Cluster Size from {c_min} to {c_max}",
    #         fontsize=20,
    #     )
    # else:
    plt.title(
        f"Elbow Method for HDBSCAN\nMin Cluster Size from {c_min} to {c_max}",
        fontsize=20,
    )

    # # annotate with the c_min, c_max, c_step
    # plt.annotate(
    #         f"c_min = {c_min}, c_max = {c_max}, c_step = {c_step}",
    #         (0.9, 0.0),
    #         (0.9, 0.0),
    #         fontsize = 12,
    #             verticalalignment = "bottom",
    #             horizontalalignment = "right",
    # )

    return fig, ax


# ===================================================================
def make_plot(
    df: pd.DataFrame,
    c_min: int = 10,
    c_max: int = 100,
    c_step: int = 10,
    make_kneed_plot: bool = False,
    # mcs_sqrt = None,
):
    # ===================================================================

    plot_df = _make_plot_df(
        df,
        c_min=c_min,
        c_max=c_max,
        c_step=c_step,
    )
    if make_kneed_plot:
        f2, f3, pts = _make_kneeplot(plot_df)
    else:
        f2, f3, pts = None, None, None

    f1, ax1 = _plot_xstar_nstar(plot_df, 
                                c_min=c_min, 
                                c_max=c_max, 
                                pts = pts,
                                # mcs_sqrt=mcs_sqrt
                                )
    


    return f1, ax1
