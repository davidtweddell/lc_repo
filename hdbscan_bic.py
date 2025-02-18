# from chatgpt

import numpy as np
import hdbscan
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import hdbscan

from hdbscan import all_points_membership_vectors



def compute_bic_hdbscan(data, min_cluster_size=10, eps = 0.1):
    # Step 1: Run HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                min_samples = min_cluster_size, 
                                cluster_selection_method = 'leaf',
                                cluster_selection_epsilon = eps,
                                prediction_data = True,
                                )
    labels = clusterer.fit_predict(data)
    
    labels = np.argmax(all_points_membership_vectors(clusterer), axis = 1)

    # Step 2: Fit a Gaussian Mixture Model (GMM) to each cluster
    gmm_models = {}
    unique_clusters = set(labels)
    unique_clusters.discard(-1)  # Ignore noise points
    
    for cluster_label in unique_clusters:
        cluster_points = data[labels == cluster_label]
        gmm = GaussianMixture(n_components=1, covariance_type='full')
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
    bic = total_parameters * np.log(N) - 2 * total_log_likelihood
    
    aic = total_parameters * 2 - 2 * total_log_likelihood

    return aic, bic, len(set(labels))

# calculate  AIC



# # Example Usage
# if __name__ == "__main__":
#     # Generate sample data
#     data, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
    
#     # Compute BIC for HDBSCAN clustering
#     bic_value, cluster_labels = compute_bic_hdbscan(data)
    
#     print(f"BIC Value: {bic_value}")
