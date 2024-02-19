


### phenotype cluster

- unsupervised clustering on train set
  - reducer = umap.UMAP()
  - reducer.fit(X)
  - save the fitted reducer
  - rr = reducer.transform(X)
  - use hdbscan to cluster: HDBSCAN.fit(rr)
  - save the fitted clusterer
  - visualize a dataframe consisting of the transformed coords and the labels

- apply the fitted UMAP and HDBSCAN to transform the evaluation set and predictt labels
  - test_labels, strengths = hdbscan.approximate_predict(clusterer, test_points)
  - visualize a dataframe consisting of the transformed test coords and the labels



- given the predicted cluster, use rfc etc., to train a model to predict cluster

- transform the holdout set and predict clusters
- visualize to assess performance





### outcome cluster and prediction

- supervised clustering on train set
  - reducer = umap.UMAP()
  - reducer.fit(X, y)
  - save the fitted reducer
  - visualize a dataframe consisting of the transformed coords and the labels

- apply the fitted UMAP and HDBSCAN to transform the evaluation set and predictt labels
  - test_labels, strengths = hdbscan.approximate_predict(clusterer, test_points)
  - visualize a dataframe consisting of the transformed test coords and the labels

- given the outcome, use rfc etc., to train a model to predict cluster

- transform the holdout set and predict outcome
- visualize to assess performance
