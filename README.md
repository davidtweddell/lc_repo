# LC OPTIMIZE

## NOTES
- look at `LC.xlsx` - it's a copy with mods of the doc that Doug sent
- I added columns (headers in all caps) with some thoughts about which data rows to use, meanings, etc.
- Please have a look and LMK what you think

## TODO

- [ ] item
- [ ] item

## Algorithm sketches and ideas

### phenotype cluster

- unsupervised clustering on train set

```python
reducer = umap.UMAP()
reducer.fit(X)
# could save the fitted reducer to a file if we want
rr = reducer.transform(X)

hdb = hdbscan.HDBSCAN()
clusterer = hdb.fit(rr)
hdb.fit(rr)

clusters = hdb.labels_
# could save the fitted clusterer to a file if we want
```
 - assemble a dataframe, etc., to visualize the transformed coords and the labels

- apply the fitted UMAP and HDBSCAN to transform the evaluation set and predict labels
```
test_labels, strengths = hdbscan.approximate_predict(clusterer, reducer.transform(X_test))
```
  - visualize a dataframe consisting of the transformed test coords and the labels

- given the predicted cluster as a target, use rfc etc., to train a model to predict cluster (features = UMAP reduced dims, or raw feature values)

- transform the holdout set and predict clusters
- visualize to assess performance

