# LC OPTIMIZE

This repo contains a collection of notebooks for the LC OPTIMIZE project.

## NOTES

The main activity - clustering via dimensionality reduction - is implemented in the `04-clustering-3.ipynb` notebook.

Many of the other notebooks are used for data preparation, exploration, and testing various approaches. Two author contributed to the same repo, for different activities.

## Requirements

See the `env.yaml` file for the list of requirements. 

Use `conda` or `mamba` to create the environment. Note that `mamba` is much faster than `conda`.

Expected installation time should be around 5 minutes, depending on your internet connection and previous installations (if packages are cached, it will be much faster).

This environment is tested on Python 3.11.

No specialized hardware is required. 

## Installation

1. Generate the environment:

   ```bash
   mamba env create -f env.yaml
   ```

2. Activate the environment:

   ```bash
   conda activate lc-optimize
   ```


## Running the notebooks
Open notebook `04-clustering-3.ipynb` in JupyterLab or Jupyter Notebook. Even better, use VS Code with the Jupyter extension.

Runtime should require only a few minutes. The UMAP dimensionality reduction step is the most time-consuming, but it should not take more than 5 minutes on a standard laptop.


## License