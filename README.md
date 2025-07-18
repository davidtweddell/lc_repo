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
This project is licensed under the MIT License. 

Copyright 2025 The LC OPTIMIZE Project Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
