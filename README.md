# DeepPNCC: Deep learning framework for pseudo-spatial cell-cell interaction landscapes containing spatial information
DeepPNCCS is a deep learning model developed based on the variational graph autoencoder combined with adversarial strategies. It can identify spatially informative pseudo-spatial cell-cell interactions from single-cell sequencing data. The DeepPNCC model ingeniously utilizes the local spatial adjacency information provided by undissociated cells in single-cell sequencing data, effectively incorporating spatial information of cells into the identification of cell-cell interactions using only single-cell sequencing data.


## Version

1.0.0

## Authors

Xuhua Li 

## Getting Started

### Dependencies and requirements

DeepPNCC depends on the following packages: numpy, pandas, scipy, matplotlib, seaborn, networkx, scikit-learn, umap-learn, tensorflow. See dependency versions in `requirements.txt`. The package has been tested on Anaconda3-4.2.0 and is platform independent (tested on Windows and Linux) and should work in any valid python environment. To speed up the training process, DeepPNCC relies on Graphic Processing Unit (GPU). If no GPU device is available, the CPU will be used for model training. No special hardware is required besides these. Installation of the dependencies may take several minutes.

```
pip install --requirement requirements.txt
```

### Usage
Assume we have (1) a CSV-formatted raw count matrix ``counts.csv`` with cells in rows and genes in columns (2) an adjacent matrix in ``adj.csv`` as a predefined local interaction map.

You can run a demo from the command line:

``python DeepPNCC.py -e ./dataset/CID44971/CID44971-cellsplit.csv -a ./dataset/CID44971/CID44971-nearmatrix.csv  -mdl ./dataset/CID44971/256-003/module -opt ./dataset/CID44971/256-003/output -add 1 -i 100``

Note: The parameter -add is an additional value to be added when the input expression spectrum is log-transformed i.e., performs a log10 (x+log_add_number) transformation '

### Results

The final output reports the AUPRC performance, the reconstructed cell adjacency matrix, the over- or under-representation of interaction between cell groups, the latent feature for each cell and the saved model.



