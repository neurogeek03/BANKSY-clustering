# basic paths 
project_dir = '/scratch/mfafouti/BANKSY'

# imports 
import os, re
import numpy as np
import pandas as pd
from IPython.display import display
import warnings
warnings.filterwarnings("ignore") 

import scipy.sparse as sparse
from scipy.io import mmread
from scipy.stats import pearsonr, pointbiserialr

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import seaborn as sns
import scanpy as sc
sc.logging.print_header()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 1 # errors (0), warnings (1), info (2), hints (3)
plt.rcParams["font.family"] = "Arial"
sns.set_style("white")

import random
# Note that BANKSY itself is deterministic, here the seeds affect the umap clusters and leiden partition
seed = 1234
np.random.seed(seed)
random.seed(seed)

# Additional imports 
import sys
print(sys.executable) #checking that I am using the right env 

data_path = os.path.join(project_dir, 'Banksy_py')
sys.path.append(data_path) # add parent dir to banksy utils
# Modifying the spatial info to match required format
from banksy_utils.load_data import load_adata, display_adata

# Define File paths
file_path = os.path.join(project_dir, "Banksy_py", "data", "query")
adata_filename = "B01_anndata.h5ad"

adata = sc.read_h5ad(os.path.join(project_dir, file_path, adata_filename))

print(adata.obsm.keys())
print(adata.obsm["X_spatial"][:5])

adata.obs["xcoord"] = adata.obsm["X_spatial"][:, 0]
adata.obs["ycoord"] = adata.obsm["X_spatial"][:, 1]
adata.obs["coord_xy"] = list(zip(adata.obs["xcoord"], adata.obs["ycoord"]))

# Renaming the var part
adata.var_names = adata.var_names.str.split("-").str[-1]
print(adata.var_names[:10])
is_unique = adata.var_names.is_unique
print(is_unique) # excellent

display_adata(adata)

print(adata.var.keys())

display(adata.obs, adata.var)

print(adata.var_names[:10])  # Shows updated gene names (index)
print(adata.var.index[:10])  # Same as above, index is renamed

adata.obs['total_counts'] = adata.obs['nCount_RNA']
adata.obs['n_genes_by_counts'] = adata.obs['nFeature_RNA']

from banksy_utils.plot_utils import plot_qc_hist, plot_cell_positions

# bin options for fomratting histograms
# Here, we set 'auto' for 1st figure, 80 bins for 2nd figure. and so on
hist_bin_options = ['auto', 80, 80, 100]

plot_qc_hist(adata, 
         total_counts_cutoff=150, 
         n_genes_high_cutoff=2500, 
         n_genes_low_cutoff=800,
         bin_options = hist_bin_options)

from banksy_utils.filter_utils import filter_cells

# rename mt column
adata.obs['pct_counts_mt'] = adata.obs['percent.mt']

# Filter cells with each respective filters
adata = filter_cells(adata, 
             min_count=50, 
             max_count=2500, 
             MT_filter=20, 
             gene_filter=10)

display_adata(adata)

hist_bin_options = ['auto', 100, 60, 100]

plot_qc_hist(adata,
        total_counts_cutoff=2000,
        n_genes_high_cutoff=2000, 
        n_genes_low_cutoff= 0,
        bin_options = hist_bin_options)

from banksy_utils.filter_utils import normalize_total, filter_hvg, print_max_min

import anndata
# Normalizes the anndata dataset
adata = normalize_total(adata)

adata, adata_allgenes = filter_hvg(
    adata,
    n_top_genes=2000,
    flavor="seurat"
)

display_adata(adata)

from banksy.main import median_dist_to_nearest_neighbour
coord_keys = ('xcoord', 'ycoord', 'coord_xy')
coord_keys = list(coord_keys)
coord_keys[2] = 'X_spatial'
# set params
# ==========
plot_graph_weights = True
k_geom = 15 # number of spatial neighbours
max_m = 1 # use both mean and AFT
nbr_weight_decay = "scaled_gaussian" # can also choose "reciprocal", "uniform" or "ranked"

print(adata.obsm.keys())

# Find median distance to closest neighbours
nbrs = median_dist_to_nearest_neighbour(adata, key='X_spatial')

from banksy.initialize_banksy import initialize_banksy
import pickle

banksy_dict = initialize_banksy(
    adata,
    coord_keys,
    k_geom,
    nbr_weight_decay=nbr_weight_decay,
    max_m=max_m,
    plt_edge_hist=True,
    plt_nbr_weights=True,
    plt_agf_angles=False, # takes long time to plot
    plt_theta=True,
)

with open(os.path.join(project_dir, "banksy_dict.pkl"), "wb") as f:
    pickle.dump(banksy_dict, f)

from banksy.embed_banksy import generate_banksy_matrix

# The following are the main hyperparameters for BANKSY
# -----------------------------------------------------
resolutions = [0.7]  # clustering resolution for UMAP
pca_dims = [20]  # Dimensionality in which PCA reduces to
lambda_list = [0.8]  # list of lambda parameters

banksy_dict, banksy_matrix = generate_banksy_matrix(adata,
                                                    banksy_dict,
                                                    lambda_list,
                                                    max_m)

with open(os.path.join(project_dir, "banksy_matrix.pkl"), "wb") as f:
    pickle.dump(banksy_matrix, f)

from banksy.main import concatenate_all

banksy_dict["nonspatial"] = {
    # Here we simply append the nonspatial matrix (adata.X) to obtain the nonspatial clustering results
    0.0: {"adata": concatenate_all([adata.X], 0, adata=adata), }
}

print(banksy_dict['nonspatial'][0.0]['adata'])

from banksy_utils.umap_pca import pca_umap

pca_umap(banksy_dict,
         pca_dims = pca_dims,
         add_umap = True,
         plt_remaining_var = False,
         )

from banksy.cluster_methods import run_Leiden_partition

results_df, max_num_labels = run_Leiden_partition(
    banksy_dict,
    resolutions,
    num_nn = 50, # k_expr: number of neighbours in expression (BANKSY embedding or non-spatial) space
    num_iterations = -1, # run to convergenece
    partition_seed = seed,
    match_labels = True,
)

with open(os.path.join(project_dir, "results_df.pkl"), "wb") as f:
    pickle.dump(results_df, f)
with open(os.path.join(project_dir, "max_num_labels.pkl"), "wb") as f:
    pickle.dump(max_num_labels, f)