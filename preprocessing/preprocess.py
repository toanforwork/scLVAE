import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse as sp

def load_real_data(file_path, min_cells=1, round_input=False):
    df = pd.read_csv(file_path, index_col=0)
    X_raw = df.values
    print(f"[INFO] Loaded data: {X_raw.shape[0]} × {X_raw.shape[1]} (before filtering)")

    if X_raw.shape[0] < X_raw.shape[1]:
        print("[INFO] Detected genes as columns → assuming cells × genes format.")
        X = X_raw
    else:
        print("[INFO] Detected genes as rows → transposing to cells × genes.")
        X = X_raw.T
        df = pd.DataFrame(X, index=None)

    if round_input:
        print("[INFO] Rounding input to nearest integer (for raw counts).")
        X = np.round(X).clip(min=0)

    adata = sc.AnnData(X)
    gene_counts = np.sum(adata.X > 0, axis=0)
    genes_to_keep = gene_counts >= min_cells
    X_filtered = X[:, genes_to_keep].astype(np.float32)
    print(f"[INFO] Filtered genes: {np.sum(genes_to_keep)} kept out of {len(genes_to_keep)}")

    df_filtered = pd.DataFrame(
        X_filtered,
        index=[f"cell{i+1}" for i in range(X_filtered.shape[0])],
        columns=[f"gene{j+1}" for j in range(X_filtered.shape[1])]
    )

    return X_filtered, df_filtered


def load_and_filter_data():
    counts_file = "data/sim.data/sim.dropout/SplatDrop_counts.csv"
    df3 = pd.read_csv(counts_file)
    Xmiss_original = df3.iloc[:, 1:].values.T

    truecounts_file = "data/sim.data/sim.dropout/SplatDrop_TrueCounts.csv"
    df1 = pd.read_csv(truecounts_file)
    X_original = df1.iloc[:, 1:].values.T
    
    adata = sc.AnnData(Xmiss_original)
    gene_counts = np.sum(adata.X > 0, axis=0)
    genes_to_keep_mask = gene_counts >= 1

    Xmiss = Xmiss_original[:, genes_to_keep_mask].astype('float32')
    X = X_original[:, genes_to_keep_mask]

    return X, Xmiss

def normalize_and_log_single(X, do_normalize=True, do_log=True, target_sum=1e4, copy=True):
    adata = ad.AnnData(X.copy() if copy else X)

    if do_normalize:
        sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=False)

    if do_log:
        sc.pp.log1p(adata)

    X_processed = adata.X
    X_processed = X_processed.astype(np.float32) if sp.issparse(X_processed) else np.asarray(X_processed, dtype=np.float32)

    return X_processed

def normalize_and_log(X, X_imp, do_normalize=True, do_log=True, target_sum=1e4, copy=True):
    if sp.issparse(X):
        total_counts_X = np.asarray(X.sum(axis=1)).ravel()
    else:
        total_counts_X = X.sum(axis=1).ravel()

    safe_total = total_counts_X.copy()
    safe_total[safe_total == 0] = 1.0


    adata = ad.AnnData(X.copy() if copy else X)
    if do_normalize:
        sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=False)
    if do_log:
        sc.pp.log1p(adata)
    X_norm = adata.X

    if do_normalize:
        factors = safe_total / float(target_sum)
        if sp.issparse(X_imp):
            X_imp_scaled = X_imp.multiply(1.0 / factors[:, None])
        else:
            X_imp_scaled = X_imp / factors[:, None]
    else:
        X_imp_scaled = X_imp.copy()

    if do_log:
        if sp.issparse(X_imp_scaled):
            coo = X_imp_scaled.tocoo()
            coo.data = np.log1p(coo.data)
            X_imp_scaled = coo.tocsr()
        else:
            X_imp_scaled = np.log1p(X_imp_scaled)

    X_norm = X_norm.astype(np.float32) if sp.issparse(X_norm) else np.asarray(X_norm, dtype=np.float32)
    X_imp_scaled = X_imp_scaled.astype(np.float32) if sp.issparse(X_imp_scaled) else np.asarray(X_imp_scaled, dtype=np.float32)

    return X_norm, X_imp_scaled