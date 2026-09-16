# %%
import time
t = time.time()

from typing_extensions import ParamSpecArgs
import numpy as np
import pandas as pd
import scanpy as sc

import sys
import argparse
import cProfile
import pstats

timings = []   # instrumented profile: (code section, seconds)

def record(section, start):
    timings.append((section, time.time() - start))
    print(f"TIME {section}: {timings[-1][1]:.3f} s")

record("imports", t)

# %%
t = time.time()
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
# sc.settings.n_jobs = int(sys.argv[4])
sc.settings.n_jobs = 1

print(f"using {sc.settings.n_jobs} threads")
record("settings", t)

# %%
t = time.time()
parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('--data-dir', type=str, help='Directory containing the dataset subdirectories', default='data')
parser.add_argument('--data-set', type=str, help='Dataset name, which is the subdirectory name', default='pbmc3k')
parser.add_argument('--out-dir', type=str, help='Output directory', required=False, default='data')
parser.add_argument('--num-threads', type=int, help='Number of threads', default=1, required=False)

args = parser.parse_args()

datadir = args.data_dir if args.data_dir.endswith('/') else args.data_dir + '/'
dataset = args.data_set
outdir = args.out_dir if args.out_dir.endswith('/') else args.out_dir + '/'
nthreads = args.num_threads
record("arguments", t)


#%%
t = time.time()

# I/O
results_file = "/".join([outdir, dataset + '.scanpy.h5ad'])  # the file that will store the analysis results

adata = sc.read_10x_mtx(
    #'/nethome/tpan7/scgc/data/' + dataset + '/filtered_gene_bc_matrices/hg19',  # the directory with the `.mtx` file
    "/".join([datadir, dataset, 'filtered_gene_bc_matrices']),  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)                              # write a cache file for faster subsequent reading

adata.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`
record("I/O read", t)


# %%
t = time.time()
# preprocessing

# basic filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
record("filtering", t)

#%%
t = time.time()
# metric
#adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
#sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# filtering by slicing the AnnData object
#adata = adata[adata.obs.n_genes_by_counts < 2500, :]
#adata = adata[adata.obs.pct_counts_mt < 5, :]


# and normalize to 10K reads per cell
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
record("normalize", t)


# %%
t = time.time()
# highly variable genes

#sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

# freeze data.
adata.raw = adata

# filtering by highly variable genes.
adata = adata[:, adata.var.highly_variable]
record("highly variable genes", t)


#%%
t = time.time()
# regres out effects of total counts per cell an d% mitochondrial genes
#sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata)
record("scale", t)

# %%
# report adata - so we can check ot see if we are comparable to Seurat
# adata.write(results_file)
# adata

# %%
t = time.time()
# pca.  parallel via OMP_NUM_THREADS
sc.tl.pca(adata, svd_solver='arpack', n_comps=30)
record("pca", t)

# adata.write(results_file)
# adata

# %%
t = time.time()
# neighborhood graph
sc.pp.neighbors(adata, n_pcs=30)
record("neighbors", t)

# %%
# for fixing disconnected clusters or connectivity issues:
#sc.tl.paga(adata)
#sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
#cs.tl.umap(adata, init_pos='paga')


# adata.write(results_file)
# adata


# %%
t = time.time()
# clustering  (currently uses leiden,  previously using louvain (like Seurat).)
#sc.tl.leiden(adata)
sc.tl.louvain(adata, resolution = 0.5)
record("louvain", t)


#%%
t = time.time()
# umap
sc.tl.umap(adata, n_components=30)
record("umap", t)

#%%
t = time.time()
adata.write(results_file)
adata
record("I/O write", t)

# %%
t = time.time()
# support t-test, wilcoxon, logistic regression
# find marker genes  (fine-grain profiling with cProfile)
prof_file = outdir + dataset + '.prof'
cProfile.run("sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon', use_raw=True)", prof_file)
record("rank_genes_groups", t)

pstats.Stats(prof_file).strip_dirs().sort_stats('cumulative').print_stats(20)

# %%
# instrumented profile summary
df = pd.DataFrame(timings, columns=['section', 'seconds'])
df['n_cells'] = adata.n_obs
df.to_csv(outdir + dataset + '_timings.csv', index=False)

print(df[['section', 'seconds']].to_string(index=False))
print(f"TOTAL: {df['seconds'].sum():.3f} s")
top = df.loc[df['seconds'].idxmax()]
print(f"MAJOR BOTTLENECK: {top['section']} ({top['seconds']:.3f} s)")