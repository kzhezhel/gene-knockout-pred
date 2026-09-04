import scanpy as sc
import pandas as pd
import numpy as np
import squidpy as sq
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph
import gseapy as gp
import json
from PIL import Image
from celcomen.models.celcomen import celcomen
from celcomen.models.simcomen import simcomen
from celcomen.utils.helpers import normalize_g2g, calc_sphex
from torch_geometric.data import Data, DataLoader
from scipy.spatial import cKDTree
import argparse
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description="Spatially-aware SOX2 knockout prediction using Celcomen/Simcomen."
)
parser.add_argument(
    "--h5", type=str, required=True,
    help="Path to the 10x Visium filtered_feature_bc_matrix.h5 file."
)
parser.add_argument(
    "--spatial_root", type=str, default=None,
    help="Path to the 'spatial' folder (if not automatically detected)."
)
parser.add_argument(
    "--targets", type=str, default="sox2_targets.json",
    help="Path to the SOX2 target genes JSON file (default: sox2_targets.json)."
)
parser.add_argument(
    "--outdir", type=str, default="sox2_knockout_results",
    help="Output directory (default: sox2_knockout_results)."
)
parser.add_argument(
    "--cells", type=int, nargs="+", default=[909, 304],
    help="List of cell indices to analyse (default: 909 304). If none are suitable, top 4 are auto-selected."
)
parser.add_argument(
    "--device", type=str, default=None,
    help="Device to use: 'cuda' or 'cpu'. Auto-detected if not specified."
)
args = parser.parse_args()

# Device setup

if args.device is not None:
    device = torch.device(args.device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths and checks

h5_path = args.h5
if not os.path.isfile(h5_path):
    raise FileNotFoundError(f"H5 file not found: {h5_path}")

if args.spatial_root is None:
    # Try to find spatial folder next to the h5 file
    base_dir = os.path.dirname(h5_path)
    possible_spatial = os.path.join(base_dir, "spatial", "spatial")
    if not os.path.isdir(possible_spatial):
        possible_spatial = os.path.join(base_dir, "spatial")
    if not os.path.isdir(possible_spatial):
        raise FileNotFoundError("Could not locate 'spatial' folder. Please specify --spatial_root.")
    spatial_root = possible_spatial
else:
    spatial_root = args.spatial_root
    if not os.path.isdir(spatial_root):
        raise FileNotFoundError(f"Spatial root not found: {spatial_root}")

targets_file = args.targets
if not os.path.isfile(targets_file):
    raise FileNotFoundError(f"Targets file not found: {targets_file}")

out_dir = args.outdir
os.makedirs(out_dir, exist_ok=True)

# Calculation of spherical coordinates (sphex)

def stable_calc_sphex(gex, eps=1e-7):
    """Compute spherical coordinates from expression matrix."""
    n_cells, n_genes = gex.shape
    sphex = torch.zeros((n_cells, n_genes-1), dtype=gex.dtype, device=gex.device)
    g = gex.clone()
    for i in range(n_genes-1):
        r = torch.sqrt(torch.sum(g[:, i:]**2, dim=1))
        r = torch.clamp(r, min=eps)
        cos_theta = g[:, i] / r
        cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
        theta = torch.acos(cos_theta)
        sphex[:, i] = theta
        sin_theta = torch.sin(theta)
        sin_theta = torch.clamp(sin_theta, min=eps)
        g[:, i:] = g[:, i:] / sin_theta.unsqueeze(1)
    return sphex

# Load raw data

print("\nLoading Visium data...")
adata_raw = sc.read_10x_h5(h5_path, gex_only=False)
adata_raw.var_names_make_unique()

# Load spatial metadata and image
scalefactors_path = os.path.join(spatial_root, "scalefactors_json.json")
with open(scalefactors_path, 'r') as f:
    scalefactors = json.load(f)

img_path = os.path.join(spatial_root, "tissue_hires_image.png")
if not os.path.exists(img_path):
    img_path = os.path.join(spatial_root, "tissue_lowres_image.png")  # fallback
img = Image.open(img_path)

adata_raw.uns['spatial'] = {
    "V1": {
        "images": {"hires": np.array(img), "lowres": np.array(img)},
        "scalefactors": {
            "tissue_hires_scalef": scalefactors.get('tissue_hires_scalef', 1.0),
            "spot_diameter_fullres": scalefactors.get('spot_diameter_fullres', 1.0),
        },
        "metadata": {"chemistry_description": "Visium FFPE"}
    }
}

# Load tissue positions
tissue_pos = pd.read_csv(os.path.join(spatial_root, "tissue_positions.csv"), header=0)
tissue_pos.set_index('barcode', inplace=True)
tissue_pos.index = tissue_pos.index.str.replace(r'-1$', '', regex=True)
adata_raw.obs_names = adata_raw.obs_names.str.replace(r'-1$', '', regex=True)
adata_raw.obs['in_tissue'] = tissue_pos.loc[adata_raw.obs_names, 'in_tissue']
adata_raw.obs['pxl_row_in_fullres'] = tissue_pos.loc[adata_raw.obs_names, 'pxl_row_in_fullres']
adata_raw.obs['pxl_col_in_fullres'] = tissue_pos.loc[adata_raw.obs_names, 'pxl_col_in_fullres']

# Keep only tissue spots
adata_raw = adata_raw[adata_raw.obs['in_tissue'] == 1].copy()
sc.pp.filter_cells(adata_raw, min_counts=100)
coords_all = adata_raw.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].values
adata_raw.obsm['spatial'] = coords_all

# Normalised version for visualisation
adata_norm = adata_raw.copy()
sc.pp.normalize_total(adata_norm, target_sum=1e6)
sc.pp.log1p(adata_norm)

print(f"Loaded {adata_raw.n_obs} spots, {adata_raw.n_vars} genes")

# Load SOX2 target genes

print("\nLoading SOX2 target genes...")
with open(targets_file, 'r') as f:
    data = json.load(f)
sox2_all_targets = [assoc['gene']['symbol'] for assoc in data['associations']]
if 'SOX2' not in sox2_all_targets:
    sox2_all_targets.append('SOX2')
sox2_targets_present = [g for g in sox2_all_targets if g in adata_raw.var_names]
print(f"SOX2 targets present in data: {len(sox2_targets_present)}")

# Parameters

n_neighbors = 13          # patch size (including target cell)
k_graph = 6
epochs_cel = 200
epochs_sim = 100
zmft_cel = 0.1
zmft_sim = 2.2
lambda_reg = 1000.0
lr_cel = 1e-1
lr_sim = 6e-5
min_patch_size = 13
initial_radius = 200
max_radius = 1500
step = 100

# Identify candidate cells for knockout (high SOX2 expression)

gene_idx = np.where(adata_raw.var_names == 'SOX2')[0][0]
expr_target = adata_raw.X[:, gene_idx].toarray().flatten()

threshold = np.percentile(expr_target, 95)
high_expr_cells = np.where(expr_target > threshold)[0]
print(f"Cells with SOX2 > {threshold:.2f}: {len(high_expr_cells)}")

# Find cells with enough neighbours
tree = cKDTree(adata_raw.obsm['spatial'])
candidates = []  # (global_index, radius, patch_indices)

for cell in high_expr_cells:
    found = False
    for radius in range(initial_radius, max_radius + 1, step):
        neighbors = tree.query_ball_point(adata_raw.obsm['spatial'][cell], r=radius)
        if len(neighbors) >= min_patch_size:
            candidates.append((cell, radius, neighbors))
            found = True
            break
    if not found:
        dist, idx = tree.query(adata_raw.obsm['spatial'][cell], k=min_patch_size)
        candidates.append((cell, None, idx.tolist()))
        print(f"  Cell {cell}: failed to get {min_patch_size} neighbours, using nearest.")
    else:
        print(f"  Cell {cell}: radius {radius}, patch size {len(neighbors)}")

# Select cells to analyse
cells_to_analyze = []
for c in args.cells:
    if any(c == cand[0] for cand in candidates):
        cells_to_analyze.append(c)
    else:
        print(f"Cell {c} not suitable (low expression or insufficient neighbours), skipping.")

if len(cells_to_analyze) == 0:
    print("No suitable cells from manual list; taking top 4 by expression.")
    cells_to_analyze = [c[0] for c in sorted(candidates, key=lambda x: expr_target[x[0]], reverse=True)[:4]]

print(f"\nSelected {len(cells_to_analyze)} cells for analysis: {cells_to_analyze}")
print("SOX2 expression in selected cells:")
for c in cells_to_analyze:
    print(f"  Cell {c}: {expr_target[c]:.2f}")

def get_patch_for_cell(cell_global, coords_all, n_neighbors=min_patch_size):
    tree = cKDTree(coords_all)
    dist, idx = tree.query(coords_all[cell_global], k=n_neighbors)
    return idx.tolist(), None
  
print("\nCollecting cells from all patches for global gene set...")
all_patch_global_indices = set()
for cell_global in cells_to_analyze:
    neigh, _ = get_patch_for_cell(cell_global, adata_raw.obsm['spatial'], n_neighbors=min_patch_size)
    all_patch_global_indices.update(neigh)
all_patch_global_indices = list(all_patch_global_indices)
print(f"Total unique cells across {len(cells_to_analyze)} patches: {len(all_patch_global_indices)}")

adata_combined = adata_raw[all_patch_global_indices, :].copy()
if hasattr(adata_combined.X, 'toarray'):
    mean_counts_comb = adata_combined.X.mean(axis=0).A1
else:
    mean_counts_comb = adata_combined.X.mean(axis=0)
mean_series_comb = pd.Series(mean_counts_comb, index=adata_combined.var_names)

sox2_mean_comb = mean_series_comb[sox2_targets_present].sort_values(ascending=False)
n_targets_comb = min(1000, len(sox2_mean_comb))
top_targets_comb = sox2_mean_comb.head(n_targets_comb).index.tolist()
if 'SOX2' not in top_targets_comb:
    top_targets_comb.append('SOX2')
final_genes = list(set(top_targets_comb))
print(f"Using fixed gene set of {len(final_genes)} genes (top SOX2 targets in combined patches)")

for i_cell, cell_global in enumerate(cells_to_analyze):
    print(f"\n========== Analysing cell {cell_global} ({i_cell+1}/{len(cells_to_analyze)}) ==========")
    cell_dir = os.path.join(out_dir, f"cell_{cell_global}")
    os.makedirs(cell_dir, exist_ok=True)

    # Build patch (nearest neighbours)
    neighbor_indices, _ = get_patch_for_cell(cell_global, adata_raw.obsm['spatial'], n_neighbors=min_patch_size)
    neighbor_indices = sorted(neighbor_indices)
    coords_test = adata_raw[neighbor_indices].obsm['spatial']
    print(f"  Patch coordinates: x {coords_test[:, 0].min():.1f}–{coords_test[:, 0].max():.1f}, "
          f"y {coords_test[:, 1].min():.1f}–{coords_test[:, 1].max():.1f}")
    cell_pos = neighbor_indices.index(cell_global)
    print(f"  Patch size: {len(neighbor_indices)} cells (nearest neighbours)")

    adata_patch_raw = adata_raw[neighbor_indices, :].copy()
    adata_patch = adata_patch_raw[:, final_genes].copy()
    sox2_local_idx = np.where(adata_patch.var_names == 'SOX2')[0][0]

    adata_norm_patch = adata_norm[neighbor_indices, :][:, final_genes].copy()
    adata_norm_patch.obsm['spatial'] = adata_patch.obsm['spatial'].copy()
    adata_norm_patch.uns['spatial'] = adata_raw.uns['spatial']

    # Build graph
    k = min(k_graph, adata_patch.n_obs - 1)
    if k < 1:
        print("  Too few cells, skipping.")
        continue
    coords = torch.from_numpy(adata_patch.obsm['spatial'])
    edges_np = kneighbors_graph(coords.numpy(), k, include_self=False).todense()
    edges = torch.from_numpy(np.array(np.where(edges_np))).to(torch.long)

    # Visualise patch
    try:
        coords_patch = adata_raw[neighbor_indices].obsm['spatial']
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(coords_patch[:, 1], coords_patch[:, 0], c='blue', s=30, alpha=0.7)
        ax.scatter(coords_patch[cell_pos, 1], coords_patch[cell_pos, 0],
                   c='red', s=100, edgecolors='black', linewidth=2, label='KO cell')
        ax.set_title(f'Patch for cell {cell_global} ({len(coords_patch)} cells)')
        ax.set_xlabel('x coordinate (pixels)')
        ax.set_ylabel('y coordinate (pixels)')
        ax.legend()
        plt.savefig(os.path.join(cell_dir, "patch_before_analysis.png"), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  Warning: could not plot patch ({e})")

    # Train Celcomen
    n_genes = len(final_genes)
    model_cel = celcomen(input_dim=n_genes, output_dim=n_genes, n_neighbors=k, seed=42)
    input_g2g = np.random.uniform(size=(n_genes, n_genes)).astype('float32')
    input_g2g = normalize_g2g((input_g2g + input_g2g.T) / 2)
    model_cel.set_g2g(torch.from_numpy(input_g2g))
    model_cel.set_g2g_intra(torch.from_numpy(input_g2g))
    model_cel.to(device)

    x_cel = torch.tensor(adata_patch.X.toarray(), dtype=torch.float32)
    pyg_data = Data(x=x_cel, edge_index=edges)
    loader = DataLoader([pyg_data], batch_size=1, shuffle=False)
    optimizer_cel = torch.optim.SGD(model_cel.parameters(), lr=lr_cel)

    for epoch in tqdm(range(epochs_cel), desc=f"Celcomen (cell {cell_global})", leave=False):
        for batch in loader:
            batch = batch.to(device)
            model_cel.gex = batch.x
            msg, msg_intra, log_z_mft = model_cel(batch.edge_index, batch=None)
            loss = -(-log_z_mft + zmft_cel * torch.trace(torch.mm(msg, torch.t(model_cel.gex))) +
                     zmft_cel * torch.trace(torch.mm(msg_intra, torch.t(model_cel.gex))))
            optimizer_cel.zero_grad()
            loss.backward()
            optimizer_cel.step()

    # Knockout SOX2 in target cell
    cells_ko = np.array([cell_pos], dtype=np.int64)
    adata_pert = adata_patch.copy()
    adata_pert.X[cells_ko, sox2_local_idx] = 0.0

    # Prepare for Simcomen
    expr_pert = torch.from_numpy(adata_pert.X.toarray()).float()
    norm_factor = torch.sqrt(torch.pow(expr_pert, 2).sum(1)).reshape(-1, 1)
    expr_norm = torch.div(expr_pert, norm_factor).to(device)
    sox2_idx = adata_patch.var_names.get_loc('SOX2')

    model_sim = simcomen(input_dim=n_genes, output_dim=n_genes, n_neighbors=k, seed=42)
    with torch.no_grad():
        g2g_w = model_cel.conv1.lin.weight.clone().detach()
        g2g_w_norm = g2g_w / (torch.norm(g2g_w, dim=1, keepdim=True) + 1e-8)
        model_sim.set_g2g(g2g_w_norm)
        g2g_intra_w = model_cel.lin.weight.clone().detach()
        g2g_intra_norm = g2g_intra_w / (torch.norm(g2g_intra_w, dim=1, keepdim=True) + 1e-8)
        model_sim.set_g2g_intra(g2g_intra_norm)

    expr_ctrl = torch.from_numpy(adata_patch.X.toarray()).float()
    norm_ctrl = torch.sqrt(torch.pow(expr_ctrl, 2).sum(1)).reshape(-1, 1)
    expr_ctrl_norm = torch.div(expr_ctrl, norm_ctrl).to(device)
    expr_ctrl_clamped = torch.clamp(expr_ctrl_norm, -0.999999, 0.999999)
    init_sphex = stable_calc_sphex(expr_ctrl_clamped).cpu().numpy()
    if np.isnan(init_sphex).any():
        init_sphex = np.random.randn(adata_patch.n_obs, n_genes-1).astype('float32') * 0.01
    model_sim.set_sphex(torch.from_numpy(init_sphex.astype('float32')))
    model_sim.to(device)

    sox2_original = expr_norm[:, sox2_idx].detach().clone()
    mask_ko = torch.zeros(adata_patch.n_obs, 1, device=device)
    mask_ko[cells_ko] = 1.0
    optimizer_sim = torch.optim.SGD(model_sim.parameters(), lr=lr_sim, momentum=0)
    print(f"  Mean SOX2 expression in patch (raw counts): {adata_patch.X[:, sox2_local_idx].mean():.2f}")

    # Train Simcomen
    model_sim.train()
    for epoch in tqdm(range(epochs_sim), desc=f"Simcomen (cell {cell_global})", leave=False):
        msg, msg_intra, log_z_mft = model_sim(edges.to(device), 1)
        if epoch == 0:
            orig_expr = model_sim.gex.clone().detach().cpu().numpy()
        loss = -(-log_z_mft + zmft_sim * torch.trace(torch.mm(msg, torch.t(model_sim.gex))) +
                 zmft_sim * torch.trace(torch.mm(msg_intra, torch.t(model_sim.gex))))
        sox2_cur = model_sim.gex[:, sox2_idx]
        reg_sox2 = torch.mean(torch.abs((1 - mask_ko) * (sox2_cur - sox2_original)))
        total_loss = loss + lambda_reg * reg_sox2
        optimizer_sim.zero_grad()
        total_loss.backward()
        optimizer_sim.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                control_mask = (mask_ko.squeeze() == 0)
                sox2_control_mean = sox2_cur[control_mask].mean().item() if control_mask.any() else float('nan')
                sox2_ko_mean = sox2_cur[cells_ko].mean().item()
            tqdm.write(f"    Epoch {epoch+1}: loss={loss.item():.4f}, SOX2_ctrl={sox2_control_mean:.6f}, SOX2_ko={sox2_ko_mean:.6f}, reg={reg_sox2.item():.6f}")

    model_sim.eval()
    with torch.no_grad():
        _, _, _ = model_sim(edges.to(device), 1)
        pred_expr = model_sim.gex.clone().detach().cpu().numpy()
    expr_diff = pred_expr - orig_expr
  
    neigh_local = [i for i in range(adata_patch.n_obs) if i != cell_pos]
    if len(neigh_local) == 0:
        print("  No neighbours, skipping.")
        continue

    mean_diff_neighbors = expr_diff[neigh_local, :].mean(axis=0)
    mean_diff_neighbors = pd.Series(mean_diff_neighbors, index=adata_patch.var_names)
    mean_diff_z = (mean_diff_neighbors - mean_diff_neighbors.mean()) / mean_diff_neighbors.std()
    down_genes = mean_diff_z[mean_diff_z < -1].index.tolist()
    up_genes = mean_diff_z[mean_diff_z > 1].index.tolist()
    down_genes = [g for g in down_genes if g != 'SOX2']
    up_genes = [g for g in up_genes if g != 'SOX2']
    print(f"  Downregulated: {len(down_genes)}, Upregulated: {len(up_genes)}")
    with open(os.path.join(cell_dir, "down_genes.txt"), 'w') as f:
        f.write("\n".join(down_genes))
    with open(os.path.join(cell_dir, "up_genes.txt"), 'w') as f:
        f.write("\n".join(up_genes))
    np.save(os.path.join(cell_dir, "expr_diff.npy"), expr_diff)

    # GSEA
    def run_gsea(gene_list, label, out_dir):
        if not gene_list:
            print(f"    No {label} genes for GSEA")
            return
        try:
            enr = gp.enrichr(gene_list=gene_list,
                             gene_sets=['GO_Biological_Process_2025'],
                             organism='human',
                             cutoff=0.05)
            if not enr.results.empty:
                top = enr.results.sort_values('Adjusted P-value').head(10)
                top.to_csv(os.path.join(out_dir, f"GSEA_{label}.csv"), index=False)
                print(f"    Top {label} terms:")
                print(top[['Term', 'Adjusted P-value']].to_string(index=False))
                fig, ax = plt.subplots(figsize=(6,4))
                sns.barplot(data=top, y='Term', x=-np.log10(top['Adjusted P-value']),
                            color='salmon' if label=='up' else 'skyblue',
                            edgecolor='red' if label=='up' else 'dodgerblue', ax=ax)
                ax.set_xlabel('-log10(FDR)')
                ax.set_title(f'{label.capitalize()}regulated genes')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"GSEA_{label}.png"), dpi=150)
                plt.close()
            else:
                print(f"    No significant enrichment for {label} genes")
        except Exception as e:
            print(f"    GSEA error ({label}): {e}")

    run_gsea(down_genes, "down", cell_dir)
    run_gsea(up_genes, "up", cell_dir)

    # Spatial visualisation of effect
    try:
        total_change = np.linalg.norm(expr_diff, axis=1)
        adata_norm_patch.obs['total_effect'] = total_change
        adata_norm_patch.obs['is_knockout'] = False
        adata_norm_patch.obs.iloc[cell_pos, adata_norm_patch.obs.columns.get_loc('is_knockout')] = True
        library_id = list(adata_norm_patch.uns['spatial'].keys())[0]

        orig_coords = adata_norm_patch.obsm['spatial'].copy()
        adata_norm_patch.obsm['spatial'] = orig_coords[:, [1, 0]]

        fig, ax = plt.subplots(figsize=(8, 6))
        sq.pl.spatial_scatter(adata_norm_patch, color='total_effect',
                              title=f'SOX2 knockout effect (cell {cell_global})',
                              size=1.5, alpha=0.8, cmap='viridis', img=True,
                              library_id=library_id, img_res_key='hires', ax=ax)
        plt.savefig(os.path.join(cell_dir, "spatial_effect.png"), dpi=150, bbox_inches='tight')
        plt.close()
        adata_norm_patch.obsm['spatial'] = orig_coords
    except Exception as e:
        print(f"  Visualisation error: {e}")

    # Save predicted expressions
    pred_log1p = torch.from_numpy(pred_expr) * norm_factor.cpu()
    adata_pred = adata_patch.copy()
    adata_pred.X = pred_log1p.cpu().numpy().astype(np.float32)
    adata_pred.write(os.path.join(cell_dir, "adata_predicted.h5ad"))

    print(f"  Results for cell {cell_global} saved in {cell_dir}")

print("\n=== Analysis of all selected cells completed successfully ===")
