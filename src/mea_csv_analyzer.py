import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# ==========================================
# 0. STTC HELPER FUNCTIONS
# ==========================================
def calculate_run_time(spikes, dt, t_start, t_end):
    """Calculate the proportion of total time within +/- dt of any spike."""
    if len(spikes) == 0:
        return 0.0
    
    windows = np.zeros((len(spikes), 2))
    windows[:, 0] = np.maximum(spikes - dt, t_start)
    windows[:, 1] = np.minimum(spikes + dt, t_end)
    
    merged_windows = []
    current_start, current_end = windows[0]
    
    for i in range(1, len(windows)):
        start, end = windows[i]
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged_windows.append((current_start, current_end))
            current_start = start
            current_end = end
    merged_windows.append((current_start, current_end))
    
    total_time = sum([end - start for start, end in merged_windows])
    return total_time / (t_end - t_start)

def calculate_spikes_in_window(spikes_1, spikes_2, dt):
    """Calculate proportion of spikes_1 that fall within +/- dt of any spike_2."""
    if len(spikes_1) == 0 or len(spikes_2) == 0:
        return 0.0
    
    indices = np.searchsorted(spikes_2, spikes_1)
    left_valid = (indices > 0) & (np.abs(spikes_1 - spikes_2[np.minimum(indices - 1, len(spikes_2)-1)]) <= dt)
    right_valid = (indices < len(spikes_2)) & (np.abs(spikes_1 - spikes_2[np.minimum(indices, len(spikes_2)-1)]) <= dt)
    matches = np.sum(left_valid | right_valid)
    return matches / len(spikes_1)

def compute_sttc(spikes_A, spikes_B, dt, t_start, t_end):
    """Compute the Spike Time Tiling Coefficient (Cutts & Eglen, 2014)."""
    if len(spikes_A) == 0 or len(spikes_B) == 0:
        return 0.0
        
    T_A = calculate_run_time(spikes_A, dt, t_start, t_end)
    T_B = calculate_run_time(spikes_B, dt, t_start, t_end)
    P_A = calculate_spikes_in_window(spikes_A, spikes_B, dt)
    P_B = calculate_spikes_in_window(spikes_B, spikes_A, dt)
    
    term1 = 0.0 if P_A * T_B == 1.0 else (P_A - T_B) / (1.0 - P_A * T_B)
    term2 = 0.0 if P_B * T_A == 1.0 else (P_B - T_A) / (1.0 - P_B * T_A)
    return 0.5 * (term1 + term2)

# ==========================================
# 1. BATCH FILE SELECTION UI
# ==========================================
print("Opening file selector...")
root = tk.Tk()
root.withdraw()

csv_paths = filedialog.askopenfilenames(
    title="Select the CSV files for your conditions",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

if not csv_paths:
    print("Operation canceled.")
    root.destroy()
    exit()

conditions_data = []

for path_str in sorted(csv_paths):
    csv_file = Path(path_str)
    cond_name = csv_file.stem.replace("spikes_", "").replace("all_spikes_consolidated_", "")
    conditions_data.append({
        "name": cond_name,
        "path": csv_file
    })

root.destroy()

output_folder = conditions_data[0]["path"].parent / "Output_Analysis"
output_folder.mkdir(parents=True, exist_ok=True)

print(f"\nFound {len(conditions_data)} recordings to process. Plots will be saved in: {output_folder.name}")

# ==========================================
# 2. RASTER PLOTTING & STTC DATA PREP
# ==========================================
dt = 0.05  # 50 ms synchrony window
datasets = {}
neuron_sets = []

for cond in conditions_data:
    cond_name = cond["name"]
    csv_file = cond["path"]
    
    print(f"\n--- Processing: {cond_name} ---")
    
    df_spikes = pd.read_csv(csv_file)
    if df_spikes.empty:
        print(f"[{cond_name}] CSV is empty. Skipping.")
        continue

    unit_labels = []
    unit_spike_times = []
    spike_trains = {}

    for unit_id, group in df_spikes.groupby("Neuron_ID"):
        unit_labels.append(unit_id)
        spikes = group["Spike_Time_Seconds"].values
        unit_spike_times.append(spikes)
        spike_trains[unit_id] = np.sort(np.unique(spikes))

    datasets[cond_name] = {
        'spike_trains': spike_trains,
        't_start': df_spikes["Spike_Time_Seconds"].min(),
        't_end': df_spikes["Spike_Time_Seconds"].max()
    }
    neuron_sets.append(set(unit_labels))

    print(f"[{cond_name}] Found {len(unit_labels)} unique units. Generating Raster Plots...")

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.colormaps.get_cmap('tab20')

    ax.eventplot(
        unit_spike_times,
        colors=[colors(i % 20) for i in range(len(unit_spike_times))],
        linewidths=1.0,
        alpha=0.8
    )

    ax.set_yticks(range(len(unit_labels)))
    ax.set_yticklabels(unit_labels, fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Sorted Units", fontsize=12)
    ax.set_title(f"Sorted Units Raster Plot - {cond_name}", fontsize=14)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    full_plot_path = output_folder / f"{cond_name}_raster_full.jpg"
    plt.savefig(full_plot_path, format='jpg', dpi=150)

    ax.set_xlim([0, min(30, datasets[cond_name]['t_end'])])
    zoom_plot_path = output_folder / f"{cond_name}_raster_30s.jpg"
    plt.savefig(zoom_plot_path, format='jpg', dpi=150)
    
    plt.close(fig) 

# ==========================================
# 3. INTERSECT COMMON NEURONS
# ==========================================
if not neuron_sets:
    print("\n[ERROR] No valid datasets loaded.")
    exit()

common_neurons = list(set.intersection(*neuron_sets))
common_neurons.sort() 
print(f"\n=== Network Analysis: {len(common_neurons)} common neurons across all conditions ===")

# ==========================================
# 4. COMPUTE RATE-CORRECTED STTC MATRICES
# ==========================================
sttc_matrices = {}
n_neurons = len(common_neurons)

for cond_name, data in datasets.items():
    print(f"Calculating STTC matrix for {cond_name}...")
    matrix = np.zeros((n_neurons, n_neurons))
    
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i == j:
                matrix[i, j] = 1.0 
            else:
                spikes_A = data['spike_trains'][common_neurons[i]]
                spikes_B = data['spike_trains'][common_neurons[j]]
                matrix[i, j] = compute_sttc(spikes_A, spikes_B, dt, data['t_start'], data['t_end'])
                
    sttc_matrices[cond_name] = pd.DataFrame(matrix, index=common_neurons, columns=common_neurons)

# ==========================================
# 5. DYNAMIC PLOTTING (RAW STTC)
# ==========================================
n_conds = len(sttc_matrices)
fig, axes = plt.subplots(1, n_conds, figsize=(7 * n_conds, 6))

if n_conds == 1:
    axes = [axes]

for ax, (cond_name, matrix) in zip(axes, sttc_matrices.items()):
    sns.heatmap(
        matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0, 
        xticklabels=True, yticklabels=True, ax=ax, square=True, 
        cbar_kws={"shrink": .8}
    )
    ax.set_title(f'{cond_name}\nRaw Connectivity ($\\Delta t$ = {dt}s)')
    ax.set_xlabel('Neuron ID')
    ax.set_ylabel('Neuron ID')
    ax.tick_params(axis='both', labelsize=8)

plt.tight_layout()
plt.savefig(output_folder / "STTC_Raw_Comparison.jpg", format='jpg', dpi=150)
plt.close(fig)

# ==========================================
# 6. PROPORTIONAL THRESHOLDING
# ==========================================
thresholded_matrices = {}
binarized_matrices = {}
target_density = 10  # Top 10% strongest connections

print(f"\n--- Applying {target_density}% Proportional Thresholding ---")

for cond_name, matrix in sttc_matrices.items():
    np_mat = matrix.values
    off_diagonal_vals = np_mat[~np.eye(np_mat.shape[0], dtype=bool)]
    
    if len(off_diagonal_vals) == 0:
        continue
    
    percentile_cutoff = 100 - target_density
    threshold = np.percentile(off_diagonal_vals, percentile_cutoff)
    print(f" -> {cond_name}: Keeping STTC >= {threshold:.4f}")
    
    # 1. Thresholded Matrix (Maintains exact STTC values but zeros out weak ones)
    thresh_mat = matrix.copy()
    thresh_mat[thresh_mat < threshold] = 0.0
    np.fill_diagonal(thresh_mat.values, 0.0) 
    thresholded_matrices[cond_name] = thresh_mat
    
    # 2. Binarized Matrix (1 if connected, 0 if not - Required for NetworkX)
    bin_mat = (matrix >= threshold).astype(int)
    np.fill_diagonal(bin_mat.values, 0)
    binarized_matrices[cond_name] = bin_mat

# ==========================================
# 7. DYNAMIC PLOTTING (THRESHOLDED STTC)
# ==========================================
if thresholded_matrices:
    fig, axes = plt.subplots(1, n_conds, figsize=(7 * n_conds, 6))
    if n_conds == 1:
        axes = [axes]

    for ax, (cond_name, thresh_mat) in zip(axes, thresholded_matrices.items()):
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        cmap.set_under('whitesmoke') 
        
        sns.heatmap(
            thresh_mat, cmap=cmap, vmin=0.001, vmax=1, 
            xticklabels=True, yticklabels=True, ax=ax, square=True, 
            cbar_kws={"shrink": .8}
        )
        
        ax.set_title(f'{cond_name}\n(Top {target_density}% Edges)')
        ax.set_xlabel('Neuron ID')
        ax.set_ylabel('Neuron ID')
        ax.tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    plt.savefig(output_folder / f"STTC_Thresholded_{target_density}pct.jpg", format='jpg', dpi=150)
    plt.close(fig)


# =========================================================
# 8. GRAPH-THEORETIC NETWORK DIAGNOSTICS
# =========================================================
print("\n--- Computing Graph Metrics ---")

use_threshold_mask = False 

graph_metrics = {
    'Condition': [],
    'Char. Path Length (L)': [],
    'Modularity (Q)': [],
    'Small-worldness (σ)': [],
    'Clustering Coeff (C)': [],
    'Global Efficiency': []
}

for cond_name in sttc_matrices.keys():
    # Apply the selected masking criteria
    if use_threshold_mask:
        working_mat = binarized_matrices[cond_name].values
    else:
        # Unmasked: Convert all positive STTC correlations into edges
        working_mat = (sttc_matrices[cond_name].values > 0).astype(int)
        np.fill_diagonal(working_mat, 0)
        
    df_adj = pd.DataFrame(working_mat, index=common_neurons, columns=common_neurons)
    G = nx.from_pandas_adjacency(df_adj)
    
    # Safety check: If the graph has zero edges, metrics will collapse to 0
    if G.number_of_edges() == 0:
        print(f"Warning: The network for '{cond_name}' has 0 edges. Metrics will default to 0.")
        graph_metrics['Condition'].append(cond_name)
        graph_metrics['Char. Path Length (L)'].append(0)
        graph_metrics['Modularity (Q)'].append(0)
        graph_metrics['Small-worldness (σ)'].append(0)
        graph_metrics['Clustering Coeff (C)'].append(0)
        graph_metrics['Global Efficiency'].append(0)
        continue
    
    lcc_nodes = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(lcc_nodes).copy()
    
    C = nx.average_clustering(G) 
    
    # Path length requires at least 2 nodes in the LCC to avoid division by zero
    if len(G_lcc) > 1:
        L = nx.average_shortest_path_length(G_lcc)
    else:
        L = 0
        
    E_glob = nx.global_efficiency(G)
    
    # Modularity requires edges to form communities
    communities = nx.community.greedy_modularity_communities(G)
    Q = nx.community.modularity(G, communities) if len(communities) > 0 else 0
    
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    p_edge = (2.0 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
    
    C_rand_list, L_rand_list = [], []
    
    for i in range(50):
        G_rand = nx.erdos_renyi_graph(n_nodes, p_edge, seed=42 + i)
        C_rand_list.append(nx.average_clustering(G_rand))
        
        if len(G_rand.edges()) > 0: 
            rand_lcc = max(nx.connected_components(G_rand), key=len)
            G_rand_lcc = G_rand.subgraph(rand_lcc)
            if len(G_rand_lcc) > 1:
                L_rand_list.append(nx.average_shortest_path_length(G_rand_lcc))
                
    C_rand = np.mean(C_rand_list) if C_rand_list else 1.0
    L_rand = np.mean(L_rand_list) if L_rand_list else 1.0
    
    sigma = (C / C_rand) / (L / L_rand) if (C_rand > 0 and L_rand > 0) else 0
    
    graph_metrics['Condition'].append(cond_name)
    graph_metrics['Char. Path Length (L)'].append(L)
    graph_metrics['Modularity (Q)'].append(Q)
    graph_metrics['Small-worldness (σ)'].append(sigma)
    graph_metrics['Clustering Coeff (C)'].append(C)
    graph_metrics['Global Efficiency'].append(E_glob)

# Print DataFrame to terminal clearly
df_metrics = pd.DataFrame(graph_metrics)
print("\n=== Topographical Network Metrics ===")
print(df_metrics.set_index('Condition').to_string())

# ---------------------------------------------------------
# Dynamic Plotting
# ---------------------------------------------------------
metrics_to_plot = ['Char. Path Length (L)', 'Modularity (Q)', 'Small-worldness (σ)']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

is_single_condition = len(df_metrics) == 1

for ax, metric in zip(axes, metrics_to_plot):
    if is_single_condition:
        ax.bar(df_metrics['Condition'], df_metrics[metric], color='royalblue', 
               edgecolor='black', linewidth=1.5, width=0.4)
    else:
        ax.plot(df_metrics['Condition'], df_metrics[metric], 
                marker='o', markersize=8, linewidth=2, color='royalblue', 
                markeredgecolor='black', markeredgewidth=1)

    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value')
    
    if is_single_condition:
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    else:
        ax.grid(True, linestyle='--', alpha=0.6)
    
    y_min, y_max = df_metrics[metric].min(), df_metrics[metric].max()
    y_range = max(y_max - y_min, 0.05) 
    ax.set_ylim(max(0, y_min - y_range*0.2), y_max + y_range*0.2) 

plt.suptitle('Functional Network Topological Metrics', fontsize=16, y=1.05)
plt.tight_layout()

# Save final plot
graph_plot_path = output_folder / "Graph_Topological_Metrics.jpg"
plt.savefig(graph_plot_path, format='jpg', dpi=150)
plt.show()

print("\nAnalysis Complete! All plots and metrics saved in the Output_Analysis folder.")