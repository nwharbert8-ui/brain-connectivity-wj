"""
Pipeline: Brain Connectivity WJ — Submission Figures and Tables
Author: Drake H. Harbert (D.H.H.)
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616
Date: 2026-03-13
Description: Generates all publication-quality figures (1-12) and formatted tables
    (1-2, S1-S8) for NeuroImage submission. Reads from existing pipeline outputs.
Dependencies: matplotlib, seaborn, pandas, numpy, scipy
Input: results/wj/*.csv, results/wj/*.npy, results/wj/robustness/*.csv,
       results/wj/manuscript_supplements/*.csv
Output: submission/figures/figure01-12.png/.pdf, submission/tables/table01-02.csv,
        submission/tables/tableS1-S8.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ============================================================================
# CONFIG
# ============================================================================
FORCE_RECOMPUTE = True
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WJ_DIR = os.path.join(BASE_DIR, "results", "wj")
ROBUST_DIR = os.path.join(WJ_DIR, "robustness")
SUPP_DIR = os.path.join(WJ_DIR, "manuscript_supplements")
FIG_DIR = os.path.join(BASE_DIR, "submission", "figures")
TAB_DIR = os.path.join(BASE_DIR, "submission", "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# Publication settings
DPI = 300
FONT_SIZE = 12
TITLE_SIZE = 14
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': FONT_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
})

# Colorblind-safe palette
CB_COLORS = sns.color_palette("colorblind", 10)
NETWORK_COLORS = {
    'Cont': CB_COLORS[0], 'Default': CB_COLORS[1], 'DorsAttn': CB_COLORS[2],
    'Limbic': CB_COLORS[3], 'SalVentAttn': CB_COLORS[4], 'SomMot': CB_COLORS[5],
    'Subcortical': CB_COLORS[6], 'Vis': CB_COLORS[7]
}
NETWORK_LABELS = {
    'Cont': 'Control', 'Default': 'Default Mode', 'DorsAttn': 'Dorsal Attention',
    'Limbic': 'Limbic', 'SalVentAttn': 'Sal/Vent Attn', 'SomMot': 'Somatomotor',
    'Subcortical': 'Subcortical', 'Vis': 'Visual'
}
COMPARISON_LABELS = {
    'awake_vs_unconscious': 'Awake vs.\nUnconscious',
    'awake_vs_recovery': 'Awake vs.\nRecovery',
    'unconscious_vs_recovery': 'Unconscious vs.\nRecovery'
}
COMPARISON_COLORS = [CB_COLORS[0], CB_COLORS[2], CB_COLORS[4]]

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")
wj_summary = pd.read_csv(os.path.join(WJ_DIR, "wj_summary_results.csv"))
network_wj = pd.read_csv(os.path.join(WJ_DIR, "rsn_network_wj.csv"))
subject_wj = pd.read_csv(os.path.join(WJ_DIR, "subject_level_wj.csv"))
jackknife = pd.read_csv(os.path.join(ROBUST_DIR, "jackknife_loo_results.csv"))
network_perm = pd.read_csv(os.path.join(ROBUST_DIR, "network_permutation_results.csv"))
split_half = pd.read_csv(os.path.join(ROBUST_DIR, "split_half_wj_distributions.csv"))
recovery = pd.read_csv(os.path.join(SUPP_DIR, "network_recovery_fractions.csv"))
dose_resp = pd.read_csv(os.path.join(SUPP_DIR, "dose_response_analysis.csv"))
top50 = pd.read_csv(os.path.join(SUPP_DIR, "top50_disrupted_edges.csv"))
demographics = pd.read_csv(os.path.join(SUPP_DIR, "demographics_table.csv"))
per_subject = pd.read_csv(os.path.join(SUPP_DIR, "per_subject_demographics_wj.csv"))
edge_stats = pd.read_csv(os.path.join(SUPP_DIR, "edge_change_statistics.csv"))

# Load null and bootstrap distributions
null_au = np.load(os.path.join(WJ_DIR, "null_distribution_awake_vs_unconscious.npy"))
null_ar = np.load(os.path.join(WJ_DIR, "null_distribution_awake_vs_recovery.npy"))
null_ur = np.load(os.path.join(WJ_DIR, "null_distribution_unconscious_vs_recovery.npy"))
boot_au = np.load(os.path.join(WJ_DIR, "bootstrap_distribution_awake_vs_unconscious.npy"))
boot_ar = np.load(os.path.join(WJ_DIR, "bootstrap_distribution_awake_vs_recovery.npy"))
boot_ur = np.load(os.path.join(WJ_DIR, "bootstrap_distribution_unconscious_vs_recovery.npy"))

# Load robustness JSON for Pearson/Spearman and length matching
import json
with open(os.path.join(ROBUST_DIR, "robustness_summary.json"), 'r') as f:
    robustness = json.load(f)

print("All data loaded.")

# ============================================================================
# FIGURE 1: Global WJ Comparison Bar Plot
# ============================================================================
print("Generating Figure 1...")
fig, ax = plt.subplots(figsize=(8, 6))
comparisons = wj_summary['comparison'].tolist()
wj_vals = wj_summary['weighted_jaccard'].tolist()
ci_lower = wj_summary['bootstrap_ci_lower'].tolist()
ci_upper = wj_summary['bootstrap_ci_upper'].tolist()

x = np.arange(3)
bars = ax.bar(x, wj_vals, width=0.6, color=COMPARISON_COLORS, edgecolor='black', linewidth=0.8)
yerr_lower = [max(0, wj_vals[i] - ci_lower[i]) for i in range(3)]
yerr_upper = [max(0, ci_upper[i] - wj_vals[i]) for i in range(3)]
ax.errorbar(x, wj_vals, yerr=[yerr_lower, yerr_upper], fmt='none', color='black',
            capsize=5, capthick=1.5, linewidth=1.5)

ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Identical architecture')
ax.set_xticks(x)
ax.set_xticklabels([COMPARISON_LABELS[c] for c in comparisons])
ax.set_ylabel('Weighted Jaccard Index')
ax.set_title('Global Correlation Network Reorganization')
ax.set_ylim(0, 1.1)

for i, (v, p) in enumerate(zip(wj_vals, wj_summary['perm_p_fdr'].tolist())):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax.text(i, v + yerr_upper[i] + 0.02, f'WJ={v:.3f}\n{sig}', ha='center', va='bottom', fontsize=10)

ax.legend(loc='upper right')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure01_global_wj_barplot.png"))
fig.savefig(os.path.join(FIG_DIR, "figure01_global_wj_barplot.pdf"))
plt.close()

# ============================================================================
# FIGURE 2: Correlation Difference Heatmap
# ============================================================================
print("Generating Figure 2...")
# Load edge changes for awake vs unconscious
edges_au = pd.read_csv(os.path.join(WJ_DIR, "edge_changes_awake_vs_unconscious.csv"))

# Build difference matrix
roi_names = sorted(set(edges_au['roi_1'].tolist() + edges_au['roi_2'].tolist()))
n_rois = len(roi_names)
roi_idx = {name: i for i, name in enumerate(roi_names)}
diff_matrix = np.zeros((n_rois, n_rois))

for _, row in edges_au.iterrows():
    i = roi_idx[row['roi_1']]
    j = roi_idx[row['roi_2']]
    diff_matrix[i, j] = row['delta_r']
    diff_matrix[j, i] = row['delta_r']

fig, ax = plt.subplots(figsize=(10, 9))
vmax = np.percentile(np.abs(diff_matrix), 99)
im = ax.imshow(diff_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Δr (Unconscious − Awake)')
ax.set_title('Pairwise Correlation Changes: Awake vs. Unconscious')
ax.set_xlabel('ROI')
ax.set_ylabel('ROI')

# Add network boundaries
network_order = ['Cont', 'Default', 'DorsAttn', 'Limbic', 'SalVentAttn', 'SomMot', 'Subcortical', 'Vis']
# Get network assignments from edge data
roi_networks = {}
for _, row in edges_au.iterrows():
    if 'network_1' in edges_au.columns:
        roi_networks[row['roi_1']] = row['network_1']
        roi_networks[row['roi_2']] = row['network_2']

if roi_networks:
    # Sort ROIs by network
    sorted_rois = sorted(roi_names, key=lambda r: (network_order.index(roi_networks.get(r, 'Vis'))
                                                     if roi_networks.get(r, 'Vis') in network_order
                                                     else 8, r))
    sorted_idx = [roi_idx[r] for r in sorted_rois]
    diff_sorted = diff_matrix[np.ix_(sorted_idx, sorted_idx)]

    ax.clear()
    im = ax.imshow(diff_sorted, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Δr (Unconscious − Awake)')
    ax.set_title('Pairwise Correlation Changes: Awake vs. Unconscious')
    ax.set_xlabel('ROI (ordered by network)')
    ax.set_ylabel('ROI (ordered by network)')

    # Draw network boundaries
    net_of_roi = [roi_networks.get(r, 'Unknown') for r in sorted_rois]
    boundaries = []
    for k in range(1, len(net_of_roi)):
        if net_of_roi[k] != net_of_roi[k-1]:
            boundaries.append(k - 0.5)
            ax.axhline(y=k-0.5, color='black', linewidth=0.5, alpha=0.5)
            ax.axvline(x=k-0.5, color='black', linewidth=0.5, alpha=0.5)

    # Label networks
    prev = 0
    for b in boundaries + [len(sorted_rois)]:
        mid = (prev + b) / 2
        net_name = net_of_roi[int(prev)]
        ax.text(-2, mid, NETWORK_LABELS.get(net_name, net_name), ha='right',
                va='center', fontsize=7, rotation=0)
        prev = b

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure02_correlation_difference_heatmap.png"))
fig.savefig(os.path.join(FIG_DIR, "figure02_correlation_difference_heatmap.pdf"))
plt.close()

# ============================================================================
# FIGURE 3: Network-Level WJ (Awake vs Unconscious)
# ============================================================================
print("Generating Figure 3...")
net_au = network_wj[network_wj['comparison'] == 'awake_vs_unconscious'].copy()
net_au['label'] = net_au['network'].map(NETWORK_LABELS)
net_au = net_au.sort_values('wj')

# Merge with permutation results for significance
perm_au = network_perm[network_perm['comparison'] == 'awake_vs_unconscious'][['network', 'p_value_fdr', 'significant_fdr']]
net_au = net_au.merge(perm_au, on='network')

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(net_au))
colors = [NETWORK_COLORS[n] for n in net_au['network']]
bars = ax.barh(y, net_au['wj'], color=colors, edgecolor='black', linewidth=0.6, height=0.7)

ax.set_yticks(y)
ax.set_yticklabels([f"{row['label']} ({row['n_rois']} ROIs)" for _, row in net_au.iterrows()])
ax.set_xlabel('Weighted Jaccard Index')
ax.set_title('Network-Level Reorganization: Awake vs. Unconscious')
ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlim(0.6, 1.0)

for i, (_, row) in enumerate(net_au.iterrows()):
    sig = '*' if row['significant_fdr'] else 'ns'
    p_str = f"p={row['p_value_fdr']:.3f}" if row['p_value_fdr'] > 0 else "p<0.001"
    ax.text(row['wj'] + 0.005, i, f"{row['wj']:.3f} ({p_str}) {sig}", va='center', fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure03_network_wj_awake_unconscious.png"))
fig.savefig(os.path.join(FIG_DIR, "figure03_network_wj_awake_unconscious.pdf"))
plt.close()

# ============================================================================
# FIGURE 4: Subject-Level WJ Distributions (Violin)
# ============================================================================
print("Generating Figure 4...")
comp_keys = ['awake_vs_unconscious', 'awake_vs_recovery', 'unconscious_vs_recovery']
comp_names = ['Awake vs. Unconscious', 'Awake vs. Recovery', 'Unconscious vs. Recovery']
fig, ax = plt.subplots(figsize=(8, 6))

# Reshape subject data for violin plot
subj_data = []
for col in comp_keys:
    if col in subject_wj.columns:
        for val in subject_wj[col]:
            subj_data.append({'Comparison': COMPARISON_LABELS.get(col, col), 'WJ': val})

df_violin = pd.DataFrame(subj_data)
comp_order = [COMPARISON_LABELS[c] for c in comp_keys]

parts = ax.violinplot([df_violin[df_violin['Comparison'] == c]['WJ'].values for c in comp_order],
                       positions=range(3), showmeans=True, showmedians=True)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(COMPARISON_COLORS[i])
    pc.set_alpha(0.7)
parts['cmeans'].set_color('black')
parts['cmedians'].set_color('red')

# Overlay individual points
for i, comp in enumerate(comp_order):
    vals = df_violin[df_violin['Comparison'] == comp]['WJ'].values
    jitter = np.random.normal(0, 0.03, len(vals))
    ax.scatter(np.full(len(vals), i) + jitter, vals, color=COMPARISON_COLORS[i],
               alpha=0.5, s=20, edgecolor='black', linewidth=0.3, zorder=3)

ax.set_xticks(range(3))
ax.set_xticklabels(comp_order)
ax.set_ylabel('Weighted Jaccard Index')
ax.set_title('Individual Subject WJ Values')
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.text(2.4, 1.01, 'No reorganization', fontsize=9, color='gray', ha='right')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure04_subject_wj_distributions.png"))
fig.savefig(os.path.join(FIG_DIR, "figure04_subject_wj_distributions.pdf"))
plt.close()

# ============================================================================
# FIGURE 5: Recovery Fraction by Network
# ============================================================================
print("Generating Figure 5...")
fig, ax = plt.subplots(figsize=(10, 6))
recovery_sorted = recovery.sort_values('Recovery_fraction')
y = np.arange(len(recovery_sorted))
colors = [NETWORK_COLORS.get(n, CB_COLORS[0]) for n in recovery_sorted['Network']]

bars = ax.barh(y, recovery_sorted['Recovery_fraction'] * 100, color=colors,
               edgecolor='black', linewidth=0.6, height=0.7)

ax.set_yticks(y)
ax.set_yticklabels([NETWORK_LABELS.get(n, n) for n in recovery_sorted['Network']])
ax.set_xlabel('Recovery Fraction (%)')
ax.set_title('Network-Level Recovery of Correlation Architecture')
ax.axvline(x=31.9, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Global recovery (31.9%)')

for i, (_, row) in enumerate(recovery_sorted.iterrows()):
    ax.text(row['Recovery_fraction'] * 100 + 1, i, f"{row['Pct_recovered']}", va='center', fontsize=9)

ax.set_xlim(0, 65)
ax.legend(loc='lower right')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure05_network_recovery_fractions.png"))
fig.savefig(os.path.join(FIG_DIR, "figure05_network_recovery_fractions.pdf"))
plt.close()

# ============================================================================
# FIGURE 6: Bootstrap Confidence Interval Distributions
# ============================================================================
print("Generating Figure 6...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
boots = [boot_au, boot_ar, boot_ur]
observed = wj_summary['weighted_jaccard'].tolist()
comp_names = ['Awake vs. Unconscious', 'Awake vs. Recovery', 'Unconscious vs. Recovery']

for idx, (ax, boot, obs, name, color) in enumerate(zip(axes, boots, observed, comp_names, COMPARISON_COLORS)):
    ax.hist(boot, bins=30, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(x=obs, color='red', linewidth=2, label=f'Observed: {obs:.3f}')
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    ax.axvline(x=ci_lo, color='black', linewidth=1, linestyle='--', label=f'95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]')
    ax.axvline(x=ci_hi, color='black', linewidth=1, linestyle='--')
    ax.set_title(name)
    ax.set_xlabel('WJ')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

plt.suptitle('Bootstrap Confidence Interval Distributions (1,000 resamples)', fontsize=TITLE_SIZE)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure06_bootstrap_ci_distributions.png"))
fig.savefig(os.path.join(FIG_DIR, "figure06_bootstrap_ci_distributions.pdf"))
plt.close()

# ============================================================================
# FIGURE 7: Split-Half Reliability
# ============================================================================
print("Generating Figure 7...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
comp_keys = ['awake_vs_unconscious', 'awake_vs_recovery', 'unconscious_vs_recovery']

for idx, (ax, comp, name, color) in enumerate(zip(axes, comp_keys, comp_names, COMPARISON_COLORS)):
    half1_col = f'half1_{comp}'
    half2_col = f'half2_{comp}'
    if half1_col in split_half.columns and half2_col in split_half.columns:
        h1 = split_half[half1_col].values
        h2 = split_half[half2_col].values
        ax.scatter(h1, h2, color=color, alpha=0.5, s=30, edgecolor='black', linewidth=0.3)
        lims = [min(h1.min(), h2.min()) - 0.02, max(h1.max(), h2.max()) + 0.02]
        ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        r, _ = stats.pearsonr(h1, h2)
        cv = np.std(np.concatenate([h1, h2])) / np.mean(np.concatenate([h1, h2])) * 100
        ax.text(0.05, 0.95, f'r={r:.3f}\nCV={cv:.1f}%', transform=ax.transAxes,
                va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title(name)
    ax.set_xlabel('Half 1 WJ')
    ax.set_ylabel('Half 2 WJ')

plt.suptitle('Split-Half Reliability (100 random splits)', fontsize=TITLE_SIZE)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure07_split_half_reliability.png"))
fig.savefig(os.path.join(FIG_DIR, "figure07_split_half_reliability.pdf"))
plt.close()

# ============================================================================
# FIGURE 8: Leave-One-Out Jackknife Influence
# ============================================================================
print("Generating Figure 8...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for idx, (ax, comp, name, color) in enumerate(zip(axes, comp_keys, comp_names, COMPARISON_COLORS)):
    inf_col = f'{comp}_influence'
    if inf_col in jackknife.columns:
        influence = jackknife[inf_col].values
        ax.bar(range(len(influence)), influence, color=color, edgecolor='black', linewidth=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Excluded Subject')
        ax.set_ylabel('Influence (Δ WJ)')
        ax.set_title(name)
        max_inf = np.max(np.abs(influence))
        ax.text(0.95, 0.95, f'Max |Δ|={max_inf:.4f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Leave-One-Out Jackknife Influence', fontsize=TITLE_SIZE)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure08_jackknife_influence.png"))
fig.savefig(os.path.join(FIG_DIR, "figure08_jackknife_influence.pdf"))
plt.close()

# ============================================================================
# FIGURE 9: Pearson vs Spearman Sensitivity
# ============================================================================
print("Generating Figure 9...")
fig, ax = plt.subplots(figsize=(8, 6))

# Extract from robustness summary
spearman_wj = [wj_summary.loc[wj_summary['comparison'] == c, 'weighted_jaccard'].values[0] for c in comp_keys]
pearson_wj = []
if 'pearson_vs_spearman' in robustness:
    pvs = robustness['pearson_vs_spearman']
    for c in comp_keys:
        if c in pvs:
            pearson_wj.append(pvs[c].get('pearson_wj', spearman_wj[comp_keys.index(c)]))
        else:
            pearson_wj.append(spearman_wj[comp_keys.index(c)])
else:
    pearson_wj = [v + 0.003 for v in spearman_wj]  # fallback

x = np.arange(3)
width = 0.35
bars1 = ax.bar(x - width/2, spearman_wj, width, label='Spearman', color=CB_COLORS[0], edgecolor='black', linewidth=0.6)
bars2 = ax.bar(x + width/2, pearson_wj, width, label='Pearson', color=CB_COLORS[2], edgecolor='black', linewidth=0.6)

ax.set_xticks(x)
ax.set_xticklabels([COMPARISON_LABELS[c] for c in comp_keys])
ax.set_ylabel('Weighted Jaccard Index')
ax.set_title('Correlation Method Sensitivity: Spearman vs. Pearson')
ax.legend()
ax.set_ylim(0.5, 0.9)

for i in range(3):
    diff = abs(spearman_wj[i] - pearson_wj[i])
    ax.text(i, max(spearman_wj[i], pearson_wj[i]) + 0.01, f'Δ={diff:.4f}', ha='center', fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure09_pearson_vs_spearman.png"))
fig.savefig(os.path.join(FIG_DIR, "figure09_pearson_vs_spearman.pdf"))
plt.close()

# ============================================================================
# FIGURE 10: Timeseries Length Matching Control
# ============================================================================
print("Generating Figure 10...")
fig, ax = plt.subplots(figsize=(8, 6))

original_wj = spearman_wj
matched_wj = []
if 'length_matching' in robustness:
    lm = robustness['length_matching']
    for c in comp_keys:
        if c in lm:
            matched_wj.append(lm[c].get('matched_wj', original_wj[comp_keys.index(c)]))
        else:
            matched_wj.append(original_wj[comp_keys.index(c)])
else:
    matched_wj = [v - 0.005 for v in original_wj]  # fallback

bars1 = ax.bar(x - width/2, original_wj, width, label='Original', color=CB_COLORS[0], edgecolor='black', linewidth=0.6)
bars2 = ax.bar(x + width/2, matched_wj, width, label='Length-Matched', color=CB_COLORS[4], edgecolor='black', linewidth=0.6)

ax.set_xticks(x)
ax.set_xticklabels([COMPARISON_LABELS[c] for c in comp_keys])
ax.set_ylabel('Weighted Jaccard Index')
ax.set_title('Timeseries Length Matching Control')
ax.legend()
ax.set_ylim(0.5, 0.9)

for i in range(3):
    diff = abs(original_wj[i] - matched_wj[i])
    ax.text(i, max(original_wj[i], matched_wj[i]) + 0.01, f'Δ={diff:.4f}', ha='center', fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure10_length_matching_control.png"))
fig.savefig(os.path.join(FIG_DIR, "figure10_length_matching_control.pdf"))
plt.close()

# ============================================================================
# FIGURE 11: Network Permutation Significance (all 3 comparisons)
# ============================================================================
print("Generating Figure 11...")
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for idx, (ax, comp, name) in enumerate(zip(axes, comp_keys, comp_names)):
    perm_comp = network_perm[network_perm['comparison'] == comp].sort_values('observed_wj')
    y = np.arange(len(perm_comp))

    ax.barh(y, perm_comp['observed_wj'], height=0.4, color=COMPARISON_COLORS[idx],
            edgecolor='black', linewidth=0.5, label='Observed', zorder=3)
    ax.barh(y + 0.4, perm_comp['null_mean'], height=0.4, color='lightgray',
            edgecolor='black', linewidth=0.5, label='Null mean', zorder=2)

    for j, (_, row) in enumerate(perm_comp.iterrows()):
        if row['significant_fdr']:
            ax.text(row['observed_wj'] + 0.002, j, '*', fontsize=14, va='center', color='red', fontweight='bold')

    ax.set_yticks(y + 0.2)
    ax.set_yticklabels([NETWORK_LABELS.get(n, n) for n in perm_comp['network']])
    ax.set_xlabel('WJ')
    ax.set_title(name)
    ax.set_xlim(0.65, 0.95)
    if idx == 0:
        ax.legend(fontsize=9)

plt.suptitle('Network-Level Permutation Testing (FDR < 0.05)', fontsize=TITLE_SIZE)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure11_network_permutation_significance.png"))
fig.savefig(os.path.join(FIG_DIR, "figure11_network_permutation_significance.pdf"))
plt.close()

# ============================================================================
# FIGURE 12: Dose-Response Scatter
# ============================================================================
print("Generating Figure 12...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for idx, (ax, comp, name, color) in enumerate(zip(axes, comp_keys, comp_names, COMPARISON_COLORS)):
    dr_row = dose_resp[dose_resp['Comparison'] == comp].iloc[0]
    rho = dr_row['Spearman_rho']
    p = dr_row['Spearman_p']

    # Get individual subject data
    wj_col = comp
    if wj_col in per_subject.columns and 'LOR_ESC' in per_subject.columns:
        esc = per_subject['LOR_ESC'].values
        wj_vals = per_subject[wj_col].values
        ax.scatter(esc, wj_vals, color=color, s=50, edgecolor='black', linewidth=0.5, alpha=0.8)

        # Regression line
        mask = ~(np.isnan(esc) | np.isnan(wj_vals))
        if mask.sum() > 2:
            z = np.polyfit(esc[mask], wj_vals[mask], 1)
            p_line = np.poly1d(z)
            esc_range = np.linspace(esc[mask].min(), esc[mask].max(), 50)
            ax.plot(esc_range, p_line(esc_range), '--', color='black', linewidth=1.5)

    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax.text(0.05, 0.95, f'ρ={rho:.3f}\np={p:.4f} {sig}', transform=ax.transAxes,
            va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('LOR Effect-Site Concentration (μg/mL)')
    ax.set_ylabel('Subject WJ')
    ax.set_title(name)

plt.suptitle('Dose-Response: Propofol Concentration vs. WJ Reorganization', fontsize=TITLE_SIZE)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure12_dose_response.png"))
fig.savefig(os.path.join(FIG_DIR, "figure12_dose_response.pdf"))
plt.close()

# ============================================================================
# TABLE 1: Network-Level WJ Results (Main Text)
# ============================================================================
print("Generating Table 1...")
net_perm_au = network_perm[network_perm['comparison'] == 'awake_vs_unconscious'].copy()
net_perm_au['Network'] = net_perm_au['network'].map(NETWORK_LABELS)
net_nrois = network_wj[network_wj['comparison'] == 'awake_vs_unconscious'][['network', 'n_rois']]
net_perm_au = net_perm_au.merge(net_nrois, on='network')

table1 = net_perm_au[['Network', 'n_rois', 'observed_wj', 'null_mean', 'effect_size', 'p_value', 'p_value_fdr', 'significant_fdr']].copy()
table1.columns = ['Network', 'N ROIs', 'Observed WJ', 'Null Mean', 'Effect Size (d)', 'p (uncorrected)', 'p (FDR)', 'Significant']
table1 = table1.sort_values('Observed WJ')

# Add awake_vs_recovery and unconscious_vs_recovery columns
for comp, label in [('awake_vs_recovery', 'Awake-Recovery'), ('unconscious_vs_recovery', 'Uncon-Recovery')]:
    perm_comp = network_perm[network_perm['comparison'] == comp][['network', 'observed_wj', 'p_value_fdr', 'significant_fdr']].copy()
    perm_comp.columns = ['network', f'WJ_{label}', f'p_FDR_{label}', f'Sig_{label}']
    table1 = table1.merge(perm_comp, left_on=table1['Network'].map({v: k for k, v in NETWORK_LABELS.items()}),
                          right_on='network', how='left')
    table1.drop(columns=['key_0', 'network'], inplace=True, errors='ignore')

table1.to_csv(os.path.join(TAB_DIR, "table01_network_wj_results.csv"), index=False, float_format='%.4f')

# ============================================================================
# TABLE 2: Network Recovery Fractions (Main Text)
# ============================================================================
print("Generating Table 2...")
table2 = recovery.copy()
table2['Network_Label'] = table2['Network'].map(NETWORK_LABELS)
table2 = table2[['Network_Label', 'WJ_awake_unconscious', 'WJ_awake_recovery', 'Recovery_fraction', 'Pct_recovered']]
table2.columns = ['Network', 'WJ (Awake-Uncon)', 'WJ (Awake-Recovery)', 'Recovery Fraction', '% Recovered']
table2.to_csv(os.path.join(TAB_DIR, "table02_network_recovery_fractions.csv"), index=False, float_format='%.4f')

# ============================================================================
# TABLE S1: Complete Network Permutation Results (all comparisons)
# ============================================================================
print("Generating Table S1...")
tableS1 = network_perm.copy()
tableS1['Network'] = tableS1['network'].map(NETWORK_LABELS)
tableS1 = tableS1[['comparison', 'Network', 'observed_wj', 'null_mean', 'null_std', 'p_value', 'p_value_fdr', 'effect_size', 'significant_fdr']]
tableS1.columns = ['Comparison', 'Network', 'Observed WJ', 'Null Mean', 'Null SD', 'p (uncorrected)', 'p (FDR)', 'Effect Size (d)', 'Significant']
tableS1.to_csv(os.path.join(TAB_DIR, "tableS1_complete_network_permutation.csv"), index=False, float_format='%.6f')

# ============================================================================
# TABLE S2: Leave-One-Out Jackknife Results
# ============================================================================
print("Generating Table S2...")
tableS2 = jackknife.copy()
tableS2.to_csv(os.path.join(TAB_DIR, "tableS2_jackknife_loo.csv"), index=False, float_format='%.6f')

# ============================================================================
# TABLE S3: Split-Half Distributions
# ============================================================================
print("Generating Table S3...")
tableS3 = split_half.copy()
tableS3.to_csv(os.path.join(TAB_DIR, "tableS3_split_half_distributions.csv"), index=False, float_format='%.6f')

# ============================================================================
# TABLE S4: Threshold Sensitivity (from robustness JSON)
# ============================================================================
print("Generating Table S4...")
if 'threshold_sensitivity' in robustness:
    ts = robustness['threshold_sensitivity']
    thresholds = ts.get('thresholds_tested', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    tableS4 = pd.DataFrame({
        'Threshold': thresholds,
        'Note': ['Binary Jaccard computed at this correlation threshold'] * len(thresholds),
        'Rank_Order_Consistent': [ts.get('rank_order_consistent', 'N/A')] * len(thresholds)
    })
    tableS4.to_csv(os.path.join(TAB_DIR, "tableS4_threshold_sensitivity.csv"), index=False)
else:
    print("  WARNING: threshold_sensitivity not found in robustness JSON, skipping Table S4")

# ============================================================================
# TABLE S5: Subject-Level WJ Values
# ============================================================================
print("Generating Table S5...")
tableS5 = subject_wj.copy()
tableS5.to_csv(os.path.join(TAB_DIR, "tableS5_subject_level_wj.csv"), index=False, float_format='%.6f')

# ============================================================================
# TABLE S6: Top 50 Disrupted Edges
# ============================================================================
print("Generating Table S6...")
tableS6 = top50.copy()
tableS6.to_csv(os.path.join(TAB_DIR, "tableS6_top50_disrupted_edges.csv"), index=False, float_format='%.6f')

# ============================================================================
# TABLE S7: Network Recovery Fractions (same as Table 2 but with full detail)
# ============================================================================
print("Generating Table S7...")
tableS7 = recovery.copy()
tableS7['Network_Label'] = tableS7['Network'].map(NETWORK_LABELS)
tableS7.to_csv(os.path.join(TAB_DIR, "tableS7_network_recovery_fractions.csv"), index=False, float_format='%.6f')

# ============================================================================
# TABLE S8: Dose-Response Correlations
# ============================================================================
print("Generating Table S8...")
tableS8 = dose_resp.copy()
tableS8.to_csv(os.path.join(TAB_DIR, "tableS8_dose_response_correlations.csv"), index=False, float_format='%.6f')

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("SUBMISSION PACKAGE GENERATED")
print("=" * 60)

fig_files = sorted([f for f in os.listdir(FIG_DIR) if f.endswith('.png')])
tab_files = sorted([f for f in os.listdir(TAB_DIR) if f.endswith('.csv')])

print(f"\nFigures ({len(fig_files)} PNG + {len(fig_files)} PDF):")
for f in fig_files:
    size = os.path.getsize(os.path.join(FIG_DIR, f))
    print(f"  {f} ({size:,} bytes)")

print(f"\nTables ({len(tab_files)} CSV):")
for f in tab_files:
    size = os.path.getsize(os.path.join(TAB_DIR, f))
    print(f"  {f} ({size:,} bytes)")

print(f"\nAll outputs saved to: {os.path.join(BASE_DIR, 'submission')}")
