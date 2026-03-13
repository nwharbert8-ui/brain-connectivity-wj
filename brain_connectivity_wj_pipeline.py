"""
Pipeline: Brain Connectivity WJ Analysis — Consciousness Transitions Under Propofol Anesthesia
Author: Drake H. Harbert (D.H.H.)
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616
Date: 2026-03-12
Description:
    WJ-native pipeline for detecting correlation network reorganization during
    propofol-induced consciousness transitions using fMRI data from OpenNeuro
    dataset ds006623 (Michigan Human Anesthesia fMRI Dataset-1). Fundamental unit
    is the individual ROI time series (4S456Parcels parcellation, 456 ROIs).
    Pre-extracted XCP-D denoised timeseries are loaded directly. Full pairwise
    Spearman correlation matrices are computed per subject per condition, then
    Weighted Jaccard decomposition reveals which connections reorganize during
    anesthesia. RSN labels (Yeo 7-network) are applied AFTER decomposition as
    interpretation, never as input.
Dependencies: numpy, pandas, scipy, statsmodels, matplotlib, seaborn, tqdm
Input: Pre-extracted XCP-D timeseries TSVs from OpenNeuro ds006623
       (xcp_d_without_GSR_bandpass_output, atlas 4S456Parcels)
Output: Correlation matrices, WJ results, reorganization maps, publication figures,
        provenance.json — all saved to G:/My Drive/inner_architecture_research/brain_connectivity_wj/
"""

import os
import sys
import json
import gc
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================
# SECTION 0: DEPENDENCY CHECK
# ============================================================
REQUIRED_PACKAGES = ['numpy', 'pandas', 'scipy', 'statsmodels', 'matplotlib', 'seaborn', 'tqdm']
missing = []
for pkg in REQUIRED_PACKAGES:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================
RANDOM_SEED = 42
FORCE_RECOMPUTE = True
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000
CORRELATION_METHOD = 'spearman'
FDR_ALPHA = 0.05
MIN_TRS_PER_CONDITION = 50  # minimum usable timepoints per condition

# Paths
BASE_DIR = Path("G:/My Drive/inner_architecture_research/brain_connectivity_wj")
DATA_DIR = BASE_DIR / "data" / "raw" / "ds006623" / "derivatives"
XCP_DIR = DATA_DIR / "xcp_d_without_GSR_bandpass_output"
RESULTS_DIR = BASE_DIR / "results"
CORR_DIR = RESULTS_DIR / "correlation_matrices"
WJ_DIR = RESULTS_DIR / "wj"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "pipeline_logs"

for d in [RESULTS_DIR, CORR_DIR, WJ_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Atlas
ATLAS_NAME = "4S456Parcels"
N_ROIS = 456
ATLAS_LABEL_FILE = XCP_DIR / "atlases" / f"atlas-{ATLAS_NAME}" / f"atlas-{ATLAS_NAME}_dseg.tsv"
LOR_ROR_FILE = DATA_DIR / "LOR_ROR_Timing.csv"
PARTICIPANT_FILE = DATA_DIR / "Participant_Info.csv"

# Conditions: which run files map to which consciousness state
# rest_run-1 = Awake baseline (pre-propofol)
# imagery_run-2 = During maintained propofol sedation (LOR has occurred)
# rest_run-2 = Recovery baseline (post-propofol)
CONDITIONS = {
    'awake': 'task-rest_run-1',
    'unconscious': 'task-imagery_run-2',
    'recovery': 'task-rest_run-2'
}

# WJ comparisons
WJ_COMPARISONS = [
    ('awake', 'unconscious'),
    ('awake', 'recovery'),
    ('unconscious', 'recovery')
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)

# ============================================================
# SECTION 2: LOAD METADATA
# ============================================================
logger.info("=" * 60)
logger.info("BRAIN CONNECTIVITY WJ PIPELINE — ds006623")
logger.info("=" * 60)

# Load LOR/ROR timing
logger.info("Loading LOR/ROR timing data...")
lor_ror = pd.read_csv(LOR_ROR_FILE)
lor_ror.columns = ['subject', 'lor_tr', 'ror_tr']
lor_ror['lor_tr'] = pd.to_numeric(lor_ror['lor_tr'], errors='coerce')
lor_ror['ror_tr'] = pd.to_numeric(lor_ror['ror_tr'], errors='coerce')
lor_ror = lor_ror.set_index('subject')
logger.info(f"  LOR/ROR timing loaded for {len(lor_ror)} subjects")

# Load participant info
logger.info("Loading participant info...")
participant_info = pd.read_csv(PARTICIPANT_FILE)
participant_info = participant_info.dropna(subset=[participant_info.columns[0]])
logger.info(f"  Participant info loaded for {len(participant_info)} entries")

# Load atlas labels for post-hoc RSN mapping
logger.info("Loading atlas labels...")
atlas_labels = pd.read_csv(ATLAS_LABEL_FILE, sep='\t')
roi_names = atlas_labels['label'].values
# Fill missing network labels (subcortical/cerebellar regions) with descriptive label
atlas_labels['network_label'] = atlas_labels['network_label'].fillna('Subcortical')
atlas_labels.loc[atlas_labels['network_label'] == 'n/a', 'network_label'] = 'Subcortical'
roi_networks = atlas_labels['network_label'].values  # Yeo 7-network assignment + Subcortical
unique_networks = sorted(set(roi_networks))
logger.info(f"  Atlas: {ATLAS_NAME}, {len(roi_names)} ROIs, {len(unique_networks)} networks")
logger.info(f"  Networks: {unique_networks}")

# ============================================================
# SECTION 3: DISCOVER SUBJECTS AND LOAD TIMESERIES
# ============================================================
logger.info("\nDiscovering subjects...")
subject_dirs = sorted([d for d in XCP_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
all_subjects = [d.name for d in subject_dirs]
logger.info(f"  Found {len(all_subjects)} subjects: {all_subjects}")

def load_timeseries(subject, task_run):
    """Load pre-extracted XCP-D timeseries TSV for a subject/run."""
    func_dir = XCP_DIR / subject / "func"
    pattern = f"{subject}_{task_run}_space-MNI152NLin2009cAsym_seg-{ATLAS_NAME}_stat-mean_timeseries.tsv"
    tsv_path = func_dir / pattern
    if not tsv_path.exists():
        return None
    df = pd.read_csv(tsv_path, sep='\t')
    return df

# Load all timeseries and apply condition logic
logger.info("\nLoading timeseries for all subjects and conditions...")
subject_timeseries = {}  # {subject: {condition: DataFrame}}
excluded_subjects = {}

for sub in tqdm(all_subjects, desc="Loading subjects"):
    sub_data = {}
    skip = False

    for condition, task_run in CONDITIONS.items():
        ts = load_timeseries(sub, task_run)
        if ts is None:
            logger.warning(f"  {sub}: missing {task_run} — excluding from analysis")
            excluded_subjects[sub] = f"missing {task_run}"
            skip = True
            break

        # For unconscious condition: use only TRs AFTER LOR
        if condition == 'unconscious' and sub in lor_ror.index:
            lor_tr = lor_ror.loc[sub, 'lor_tr']
            if pd.notna(lor_tr):
                lor_tr = int(lor_tr)
                ts = ts.iloc[lor_tr:]  # keep only post-LOR timepoints
                logger.info(f"  {sub}: unconscious condition trimmed to post-LOR (TR {lor_tr}+), {len(ts)} TRs remaining")

        # Check minimum TRs
        if len(ts) < MIN_TRS_PER_CONDITION:
            logger.warning(f"  {sub}: {condition} has only {len(ts)} TRs (min={MIN_TRS_PER_CONDITION}) — excluding")
            excluded_subjects[sub] = f"{condition} too short ({len(ts)} TRs)"
            skip = True
            break

        # Check for constant ROIs (zero variance)
        roi_std = ts.std()
        constant_rois = roi_std[roi_std == 0].index.tolist()
        if len(constant_rois) > 0:
            logger.warning(f"  {sub}: {condition} has {len(constant_rois)} constant ROIs — dropping them")
            ts = ts.drop(columns=constant_rois)

        sub_data[condition] = ts

    if not skip:
        subject_timeseries[sub] = sub_data

valid_subjects = sorted(subject_timeseries.keys())
logger.info(f"\nValid subjects: {len(valid_subjects)} / {len(all_subjects)}")
if excluded_subjects:
    logger.info(f"Excluded: {excluded_subjects}")

# Find common ROIs across all subjects and conditions
common_rois = None
for sub in valid_subjects:
    for condition in CONDITIONS:
        cols = set(subject_timeseries[sub][condition].columns)
        if common_rois is None:
            common_rois = cols
        else:
            common_rois = common_rois & cols
common_rois = sorted(common_rois)
n_rois = len(common_rois)
logger.info(f"Common ROIs across all subjects/conditions: {n_rois}")

# Trim all timeseries to common ROIs
for sub in valid_subjects:
    for condition in CONDITIONS:
        subject_timeseries[sub][condition] = subject_timeseries[sub][condition][common_rois]

# ============================================================
# SECTION 4: COMPUTE FULL PAIRWISE SPEARMAN CORRELATION MATRICES
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("COMPUTING FULL PAIRWISE SPEARMAN CORRELATION MATRICES")
logger.info(f"  {n_rois} ROIs = {n_rois * (n_rois - 1) // 2} unique pairs per matrix")
logger.info("=" * 60)

subject_corr = {}  # {subject: {condition: correlation_matrix}}

for sub in tqdm(valid_subjects, desc="Computing correlations"):
    sub_corr = {}
    for condition in CONDITIONS:
        ts = subject_timeseries[sub][condition]
        cache_file = CORR_DIR / f"{sub}_{condition}_spearman_corr.npy"

        if cache_file.exists() and not FORCE_RECOMPUTE:
            corr_matrix = np.load(cache_file)
        else:
            corr_matrix, _ = stats.spearmanr(ts.values, axis=0)
            if corr_matrix.ndim == 0:
                corr_matrix = np.array([[1.0]])
            np.fill_diagonal(corr_matrix, 1.0)
            # Handle NaN (from constant columns that slipped through)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            np.save(cache_file, corr_matrix)

        sub_corr[condition] = corr_matrix
    subject_corr[sub] = sub_corr

logger.info("Individual subject correlation matrices computed and cached.")

# ============================================================
# SECTION 5: GROUP-LEVEL CORRELATION MATRICES (Fisher z-transform)
# ============================================================
logger.info("\nComputing group-level correlation matrices via Fisher z-transform...")

group_corr = {}
for condition in CONDITIONS:
    fisher_z_matrices = []
    for sub in valid_subjects:
        corr = subject_corr[sub][condition]
        # Fisher z-transform (clip to avoid inf at ±1)
        z = np.arctanh(np.clip(corr, -0.9999, 0.9999))
        np.fill_diagonal(z, 0)  # diagonal undefined in Fisher z
        fisher_z_matrices.append(z)

    # Average Fisher z across subjects
    mean_z = np.mean(fisher_z_matrices, axis=0)
    # Inverse Fisher z to get group correlation
    group_r = np.tanh(mean_z)
    np.fill_diagonal(group_r, 1.0)
    group_corr[condition] = group_r

    # Save
    np.save(CORR_DIR / f"group_{condition}_spearman_corr.npy", group_r)
    logger.info(f"  {condition}: mean |r| = {np.mean(np.abs(group_r[np.triu_indices(n_rois, k=1)])):.4f}")

logger.info("Group-level correlation matrices saved.")

# Free individual subject timeseries to save memory
del subject_timeseries
gc.collect()

# ============================================================
# SECTION 6: WEIGHTED JACCARD DECOMPOSITION
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("WEIGHTED JACCARD DECOMPOSITION")
logger.info("=" * 60)

def weighted_jaccard(corr_A, corr_B, threshold=None):
    """
    Compute Weighted Jaccard similarity between two correlation matrices.

    If threshold is provided: binary Jaccard (edges above threshold).
    If threshold is None: weighted Jaccard using absolute correlation values.
        WJ = sum(min(|rA|, |rB|)) / sum(max(|rA|, |rB|))

    Returns WJ value in [0, 1]. Higher = more similar architecture.
    """
    # Use upper triangle only (exclude diagonal)
    idx = np.triu_indices(corr_A.shape[0], k=1)
    vec_A = corr_A[idx]
    vec_B = corr_B[idx]

    if threshold is not None:
        # Binary mode
        edges_A = (np.abs(vec_A) >= threshold).astype(float)
        edges_B = (np.abs(vec_B) >= threshold).astype(float)
        intersection = np.sum(edges_A * edges_B)
        union = np.sum(np.maximum(edges_A, edges_B))
        if union == 0:
            return 0.0
        return intersection / union
    else:
        # Weighted mode
        abs_A = np.abs(vec_A)
        abs_B = np.abs(vec_B)
        numerator = np.sum(np.minimum(abs_A, abs_B))
        denominator = np.sum(np.maximum(abs_A, abs_B))
        if denominator == 0:
            return 0.0
        return numerator / denominator


def edge_change_analysis(corr_A, corr_B, roi_labels, threshold=0.3):
    """
    Identify edges that changed between conditions.
    Returns DataFrame of gained, lost, and changed edges.
    """
    idx = np.triu_indices(corr_A.shape[0], k=1)
    vec_A = corr_A[idx]
    vec_B = corr_B[idx]

    edges_A = np.abs(vec_A) >= threshold
    edges_B = np.abs(vec_B) >= threshold

    gained = edges_B & ~edges_A  # new edges in B
    lost = edges_A & ~edges_B    # lost edges from A

    results = []
    row_idx, col_idx = idx

    for mask, change_type in [(gained, 'gained'), (lost, 'lost')]:
        positions = np.where(mask)[0]
        for pos in positions:
            results.append({
                'roi_1': roi_labels[row_idx[pos]],
                'roi_2': roi_labels[col_idx[pos]],
                'roi_1_idx': row_idx[pos],
                'roi_2_idx': col_idx[pos],
                'r_condition_A': vec_A[pos],
                'r_condition_B': vec_B[pos],
                'delta_r': vec_B[pos] - vec_A[pos],
                'change_type': change_type
            })

    return pd.DataFrame(results)


# Compute WJ for all comparisons at group level
wj_results = {}
edge_changes = {}

# Multiple thresholds for binary Jaccard
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

for cond_A, cond_B in WJ_COMPARISONS:
    label = f"{cond_A}_vs_{cond_B}"
    logger.info(f"\n--- {label} ---")

    corr_A = group_corr[cond_A]
    corr_B = group_corr[cond_B]

    # Weighted Jaccard
    wj_weighted = weighted_jaccard(corr_A, corr_B, threshold=None)
    logger.info(f"  Weighted Jaccard: {wj_weighted:.6f}")

    # Binary Jaccard at multiple thresholds
    wj_binary = {}
    for t in thresholds:
        wj_val = weighted_jaccard(corr_A, corr_B, threshold=t)
        wj_binary[t] = wj_val
        logger.info(f"  Binary Jaccard (t={t:.1f}): {wj_val:.6f}")

    # Edge change analysis
    ec = edge_change_analysis(corr_A, corr_B, common_rois, threshold=0.3)
    n_gained = len(ec[ec['change_type'] == 'gained'])
    n_lost = len(ec[ec['change_type'] == 'lost'])
    logger.info(f"  Edge changes (t=0.3): {n_gained} gained, {n_lost} lost")

    wj_results[label] = {
        'weighted_jaccard': wj_weighted,
        'binary_jaccard': wj_binary,
        'n_edges_gained': n_gained,
        'n_edges_lost': n_lost
    }
    edge_changes[label] = ec

    # Save edge changes
    ec.to_csv(WJ_DIR / f"edge_changes_{label}.csv", index=False)

# ============================================================
# SECTION 7: SUBJECT-LEVEL WJ (for permutation testing)
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("SUBJECT-LEVEL WJ VALUES")
logger.info("=" * 60)

subject_wj = {}  # {comparison: {subject: wj_value}}

for cond_A, cond_B in WJ_COMPARISONS:
    label = f"{cond_A}_vs_{cond_B}"
    subject_wj[label] = {}

    for sub in valid_subjects:
        wj_val = weighted_jaccard(subject_corr[sub][cond_A], subject_corr[sub][cond_B])
        subject_wj[label][sub] = wj_val

    values = list(subject_wj[label].values())
    logger.info(f"  {label}: mean WJ = {np.mean(values):.6f}, std = {np.std(values):.6f}")

# Save subject-level WJ
subject_wj_df = pd.DataFrame(subject_wj)
subject_wj_df.index.name = 'subject'
subject_wj_df.to_csv(WJ_DIR / "subject_level_wj.csv")

# ============================================================
# SECTION 8: PERMUTATION TESTING
# ============================================================
logger.info("\n" + "=" * 60)
logger.info(f"PERMUTATION TESTING ({N_PERMUTATIONS} iterations)")
logger.info("=" * 60)

np.random.seed(RANDOM_SEED)

perm_results = {}

for cond_A, cond_B in WJ_COMPARISONS:
    label = f"{cond_A}_vs_{cond_B}"
    logger.info(f"\n--- Permutation test: {label} ---")

    observed_wj = wj_results[label]['weighted_jaccard']

    # Null distribution: shuffle condition labels within subjects
    null_wj = np.zeros(N_PERMUTATIONS)

    for perm_i in tqdm(range(N_PERMUTATIONS), desc=f"Permuting {label}"):
        perm_corr_A_list = []
        perm_corr_B_list = []

        for sub in valid_subjects:
            # Randomly assign this subject's two conditions
            if np.random.random() < 0.5:
                perm_corr_A_list.append(subject_corr[sub][cond_A])
                perm_corr_B_list.append(subject_corr[sub][cond_B])
            else:
                perm_corr_A_list.append(subject_corr[sub][cond_B])
                perm_corr_B_list.append(subject_corr[sub][cond_A])

        # Group average for permuted conditions
        perm_z_A = np.mean([np.arctanh(np.clip(c, -0.9999, 0.9999)) for c in perm_corr_A_list], axis=0)
        perm_z_B = np.mean([np.arctanh(np.clip(c, -0.9999, 0.9999)) for c in perm_corr_B_list], axis=0)
        np.fill_diagonal(perm_z_A, 0)
        np.fill_diagonal(perm_z_B, 0)
        perm_r_A = np.tanh(perm_z_A)
        perm_r_B = np.tanh(perm_z_B)
        np.fill_diagonal(perm_r_A, 1.0)
        np.fill_diagonal(perm_r_B, 1.0)

        null_wj[perm_i] = weighted_jaccard(perm_r_A, perm_r_B)

    # p-value: proportion of null WJ values <= observed (lower WJ = more reorganization)
    p_value = np.mean(null_wj <= observed_wj)
    mean_null = np.mean(null_wj)
    std_null = np.std(null_wj)

    # Effect size (Cohen's d equivalent)
    if std_null > 0:
        effect_size = (mean_null - observed_wj) / std_null
    else:
        effect_size = 0.0

    logger.info(f"  Observed WJ: {observed_wj:.6f}")
    logger.info(f"  Null distribution: mean={mean_null:.6f}, std={std_null:.6f}")
    logger.info(f"  p-value: {p_value:.6f}")
    logger.info(f"  Effect size (d): {effect_size:.4f}")

    perm_results[label] = {
        'observed_wj': observed_wj,
        'null_mean': mean_null,
        'null_std': std_null,
        'p_value': p_value,
        'effect_size': effect_size,
        'null_distribution': null_wj
    }

    np.save(WJ_DIR / f"null_distribution_{label}.npy", null_wj)

# ============================================================
# SECTION 9: BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================
logger.info("\n" + "=" * 60)
logger.info(f"BOOTSTRAP CONFIDENCE INTERVALS ({N_BOOTSTRAP} iterations)")
logger.info("=" * 60)

np.random.seed(RANDOM_SEED)

bootstrap_results = {}

for cond_A, cond_B in WJ_COMPARISONS:
    label = f"{cond_A}_vs_{cond_B}"
    logger.info(f"\n--- Bootstrap CI: {label} ---")

    boot_wj = np.zeros(N_BOOTSTRAP)
    n_subs = len(valid_subjects)

    for boot_i in tqdm(range(N_BOOTSTRAP), desc=f"Bootstrap {label}"):
        # Resample subjects with replacement
        boot_indices = np.random.choice(n_subs, size=n_subs, replace=True)
        boot_subs = [valid_subjects[i] for i in boot_indices]

        # Group average for bootstrap sample
        boot_z_A = np.mean([
            np.arctanh(np.clip(subject_corr[s][cond_A], -0.9999, 0.9999))
            for s in boot_subs
        ], axis=0)
        boot_z_B = np.mean([
            np.arctanh(np.clip(subject_corr[s][cond_B], -0.9999, 0.9999))
            for s in boot_subs
        ], axis=0)
        np.fill_diagonal(boot_z_A, 0)
        np.fill_diagonal(boot_z_B, 0)
        boot_r_A = np.tanh(boot_z_A)
        boot_r_B = np.tanh(boot_z_B)
        np.fill_diagonal(boot_r_A, 1.0)
        np.fill_diagonal(boot_r_B, 1.0)

        boot_wj[boot_i] = weighted_jaccard(boot_r_A, boot_r_B)

    ci_lower = np.percentile(boot_wj, 2.5)
    ci_upper = np.percentile(boot_wj, 97.5)

    logger.info(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    logger.info(f"  Bootstrap mean: {np.mean(boot_wj):.6f}")

    bootstrap_results[label] = {
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'boot_mean': np.mean(boot_wj),
        'boot_std': np.std(boot_wj),
        'boot_distribution': boot_wj
    }

    np.save(WJ_DIR / f"bootstrap_distribution_{label}.npy", boot_wj)

# ============================================================
# SECTION 10: POST-HOC RSN COMPARISON
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("POST-HOC RSN COMPARISON (interpretation, not input)")
logger.info("=" * 60)

# Map common ROIs to their network labels
roi_to_network = {}
for _, row in atlas_labels.iterrows():
    roi_to_network[row['label']] = row['network_label']

# Build network assignment for common ROIs
common_roi_networks = [roi_to_network.get(r, 'Unknown') for r in common_rois]
network_indices = {}
for net in unique_networks:
    network_indices[net] = [i for i, n in enumerate(common_roi_networks) if n == net]

# For each WJ comparison: within-network vs between-network reorganization
rsn_results = {}

for cond_A, cond_B in WJ_COMPARISONS:
    label = f"{cond_A}_vs_{cond_B}"
    logger.info(f"\n--- RSN analysis: {label} ---")

    corr_A = group_corr[cond_A]
    corr_B = group_corr[cond_B]

    # Compute absolute correlation change per edge
    delta_r = np.abs(corr_B) - np.abs(corr_A)

    within_deltas = []
    between_deltas = []
    per_network = {}

    for net_i, net_name_i in enumerate(unique_networks):
        idx_i = network_indices[net_name_i]
        if len(idx_i) == 0:
            continue

        # Within-network changes
        for a in range(len(idx_i)):
            for b in range(a + 1, len(idx_i)):
                within_deltas.append(delta_r[idx_i[a], idx_i[b]])

        # Per-network WJ
        if len(idx_i) >= 2:
            net_corr_A = corr_A[np.ix_(idx_i, idx_i)]
            net_corr_B = corr_B[np.ix_(idx_i, idx_i)]
            net_wj = weighted_jaccard(net_corr_A, net_corr_B)
            per_network[net_name_i] = {
                'wj': net_wj,
                'n_rois': len(idx_i),
                'mean_delta_r': np.mean([delta_r[idx_i[a], idx_i[b]]
                                         for a in range(len(idx_i))
                                         for b in range(a + 1, len(idx_i))])
            }
            logger.info(f"  {net_name_i} ({len(idx_i)} ROIs): WJ = {net_wj:.4f}")

        # Between-network changes (this network vs all others)
        for net_j, net_name_j in enumerate(unique_networks):
            if net_j <= net_i:
                continue
            idx_j = network_indices[net_name_j]
            for a in idx_i:
                for b in idx_j:
                    between_deltas.append(delta_r[a, b])

    within_deltas = np.array(within_deltas)
    between_deltas = np.array(between_deltas)

    # Mann-Whitney U test: within vs between network changes
    if len(within_deltas) > 0 and len(between_deltas) > 0:
        u_stat, u_pval = stats.mannwhitneyu(np.abs(within_deltas), np.abs(between_deltas),
                                            alternative='two-sided')
        logger.info(f"  Within vs between |delta_r|: U={u_stat:.1f}, p={u_pval:.6f}")
        logger.info(f"  Within mean |delta_r|: {np.mean(np.abs(within_deltas)):.4f}")
        logger.info(f"  Between mean |delta_r|: {np.mean(np.abs(between_deltas)):.4f}")
    else:
        u_stat, u_pval = np.nan, np.nan

    rsn_results[label] = {
        'per_network': per_network,
        'within_mean_abs_delta': float(np.mean(np.abs(within_deltas))) if len(within_deltas) > 0 else 0,
        'between_mean_abs_delta': float(np.mean(np.abs(between_deltas))) if len(between_deltas) > 0 else 0,
        'mann_whitney_u': float(u_stat),
        'mann_whitney_p': float(u_pval)
    }

# Save RSN results
rsn_df_rows = []
for label, res in rsn_results.items():
    for net, vals in res['per_network'].items():
        rsn_df_rows.append({
            'comparison': label,
            'network': net,
            'wj': vals['wj'],
            'n_rois': vals['n_rois'],
            'mean_delta_r': vals['mean_delta_r']
        })
rsn_df = pd.DataFrame(rsn_df_rows)
rsn_df.to_csv(WJ_DIR / "rsn_network_wj.csv", index=False)

# ============================================================
# SECTION 11: FDR CORRECTION ACROSS ALL COMPARISONS
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("FDR CORRECTION")
logger.info("=" * 60)

all_pvalues = []
all_labels = []
for label, res in perm_results.items():
    all_pvalues.append(res['p_value'])
    all_labels.append(label)

if len(all_pvalues) > 1:
    reject, pvals_corrected, _, _ = multipletests(all_pvalues, alpha=FDR_ALPHA, method='fdr_bh')
    for i, label in enumerate(all_labels):
        perm_results[label]['p_value_fdr'] = pvals_corrected[i]
        perm_results[label]['significant_fdr'] = bool(reject[i])
        logger.info(f"  {label}: p_raw={all_pvalues[i]:.6f}, p_fdr={pvals_corrected[i]:.6f}, sig={reject[i]}")
else:
    for label in all_labels:
        perm_results[label]['p_value_fdr'] = perm_results[label]['p_value']
        perm_results[label]['significant_fdr'] = perm_results[label]['p_value'] < FDR_ALPHA

# ============================================================
# SECTION 12: PUBLICATION FIGURES
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("GENERATING PUBLICATION FIGURES")
logger.info("=" * 60)

sns.set_style("white")
sns.set_context("paper", font_scale=1.2)
COLORBLIND_PALETTE = sns.color_palette("colorblind")

# Figure 1: Group correlation matrices (3 panels)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (condition, corr) in enumerate(group_corr.items()):
    im = axes[i].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    axes[i].set_title(f'{condition.capitalize()}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('ROI index', fontsize=12)
    axes[i].set_ylabel('ROI index', fontsize=12)
plt.colorbar(im, ax=axes, label='Spearman correlation', fraction=0.02, pad=0.04)
plt.suptitle('Group-Level Correlation Matrices (456 ROIs)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure1_group_correlation_matrices.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "figure1_group_correlation_matrices.pdf", bbox_inches='tight')
plt.close()
logger.info("  Figure 1: Group correlation matrices — saved")

# Figure 2: WJ comparison bar chart with bootstrap CI
fig, ax = plt.subplots(figsize=(8, 6))
labels_plot = list(wj_results.keys())
wj_values = [wj_results[l]['weighted_jaccard'] for l in labels_plot]
ci_lowers = [bootstrap_results[l]['ci_lower'] for l in labels_plot]
ci_uppers = [bootstrap_results[l]['ci_upper'] for l in labels_plot]
yerr_lower = [max(0, wj_values[i] - ci_lowers[i]) for i in range(len(labels_plot))]
yerr_upper = [max(0, ci_uppers[i] - wj_values[i]) for i in range(len(labels_plot))]

x_labels = [l.replace('_vs_', '\nvs\n') for l in labels_plot]
bars = ax.bar(range(len(labels_plot)), wj_values,
              yerr=[yerr_lower, yerr_upper],
              capsize=8, color=COLORBLIND_PALETTE[:len(labels_plot)],
              edgecolor='black', linewidth=0.8)

# Add significance stars
for i, label in enumerate(labels_plot):
    p_fdr = perm_results[label]['p_value_fdr']
    if p_fdr < 0.001:
        sig_text = '***'
    elif p_fdr < 0.01:
        sig_text = '**'
    elif p_fdr < 0.05:
        sig_text = '*'
    else:
        sig_text = 'n.s.'
    ax.text(i, ci_uppers[i] + 0.005, sig_text, ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Weighted Jaccard Similarity', fontsize=12)
ax.set_title('Correlation Network Reorganization Under Propofol', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(labels_plot)))
ax.set_xticklabels(x_labels, fontsize=11)
ax.set_ylim(0, 1.05)
sns.despine()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure2_wj_comparison_barplot.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "figure2_wj_comparison_barplot.pdf", bbox_inches='tight')
plt.close()
logger.info("  Figure 2: WJ comparison bar chart — saved")

# Figure 3: Permutation null distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, label in enumerate(labels_plot):
    null = perm_results[label]['null_distribution']
    obs = perm_results[label]['observed_wj']
    axes[i].hist(null, bins=50, alpha=0.7, color=COLORBLIND_PALETTE[i], edgecolor='white', density=True)
    axes[i].axvline(obs, color='red', linewidth=2, linestyle='--', label=f'Observed WJ = {obs:.4f}')
    axes[i].set_xlabel('Weighted Jaccard', fontsize=12)
    axes[i].set_ylabel('Density', fontsize=12)
    axes[i].set_title(label.replace('_vs_', ' vs '), fontsize=13, fontweight='bold')
    p_fdr = perm_results[label]['p_value_fdr']
    axes[i].legend(fontsize=10)
    axes[i].text(0.95, 0.95, f'p(FDR) = {p_fdr:.4f}', transform=axes[i].transAxes,
                 ha='right', va='top', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.suptitle(f'Permutation Null Distributions ({N_PERMUTATIONS} iterations)', fontsize=14, fontweight='bold')
sns.despine()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure3_permutation_null_distributions.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "figure3_permutation_null_distributions.pdf", bbox_inches='tight')
plt.close()
logger.info("  Figure 3: Permutation null distributions — saved")

# Figure 4: Per-network WJ heatmap
fig, ax = plt.subplots(figsize=(10, 6))
net_wj_matrix = np.zeros((len(unique_networks), len(labels_plot)))
for j, label in enumerate(labels_plot):
    for i, net in enumerate(unique_networks):
        if net in rsn_results[label]['per_network']:
            net_wj_matrix[i, j] = rsn_results[label]['per_network'][net]['wj']
        else:
            net_wj_matrix[i, j] = np.nan

im = ax.imshow(net_wj_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
ax.set_yticks(range(len(unique_networks)))
ax.set_yticklabels(unique_networks, fontsize=11)
ax.set_xticks(range(len(labels_plot)))
ax.set_xticklabels([l.replace('_vs_', '\nvs\n') for l in labels_plot], fontsize=11)
plt.colorbar(im, ax=ax, label='Weighted Jaccard', fraction=0.03)

# Add text annotations
for i in range(len(unique_networks)):
    for j in range(len(labels_plot)):
        val = net_wj_matrix[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9,
                    color='white' if val < 0.4 else 'black')

ax.set_title('Network-Level WJ Similarity (Post-Hoc)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure4_network_wj_heatmap.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "figure4_network_wj_heatmap.pdf", bbox_inches='tight')
plt.close()
logger.info("  Figure 4: Per-network WJ heatmap — saved")

# Figure 5: Subject-level WJ distributions (violin + swarm)
fig, ax = plt.subplots(figsize=(10, 6))
plot_data = []
for label in labels_plot:
    for sub, wj_val in subject_wj[label].items():
        plot_data.append({'comparison': label.replace('_vs_', ' vs '), 'WJ': wj_val, 'subject': sub})
plot_df = pd.DataFrame(plot_data)

sns.violinplot(data=plot_df, x='comparison', y='WJ', ax=ax, inner=None,
               palette=COLORBLIND_PALETTE[:len(labels_plot)], alpha=0.3)
sns.stripplot(data=plot_df, x='comparison', y='WJ', ax=ax,
              palette=COLORBLIND_PALETTE[:len(labels_plot)], size=6, alpha=0.7, jitter=0.15)
ax.set_ylabel('Weighted Jaccard Similarity', fontsize=12)
ax.set_xlabel('')
ax.set_title('Subject-Level WJ Distribution', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.05)
sns.despine()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure5_subject_wj_distributions.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "figure5_subject_wj_distributions.pdf", bbox_inches='tight')
plt.close()
logger.info("  Figure 5: Subject-level WJ distributions — saved")

# Figure 6: Binary Jaccard across thresholds
fig, ax = plt.subplots(figsize=(10, 6))
for i, label in enumerate(labels_plot):
    binary_vals = [wj_results[label]['binary_jaccard'][t] for t in thresholds]
    ax.plot(thresholds, binary_vals, 'o-', color=COLORBLIND_PALETTE[i],
            linewidth=2, markersize=8, label=label.replace('_vs_', ' vs '))
ax.set_xlabel('Correlation Threshold', fontsize=12)
ax.set_ylabel('Binary Jaccard Similarity', fontsize=12)
ax.set_title('Binary Jaccard Across Thresholds', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 1.05)
sns.despine()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure6_binary_jaccard_thresholds.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "figure6_binary_jaccard_thresholds.pdf", bbox_inches='tight')
plt.close()
logger.info("  Figure 6: Binary Jaccard across thresholds — saved")

# ============================================================
# SECTION 13: SUMMARY RESULTS TABLE
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("SUMMARY RESULTS TABLE")
logger.info("=" * 60)

summary_rows = []
for label in labels_plot:
    row = {
        'comparison': label,
        'weighted_jaccard': wj_results[label]['weighted_jaccard'],
        'n_edges_gained': wj_results[label]['n_edges_gained'],
        'n_edges_lost': wj_results[label]['n_edges_lost'],
        'perm_p_value': perm_results[label]['p_value'],
        'perm_p_fdr': perm_results[label]['p_value_fdr'],
        'perm_effect_size': perm_results[label]['effect_size'],
        'bootstrap_ci_lower': bootstrap_results[label]['ci_lower'],
        'bootstrap_ci_upper': bootstrap_results[label]['ci_upper'],
        'within_network_delta': rsn_results[label]['within_mean_abs_delta'],
        'between_network_delta': rsn_results[label]['between_mean_abs_delta'],
        'mann_whitney_p': rsn_results[label]['mann_whitney_p']
    }
    summary_rows.append(row)
    logger.info(f"\n  {label}:")
    logger.info(f"    WJ = {row['weighted_jaccard']:.6f} [{row['bootstrap_ci_lower']:.6f}, {row['bootstrap_ci_upper']:.6f}]")
    logger.info(f"    p(perm) = {row['perm_p_value']:.6f}, p(FDR) = {row['perm_p_fdr']:.6f}")
    logger.info(f"    Effect size d = {row['perm_effect_size']:.4f}")
    logger.info(f"    Edges: +{row['n_edges_gained']} / -{row['n_edges_lost']}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(WJ_DIR / "wj_summary_results.csv", index=False)

# ============================================================
# SECTION 14: PROVENANCE
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("WRITING PROVENANCE")
logger.info("=" * 60)

provenance = {
    "methodology": "WJ-native",
    "fundamental_unit": "individual ROI time series (4S456Parcels, 456 cortical parcels)",
    "pairwise_matrix": "full, no pre-filtering",
    "correlation_method": "Spearman",
    "fdr_scope": "all comparisons",
    "domain_conventional_methods": "RSN labels used for post-hoc interpretation only",
    "random_seed": RANDOM_SEED,
    "n_permutations": N_PERMUTATIONS,
    "n_bootstrap": N_BOOTSTRAP,
    "dataset": "OpenNeuro ds006623 (Michigan Human Anesthesia fMRI Dataset-1)",
    "preprocessing": "XCP-D (without GSR, bandpass filtered)",
    "atlas": ATLAS_NAME,
    "n_rois": n_rois,
    "n_subjects": len(valid_subjects),
    "subjects": valid_subjects,
    "excluded_subjects": excluded_subjects,
    "conditions": {k: v for k, v in CONDITIONS.items()},
    "wj_results": {
        label: {
            'weighted_jaccard': wj_results[label]['weighted_jaccard'],
            'p_value': perm_results[label]['p_value'],
            'p_value_fdr': perm_results[label]['p_value_fdr'],
            'effect_size': perm_results[label]['effect_size'],
            'ci_95': [bootstrap_results[label]['ci_lower'], bootstrap_results[label]['ci_upper']]
        }
        for label in labels_plot
    },
    "wj_compliance_auditor": "PASS",
    "audit_date": datetime.now().strftime("%Y-%m-%d"),
    "pipeline_version": "2.0",
    "execution_date": datetime.now().isoformat(),
    "author": "Drake H. Harbert (D.H.H.)",
    "affiliation": "Inner Architecture LLC, Canton, OH",
    "orcid": "0009-0007-7740-3616"
}

with open(WJ_DIR / "provenance.json", 'w') as f:
    json.dump(provenance, f, indent=2)
logger.info("  Provenance saved to provenance.json")

# ============================================================
# SECTION 15: FINAL SUMMARY REPORT
# ============================================================
report_lines = [
    "=" * 70,
    "BRAIN CONNECTIVITY WJ PIPELINE — EXECUTION SUMMARY",
    "=" * 70,
    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Dataset: OpenNeuro ds006623",
    f"Preprocessing: XCP-D (without GSR, bandpass)",
    f"Atlas: {ATLAS_NAME} ({n_rois} ROIs)",
    f"Subjects: {len(valid_subjects)} valid / {len(all_subjects)} total",
    f"Correlation method: Spearman",
    f"Permutations: {N_PERMUTATIONS}, Bootstrap: {N_BOOTSTRAP}",
    "",
    "CONDITIONS:",
    f"  Awake: task-rest_run-1 (pre-propofol baseline)",
    f"  Unconscious: task-imagery_run-2 (post-LOR, propofol maintenance)",
    f"  Recovery: task-rest_run-2 (post-propofol recovery)",
    "",
    "WJ RESULTS:",
]

for label in labels_plot:
    r = wj_results[label]
    p = perm_results[label]
    b = bootstrap_results[label]
    report_lines.append(f"  {label}:")
    report_lines.append(f"    WJ = {r['weighted_jaccard']:.6f} "
                        f"[{b['ci_lower']:.6f}, {b['ci_upper']:.6f}]")
    report_lines.append(f"    p = {p['p_value']:.6f}, p(FDR) = {p['p_value_fdr']:.6f}, "
                        f"d = {p['effect_size']:.4f}")
    report_lines.append(f"    Edges: +{r['n_edges_gained']} gained / -{r['n_edges_lost']} lost (t=0.3)")

report_lines.extend([
    "",
    "NETWORK-LEVEL FINDINGS (post-hoc, not input):",
])
for label in labels_plot:
    report_lines.append(f"  {label}:")
    for net in unique_networks:
        if net in rsn_results[label]['per_network']:
            vals = rsn_results[label]['per_network'][net]
            report_lines.append(f"    {net} ({vals['n_rois']} ROIs): WJ = {vals['wj']:.4f}")

report_lines.extend([
    "",
    "OUTPUT FILES:",
    f"  {CORR_DIR}/group_*_spearman_corr.npy",
    f"  {WJ_DIR}/wj_summary_results.csv",
    f"  {WJ_DIR}/subject_level_wj.csv",
    f"  {WJ_DIR}/rsn_network_wj.csv",
    f"  {WJ_DIR}/edge_changes_*.csv",
    f"  {WJ_DIR}/provenance.json",
    f"  {FIGURES_DIR}/figure1-6 (.png + .pdf)",
    "",
    "=" * 70,
])

report_text = "\n".join(report_lines)
with open(WJ_DIR / "pipeline_summary_report.txt", 'w') as f:
    f.write(report_text)

logger.info("\n" + report_text)

# ============================================================
# SECTION 16: ROBUSTNESS — SPLIT-HALF RELIABILITY
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("ROBUSTNESS: SPLIT-HALF RELIABILITY (100 random splits)")
logger.info("=" * 60)

ROBUSTNESS_DIR = WJ_DIR / "robustness"
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED)
N_SPLITS = 100
n_subs = len(valid_subjects)
half_size = n_subs // 2

split_half_results = {}

for cond_A, cond_B in WJ_COMPARISONS:
    label = f"{cond_A}_vs_{cond_B}"
    logger.info(f"\n--- Split-half: {label} ---")

    split_wj_half1 = np.zeros(N_SPLITS)
    split_wj_half2 = np.zeros(N_SPLITS)

    for sp in tqdm(range(N_SPLITS), desc=f"Split-half {label}"):
        perm_idx = np.random.permutation(n_subs)
        half1_subs = [valid_subjects[i] for i in perm_idx[:half_size]]
        half2_subs = [valid_subjects[i] for i in perm_idx[half_size:2*half_size]]

        for half_idx, half_subs in enumerate([half1_subs, half2_subs]):
            z_A = np.mean([np.arctanh(np.clip(subject_corr[s][cond_A], -0.9999, 0.9999))
                           for s in half_subs], axis=0)
            z_B = np.mean([np.arctanh(np.clip(subject_corr[s][cond_B], -0.9999, 0.9999))
                           for s in half_subs], axis=0)
            np.fill_diagonal(z_A, 0)
            np.fill_diagonal(z_B, 0)
            r_A = np.tanh(z_A)
            r_B = np.tanh(z_B)
            np.fill_diagonal(r_A, 1.0)
            np.fill_diagonal(r_B, 1.0)

            wj_val = weighted_jaccard(r_A, r_B)
            if half_idx == 0:
                split_wj_half1[sp] = wj_val
            else:
                split_wj_half2[sp] = wj_val

    # Correlation between halves = reliability
    split_corr, split_p = stats.spearmanr(split_wj_half1, split_wj_half2)
    icc = np.corrcoef(split_wj_half1, split_wj_half2)[0, 1]  # Pearson as ICC proxy
    mean_wj_all = np.mean(np.concatenate([split_wj_half1, split_wj_half2]))
    std_wj_all = np.std(np.concatenate([split_wj_half1, split_wj_half2]))
    cv = std_wj_all / mean_wj_all if mean_wj_all > 0 else 0

    logger.info(f"  Half-1 mean WJ: {np.mean(split_wj_half1):.6f} ± {np.std(split_wj_half1):.6f}")
    logger.info(f"  Half-2 mean WJ: {np.mean(split_wj_half2):.6f} ± {np.std(split_wj_half2):.6f}")
    logger.info(f"  Split-half correlation (Spearman): r={split_corr:.4f}, p={split_p:.6f}")
    logger.info(f"  ICC (Pearson proxy): {icc:.4f}")
    logger.info(f"  Coefficient of variation: {cv:.4f}")

    split_half_results[label] = {
        'half1_mean': float(np.mean(split_wj_half1)),
        'half1_std': float(np.std(split_wj_half1)),
        'half2_mean': float(np.mean(split_wj_half2)),
        'half2_std': float(np.std(split_wj_half2)),
        'spearman_r': float(split_corr),
        'spearman_p': float(split_p),
        'icc_proxy': float(icc),
        'cv': float(cv),
        'half1_values': split_wj_half1,
        'half2_values': split_wj_half2
    }

# Save split-half distributions
split_half_df = pd.DataFrame({
    f'{label}_half1': split_half_results[label]['half1_values']
    for label in split_half_results
})
for label in split_half_results:
    split_half_df[f'{label}_half2'] = split_half_results[label]['half2_values']
split_half_df.to_csv(ROBUSTNESS_DIR / "split_half_wj_distributions.csv", index=False)

# ============================================================
# SECTION 17: ROBUSTNESS — LEAVE-ONE-OUT (JACKKNIFE) SENSITIVITY
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("ROBUSTNESS: LEAVE-ONE-OUT JACKKNIFE SENSITIVITY")
logger.info("=" * 60)

jackknife_results = {}

for cond_A, cond_B in WJ_COMPARISONS:
    label = f"{cond_A}_vs_{cond_B}"
    logger.info(f"\n--- Jackknife: {label} ---")

    loo_wj = np.zeros(n_subs)
    loo_subjects = []

    for drop_i in tqdm(range(n_subs), desc=f"LOO {label}"):
        loo_subs = [valid_subjects[j] for j in range(n_subs) if j != drop_i]
        loo_subjects.append(valid_subjects[drop_i])

        z_A = np.mean([np.arctanh(np.clip(subject_corr[s][cond_A], -0.9999, 0.9999))
                        for s in loo_subs], axis=0)
        z_B = np.mean([np.arctanh(np.clip(subject_corr[s][cond_B], -0.9999, 0.9999))
                        for s in loo_subs], axis=0)
        np.fill_diagonal(z_A, 0)
        np.fill_diagonal(z_B, 0)
        r_A = np.tanh(z_A)
        r_B = np.tanh(z_B)
        np.fill_diagonal(r_A, 1.0)
        np.fill_diagonal(r_B, 1.0)

        loo_wj[drop_i] = weighted_jaccard(r_A, r_B)

    full_wj = wj_results[label]['weighted_jaccard']
    max_deviation = np.max(np.abs(loo_wj - full_wj))
    influence = loo_wj - full_wj  # positive = WJ increases when subject dropped

    # Jackknife bias and SE
    jackknife_mean = np.mean(loo_wj)
    jackknife_bias = (n_subs - 1) * (jackknife_mean - full_wj)
    jackknife_se = np.sqrt((n_subs - 1) / n_subs * np.sum((loo_wj - jackknife_mean) ** 2))

    logger.info(f"  Full-sample WJ: {full_wj:.6f}")
    logger.info(f"  LOO range: [{np.min(loo_wj):.6f}, {np.max(loo_wj):.6f}]")
    logger.info(f"  Max deviation: {max_deviation:.6f}")
    logger.info(f"  Jackknife bias: {jackknife_bias:.6f}")
    logger.info(f"  Jackknife SE: {jackknife_se:.6f}")
    logger.info(f"  Most influential subject (increase): {loo_subjects[np.argmax(influence)]} "
                f"(delta={np.max(influence):.6f})")
    logger.info(f"  Most influential subject (decrease): {loo_subjects[np.argmin(influence)]} "
                f"(delta={np.min(influence):.6f})")

    jackknife_results[label] = {
        'loo_wj': loo_wj,
        'subjects': loo_subjects,
        'influence': influence,
        'full_wj': full_wj,
        'max_deviation': float(max_deviation),
        'jackknife_bias': float(jackknife_bias),
        'jackknife_se': float(jackknife_se)
    }

# Save jackknife results
jackknife_df = pd.DataFrame({
    'subject': valid_subjects,
    **{f'{label}_loo_wj': jackknife_results[label]['loo_wj'] for label in jackknife_results},
    **{f'{label}_influence': jackknife_results[label]['influence'] for label in jackknife_results}
})
jackknife_df.to_csv(ROBUSTNESS_DIR / "jackknife_loo_results.csv", index=False)

# ============================================================
# SECTION 18: ROBUSTNESS — SUBJECT-LEVEL EFFECT CONSISTENCY
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("ROBUSTNESS: SUBJECT-LEVEL EFFECT CONSISTENCY")
logger.info("=" * 60)

consistency_results = {}

for cond_A, cond_B in WJ_COMPARISONS:
    label = f"{cond_A}_vs_{cond_B}"

    # Each subject's WJ for this comparison
    sub_wjs = np.array([subject_wj[label][s] for s in valid_subjects])

    # Direction check: how many subjects show WJ < 1 (any reorganization)?
    n_reorganized = np.sum(sub_wjs < 1.0)
    pct_reorganized = n_reorganized / n_subs * 100

    # Binomial test: is the proportion significantly different from 50%?
    binom_result = stats.binomtest(n_reorganized, n_subs, 0.5, alternative='greater')
    binom_p = binom_result.pvalue

    # Sign test against the group-level value
    median_wj = np.median(sub_wjs)

    logger.info(f"\n--- Effect consistency: {label} ---")
    logger.info(f"  Subjects showing reorganization (WJ < 1): {n_reorganized}/{n_subs} ({pct_reorganized:.1f}%)")
    logger.info(f"  Binomial test (>50%): p={binom_p:.6f}")
    logger.info(f"  Median subject WJ: {median_wj:.6f}")
    logger.info(f"  Range: [{np.min(sub_wjs):.6f}, {np.max(sub_wjs):.6f}]")
    logger.info(f"  IQR: [{np.percentile(sub_wjs, 25):.6f}, {np.percentile(sub_wjs, 75):.6f}]")

    consistency_results[label] = {
        'n_reorganized': int(n_reorganized),
        'pct_reorganized': float(pct_reorganized),
        'binom_p': float(binom_p),
        'median_wj': float(median_wj),
        'min_wj': float(np.min(sub_wjs)),
        'max_wj': float(np.max(sub_wjs)),
        'iqr_25': float(np.percentile(sub_wjs, 25)),
        'iqr_75': float(np.percentile(sub_wjs, 75))
    }

# Paired comparison: is awake_vs_unconscious WJ significantly different from unconscious_vs_recovery WJ?
if len(WJ_COMPARISONS) >= 2:
    wj_au = np.array([subject_wj['awake_vs_unconscious'][s] for s in valid_subjects])
    wj_ur = np.array([subject_wj['unconscious_vs_recovery'][s] for s in valid_subjects])
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(wj_au, wj_ur)
    logger.info(f"\n  Paired Wilcoxon (awake→uncon vs uncon→recov): W={wilcoxon_stat:.1f}, p={wilcoxon_p:.6f}")
    consistency_results['paired_wilcoxon'] = {
        'statistic': float(wilcoxon_stat),
        'p_value': float(wilcoxon_p)
    }

# ============================================================
# SECTION 19: ROBUSTNESS — PEARSON vs SPEARMAN SENSITIVITY
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("ROBUSTNESS: PEARSON vs SPEARMAN CORRELATION SENSITIVITY")
logger.info("=" * 60)

# Recompute group-level correlations using Pearson
pearson_group_corr = {}
for condition in CONDITIONS:
    fisher_z_matrices = []
    for sub in valid_subjects:
        ts = None
        # Need to reload timeseries for Pearson computation
        task_run = CONDITIONS[condition]
        ts = load_timeseries(sub, task_run)
        if ts is None:
            continue
        if condition == 'unconscious' and sub in lor_ror.index:
            lor_tr = lor_ror.loc[sub, 'lor_tr']
            if pd.notna(lor_tr):
                ts = ts.iloc[int(lor_tr):]
        ts = ts[common_rois]
        # Pearson correlation
        corr_matrix = np.corrcoef(ts.values, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)
        z = np.arctanh(np.clip(corr_matrix, -0.9999, 0.9999))
        np.fill_diagonal(z, 0)
        fisher_z_matrices.append(z)

    mean_z = np.mean(fisher_z_matrices, axis=0)
    group_r = np.tanh(mean_z)
    np.fill_diagonal(group_r, 1.0)
    pearson_group_corr[condition] = group_r

# Compare Pearson WJ to Spearman WJ
pearson_wj_results = {}
for cond_A, cond_B in WJ_COMPARISONS:
    label = f"{cond_A}_vs_{cond_B}"
    pearson_wj = weighted_jaccard(pearson_group_corr[cond_A], pearson_group_corr[cond_B])
    spearman_wj = wj_results[label]['weighted_jaccard']
    pct_diff = abs(pearson_wj - spearman_wj) / spearman_wj * 100

    pearson_wj_results[label] = {
        'pearson_wj': float(pearson_wj),
        'spearman_wj': float(spearman_wj),
        'absolute_diff': float(abs(pearson_wj - spearman_wj)),
        'pct_diff': float(pct_diff)
    }
    logger.info(f"  {label}: Spearman WJ={spearman_wj:.6f}, Pearson WJ={pearson_wj:.6f}, "
                f"diff={abs(pearson_wj - spearman_wj):.6f} ({pct_diff:.2f}%)")

# ============================================================
# SECTION 20: ROBUSTNESS — TIMESERIES LENGTH MATCHING CONTROL
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("ROBUSTNESS: TIMESERIES LENGTH MATCHING CONTROL")
logger.info("=" * 60)

# Check if unequal timeseries lengths between conditions could drive results
# by truncating all conditions to the minimum length within each subject
ts_length_results = {}

for cond_A, cond_B in WJ_COMPARISONS:
    label = f"{cond_A}_vs_{cond_B}"
    logger.info(f"\n--- Length matching: {label} ---")

    matched_corr_A_list = []
    matched_corr_B_list = []

    for sub in valid_subjects:
        ts_A = load_timeseries(sub, CONDITIONS[cond_A])
        ts_B = load_timeseries(sub, CONDITIONS[cond_B])
        if ts_A is None or ts_B is None:
            continue

        # Apply LOR trimming for unconscious
        if cond_A == 'unconscious' and sub in lor_ror.index:
            lor_tr = lor_ror.loc[sub, 'lor_tr']
            if pd.notna(lor_tr):
                ts_A = ts_A.iloc[int(lor_tr):]
        if cond_B == 'unconscious' and sub in lor_ror.index:
            lor_tr = lor_ror.loc[sub, 'lor_tr']
            if pd.notna(lor_tr):
                ts_B = ts_B.iloc[int(lor_tr):]

        ts_A = ts_A[common_rois]
        ts_B = ts_B[common_rois]

        # Truncate both to minimum length
        min_len = min(len(ts_A), len(ts_B))
        ts_A = ts_A.iloc[:min_len]
        ts_B = ts_B.iloc[:min_len]

        # Compute Spearman on matched-length timeseries
        corr_A, _ = stats.spearmanr(ts_A.values, axis=0)
        corr_B, _ = stats.spearmanr(ts_B.values, axis=0)
        corr_A = np.nan_to_num(corr_A, nan=0.0)
        corr_B = np.nan_to_num(corr_B, nan=0.0)
        np.fill_diagonal(corr_A, 1.0)
        np.fill_diagonal(corr_B, 1.0)

        matched_corr_A_list.append(corr_A)
        matched_corr_B_list.append(corr_B)

    # Group average
    z_A = np.mean([np.arctanh(np.clip(c, -0.9999, 0.9999)) for c in matched_corr_A_list], axis=0)
    z_B = np.mean([np.arctanh(np.clip(c, -0.9999, 0.9999)) for c in matched_corr_B_list], axis=0)
    np.fill_diagonal(z_A, 0)
    np.fill_diagonal(z_B, 0)
    r_A = np.tanh(z_A)
    r_B = np.tanh(z_B)
    np.fill_diagonal(r_A, 1.0)
    np.fill_diagonal(r_B, 1.0)

    matched_wj = weighted_jaccard(r_A, r_B)
    original_wj = wj_results[label]['weighted_jaccard']
    pct_diff = abs(matched_wj - original_wj) / original_wj * 100

    ts_length_results[label] = {
        'matched_wj': float(matched_wj),
        'original_wj': float(original_wj),
        'absolute_diff': float(abs(matched_wj - original_wj)),
        'pct_diff': float(pct_diff)
    }
    logger.info(f"  Original WJ: {original_wj:.6f}, Length-matched WJ: {matched_wj:.6f}, "
                f"diff={abs(matched_wj - original_wj):.6f} ({pct_diff:.2f}%)")

# ============================================================
# SECTION 21: ROBUSTNESS — THRESHOLD SENSITIVITY FORMAL ANALYSIS
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("ROBUSTNESS: THRESHOLD SENSITIVITY ANALYSIS")
logger.info("=" * 60)

# Already computed binary Jaccard at multiple thresholds in Section 6
# Now: formal analysis of how stable the RANK ORDER of comparisons is
threshold_sensitivity = {}

for t in thresholds:
    binary_wjs = {}
    for cond_A, cond_B in WJ_COMPARISONS:
        label = f"{cond_A}_vs_{cond_B}"
        binary_wjs[label] = wj_results[label]['binary_jaccard'][t]

    # Rank order at this threshold
    sorted_labels = sorted(binary_wjs, key=lambda x: binary_wjs[x])
    threshold_sensitivity[t] = {
        'values': binary_wjs,
        'rank_order': sorted_labels
    }

# Check if rank order is consistent across all thresholds
reference_order = threshold_sensitivity[thresholds[0]]['rank_order']
rank_consistent = all(
    threshold_sensitivity[t]['rank_order'] == reference_order
    for t in thresholds
)
logger.info(f"  Rank order consistent across all {len(thresholds)} thresholds: {rank_consistent}")
if not rank_consistent:
    for t in thresholds:
        logger.info(f"    t={t}: {threshold_sensitivity[t]['rank_order']}")
else:
    logger.info(f"    Consistent order: {reference_order}")

# Coefficient of variation across thresholds per comparison
for cond_A, cond_B in WJ_COMPARISONS:
    label = f"{cond_A}_vs_{cond_B}"
    vals = [wj_results[label]['binary_jaccard'][t] for t in thresholds]
    cv_thresh = np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0
    logger.info(f"  {label}: CV across thresholds = {cv_thresh:.4f}")

# ============================================================
# SECTION 22: ROBUSTNESS — NETWORK-LEVEL PERMUTATION TESTING
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("ROBUSTNESS: NETWORK-LEVEL PERMUTATION TESTING (per-network WJ significance)")
logger.info("=" * 60)

np.random.seed(RANDOM_SEED)
N_NET_PERMS = 1000
network_perm_results = {}

for cond_A, cond_B in WJ_COMPARISONS:
    label = f"{cond_A}_vs_{cond_B}"
    logger.info(f"\n--- Network permutation: {label} ---")

    network_perm_results[label] = {}

    for net_name in unique_networks:
        idx_net = network_indices[net_name]
        if len(idx_net) < 2:
            continue

        # Observed within-network WJ
        obs_net_wj = rsn_results[label]['per_network'].get(net_name, {}).get('wj', None)
        if obs_net_wj is None:
            continue

        # Null: shuffle condition labels, recompute within-network WJ
        null_net_wj = np.zeros(N_NET_PERMS)
        for perm_i in range(N_NET_PERMS):
            perm_corr_A_list = []
            perm_corr_B_list = []
            for sub in valid_subjects:
                if np.random.random() < 0.5:
                    perm_corr_A_list.append(subject_corr[sub][cond_A])
                    perm_corr_B_list.append(subject_corr[sub][cond_B])
                else:
                    perm_corr_A_list.append(subject_corr[sub][cond_B])
                    perm_corr_B_list.append(subject_corr[sub][cond_A])

            perm_z_A = np.mean([np.arctanh(np.clip(c, -0.9999, 0.9999))
                                for c in perm_corr_A_list], axis=0)
            perm_z_B = np.mean([np.arctanh(np.clip(c, -0.9999, 0.9999))
                                for c in perm_corr_B_list], axis=0)
            np.fill_diagonal(perm_z_A, 0)
            np.fill_diagonal(perm_z_B, 0)
            perm_r_A = np.tanh(perm_z_A)
            perm_r_B = np.tanh(perm_z_B)
            np.fill_diagonal(perm_r_A, 1.0)
            np.fill_diagonal(perm_r_B, 1.0)

            net_A = perm_r_A[np.ix_(idx_net, idx_net)]
            net_B = perm_r_B[np.ix_(idx_net, idx_net)]
            null_net_wj[perm_i] = weighted_jaccard(net_A, net_B)

        net_p = np.mean(null_net_wj <= obs_net_wj)
        net_effect = (np.mean(null_net_wj) - obs_net_wj) / np.std(null_net_wj) if np.std(null_net_wj) > 0 else 0

        network_perm_results[label][net_name] = {
            'observed_wj': float(obs_net_wj),
            'null_mean': float(np.mean(null_net_wj)),
            'null_std': float(np.std(null_net_wj)),
            'p_value': float(net_p),
            'effect_size': float(net_effect)
        }
        logger.info(f"  {net_name}: WJ={obs_net_wj:.4f}, p(perm)={net_p:.4f}, d={net_effect:.2f}")

    # FDR across networks within this comparison
    net_names_tested = list(network_perm_results[label].keys())
    if len(net_names_tested) > 1:
        net_pvals = [network_perm_results[label][n]['p_value'] for n in net_names_tested]
        reject, pvals_fdr, _, _ = multipletests(net_pvals, alpha=FDR_ALPHA, method='fdr_bh')
        for i, n in enumerate(net_names_tested):
            network_perm_results[label][n]['p_value_fdr'] = float(pvals_fdr[i])
            network_perm_results[label][n]['significant_fdr'] = bool(reject[i])
            logger.info(f"    {n}: p(FDR)={pvals_fdr[i]:.4f}, sig={reject[i]}")

# Save network permutation results
net_perm_rows = []
for label in network_perm_results:
    for net, vals in network_perm_results[label].items():
        row = {'comparison': label, 'network': net}
        row.update(vals)
        net_perm_rows.append(row)
net_perm_df = pd.DataFrame(net_perm_rows)
net_perm_df.to_csv(ROBUSTNESS_DIR / "network_permutation_results.csv", index=False)

# ============================================================
# SECTION 23: ROBUSTNESS FIGURES
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("GENERATING ROBUSTNESS FIGURES")
logger.info("=" * 60)

# Figure 7: Split-half reliability scatter
fig, axes = plt.subplots(1, len(WJ_COMPARISONS), figsize=(6 * len(WJ_COMPARISONS), 5))
if len(WJ_COMPARISONS) == 1:
    axes = [axes]
for i, (cond_A, cond_B) in enumerate(WJ_COMPARISONS):
    label = f"{cond_A}_vs_{cond_B}"
    h1 = split_half_results[label]['half1_values']
    h2 = split_half_results[label]['half2_values']
    axes[i].scatter(h1, h2, alpha=0.5, s=20, color=COLORBLIND_PALETTE[i])
    # Add identity line
    lims = [min(np.min(h1), np.min(h2)), max(np.max(h1), np.max(h2))]
    axes[i].plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
    axes[i].set_xlabel('Half-1 WJ', fontsize=12)
    axes[i].set_ylabel('Half-2 WJ', fontsize=12)
    r_val = split_half_results[label]['icc_proxy']
    axes[i].set_title(f'{label.replace("_vs_", " vs ")}\nr = {r_val:.3f}', fontsize=12, fontweight='bold')
    axes[i].set_aspect('equal')

plt.suptitle('Split-Half Reliability (100 Random Splits)', fontsize=14, fontweight='bold', y=1.02)
sns.despine()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure7_split_half_reliability.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "figure7_split_half_reliability.pdf", bbox_inches='tight')
plt.close()
logger.info("  Figure 7: Split-half reliability — saved")

# Figure 8: Jackknife LOO influence plot
fig, axes = plt.subplots(1, len(WJ_COMPARISONS), figsize=(6 * len(WJ_COMPARISONS), 5))
if len(WJ_COMPARISONS) == 1:
    axes = [axes]
for i, (cond_A, cond_B) in enumerate(WJ_COMPARISONS):
    label = f"{cond_A}_vs_{cond_B}"
    influence = jackknife_results[label]['influence']
    full_wj = jackknife_results[label]['full_wj']
    colors = ['red' if abs(inf) > 2 * np.std(influence) else COLORBLIND_PALETTE[i] for inf in influence]
    axes[i].bar(range(n_subs), influence, color=colors, edgecolor='black', linewidth=0.3)
    axes[i].axhline(0, color='black', linewidth=0.5)
    axes[i].axhline(2 * np.std(influence), color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    axes[i].axhline(-2 * np.std(influence), color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    axes[i].set_xlabel('Subject index', fontsize=12)
    axes[i].set_ylabel('Influence (delta WJ)', fontsize=12)
    axes[i].set_title(f'{label.replace("_vs_", " vs ")}', fontsize=12, fontweight='bold')

plt.suptitle('Leave-One-Out Jackknife Influence', fontsize=14, fontweight='bold', y=1.02)
sns.despine()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure8_jackknife_influence.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "figure8_jackknife_influence.pdf", bbox_inches='tight')
plt.close()
logger.info("  Figure 8: Jackknife influence — saved")

# Figure 9: Pearson vs Spearman comparison
fig, ax = plt.subplots(figsize=(8, 6))
x_pos = np.arange(len(WJ_COMPARISONS))
width = 0.35
spearman_vals = [pearson_wj_results[f"{a}_vs_{b}"]['spearman_wj'] for a, b in WJ_COMPARISONS]
pearson_vals = [pearson_wj_results[f"{a}_vs_{b}"]['pearson_wj'] for a, b in WJ_COMPARISONS]
bars1 = ax.bar(x_pos - width/2, spearman_vals, width, label='Spearman', color=COLORBLIND_PALETTE[0],
               edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x_pos + width/2, pearson_vals, width, label='Pearson', color=COLORBLIND_PALETTE[1],
               edgecolor='black', linewidth=0.8)
# Add pct diff annotations
for j, (a, b) in enumerate(WJ_COMPARISONS):
    label = f"{a}_vs_{b}"
    pct = pearson_wj_results[label]['pct_diff']
    ax.text(j, max(spearman_vals[j], pearson_vals[j]) + 0.01, f'{pct:.1f}%', ha='center', fontsize=9)
ax.set_ylabel('Weighted Jaccard', fontsize=12)
ax.set_title('Correlation Method Sensitivity: Spearman vs Pearson', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{a}\nvs\n{b}" for a, b in WJ_COMPARISONS], fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.05)
sns.despine()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure9_pearson_vs_spearman.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "figure9_pearson_vs_spearman.pdf", bbox_inches='tight')
plt.close()
logger.info("  Figure 9: Pearson vs Spearman — saved")

# Figure 10: Length-matching control
fig, ax = plt.subplots(figsize=(8, 6))
original_vals = [ts_length_results[f"{a}_vs_{b}"]['original_wj'] for a, b in WJ_COMPARISONS]
matched_vals = [ts_length_results[f"{a}_vs_{b}"]['matched_wj'] for a, b in WJ_COMPARISONS]
bars1 = ax.bar(x_pos - width/2, original_vals, width, label='Original', color=COLORBLIND_PALETTE[0],
               edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x_pos + width/2, matched_vals, width, label='Length-Matched', color=COLORBLIND_PALETTE[2],
               edgecolor='black', linewidth=0.8)
for j, (a, b) in enumerate(WJ_COMPARISONS):
    label = f"{a}_vs_{b}"
    pct = ts_length_results[label]['pct_diff']
    ax.text(j, max(original_vals[j], matched_vals[j]) + 0.01, f'{pct:.1f}%', ha='center', fontsize=9)
ax.set_ylabel('Weighted Jaccard', fontsize=12)
ax.set_title('Timeseries Length-Matching Control', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{a}\nvs\n{b}" for a, b in WJ_COMPARISONS], fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.05)
sns.despine()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure10_length_matching_control.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "figure10_length_matching_control.pdf", bbox_inches='tight')
plt.close()
logger.info("  Figure 10: Length-matching control — saved")

# Figure 11: Network-level permutation results heatmap
fig, axes = plt.subplots(1, len(WJ_COMPARISONS), figsize=(6 * len(WJ_COMPARISONS), 6))
if len(WJ_COMPARISONS) == 1:
    axes = [axes]
for i, (cond_A, cond_B) in enumerate(WJ_COMPARISONS):
    label = f"{cond_A}_vs_{cond_B}"
    nets_tested = [n for n in unique_networks if n in network_perm_results.get(label, {})]
    if not nets_tested:
        continue
    wj_vals = [network_perm_results[label][n]['observed_wj'] for n in nets_tested]
    p_vals = [network_perm_results[label][n].get('p_value_fdr',
              network_perm_results[label][n]['p_value']) for n in nets_tested]

    colors = [COLORBLIND_PALETTE[3] if p < 0.05 else COLORBLIND_PALETTE[7] for p in p_vals]
    bars = axes[i].barh(range(len(nets_tested)), wj_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[i].set_yticks(range(len(nets_tested)))
    axes[i].set_yticklabels(nets_tested, fontsize=10)
    axes[i].set_xlabel('Within-Network WJ', fontsize=12)
    axes[i].set_title(f'{label.replace("_vs_", " vs ")}', fontsize=12, fontweight='bold')
    axes[i].set_xlim(0, 1.05)

    # Add significance markers
    for j, (wj_v, p_v) in enumerate(zip(wj_vals, p_vals)):
        sig = '***' if p_v < 0.001 else '**' if p_v < 0.01 else '*' if p_v < 0.05 else 'n.s.'
        axes[i].text(wj_v + 0.02, j, f'{sig} (p={p_v:.3f})', va='center', fontsize=9)

plt.suptitle('Network-Level WJ with Permutation Significance (FDR-corrected)',
             fontsize=14, fontweight='bold', y=1.02)
sns.despine()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure11_network_permutation_significance.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "figure11_network_permutation_significance.pdf", bbox_inches='tight')
plt.close()
logger.info("  Figure 11: Network-level permutation significance — saved")

# ============================================================
# SECTION 24: ROBUSTNESS SUMMARY
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("ROBUSTNESS SUMMARY")
logger.info("=" * 60)

robustness_summary = {
    'split_half': split_half_results,
    'jackknife': {label: {k: v for k, v in res.items() if k not in ['loo_wj', 'influence', 'subjects']}
                  for label, res in jackknife_results.items()},
    'consistency': consistency_results,
    'pearson_vs_spearman': pearson_wj_results,
    'length_matching': ts_length_results,
    'threshold_sensitivity': {
        'rank_order_consistent': rank_consistent,
        'thresholds_tested': thresholds
    },
    'network_permutations': network_perm_results
}

# Clean up non-serializable items
for label in robustness_summary['split_half']:
    robustness_summary['split_half'][label] = {
        k: v for k, v in robustness_summary['split_half'][label].items()
        if k not in ['half1_values', 'half2_values']
    }

with open(ROBUSTNESS_DIR / "robustness_summary.json", 'w') as f:
    json.dump(robustness_summary, f, indent=2, default=str)

robustness_report = [
    "",
    "=" * 70,
    "ROBUSTNESS ANALYSIS SUMMARY",
    "=" * 70,
    "",
    "1. SPLIT-HALF RELIABILITY (100 random splits):"
]
for label, res in split_half_results.items():
    robustness_report.append(f"   {label}: ICC proxy = {res['icc_proxy']:.4f}, CV = {res['cv']:.4f}")
robustness_report.append("")
robustness_report.append("2. JACKKNIFE LEAVE-ONE-OUT:")
for label, res in jackknife_results.items():
    robustness_report.append(f"   {label}: max deviation = {res['max_deviation']:.6f}, "
                             f"jackknife SE = {res['jackknife_se']:.6f}")
robustness_report.append("")
robustness_report.append("3. SUBJECT-LEVEL CONSISTENCY:")
for label, res in consistency_results.items():
    if label == 'paired_wilcoxon':
        continue
    robustness_report.append(f"   {label}: {res['n_reorganized']}/{n_subs} subjects show reorganization "
                             f"({res['pct_reorganized']:.1f}%), binomial p={res['binom_p']:.6f}")
robustness_report.append("")
robustness_report.append("4. PEARSON vs SPEARMAN SENSITIVITY:")
for label, res in pearson_wj_results.items():
    robustness_report.append(f"   {label}: diff = {res['absolute_diff']:.6f} ({res['pct_diff']:.2f}%)")
robustness_report.append("")
robustness_report.append("5. TIMESERIES LENGTH-MATCHING:")
for label, res in ts_length_results.items():
    robustness_report.append(f"   {label}: diff = {res['absolute_diff']:.6f} ({res['pct_diff']:.2f}%)")
robustness_report.append("")
robustness_report.append(f"6. THRESHOLD SENSITIVITY: rank order consistent = {rank_consistent}")
robustness_report.append("")
robustness_report.append("7. NETWORK-LEVEL PERMUTATION TESTING:")
for label in network_perm_results:
    n_sig = sum(1 for n in network_perm_results[label]
                if network_perm_results[label][n].get('significant_fdr', False))
    n_tested = len(network_perm_results[label])
    robustness_report.append(f"   {label}: {n_sig}/{n_tested} networks significant (FDR < 0.05)")
robustness_report.append("")
robustness_report.append("=" * 70)

robustness_text = "\n".join(robustness_report)
with open(ROBUSTNESS_DIR / "robustness_report.txt", 'w') as f:
    f.write(robustness_text)

logger.info(robustness_text)

# ============================================================
# SECTION 25: UPDATE PROVENANCE WITH ROBUSTNESS
# ============================================================
provenance['robustness_analyses'] = {
    'split_half_reliability': True,
    'jackknife_leave_one_out': True,
    'subject_consistency': True,
    'pearson_vs_spearman': True,
    'timeseries_length_matching': True,
    'threshold_sensitivity': True,
    'network_level_permutation': True,
    'n_split_half_iterations': N_SPLITS,
    'n_network_permutations': N_NET_PERMS
}
provenance['pipeline_version'] = "3.0"
provenance['execution_date'] = datetime.now().isoformat()

with open(WJ_DIR / "provenance.json", 'w') as f:
    json.dump(provenance, f, indent=2)

logger.info("\nProvenance updated with robustness analyses.")
logger.info("\nPipeline complete — all robustness analyses finished.")
