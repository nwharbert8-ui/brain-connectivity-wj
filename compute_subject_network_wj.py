"""
Pipeline: Per-Subject Network-Level WJ Analysis
Author: Drake H. Harbert (D.H.H.)
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616
Date: 2026-03-13
Description: Computes within-network WJ for each subject individually across
    all 8 canonical networks and 3 pairwise comparisons. This makes the
    network-level analysis subject-level primary, consistent with the global
    WJ restructuring. Group statistics are derived FROM the subject-level
    distributions, not from group-averaged matrices.
Dependencies: numpy, pandas, scipy
Input: Per-subject correlation matrices (.npy) from brain_connectivity_wj_pipeline.py
Output: subject_network_wj.csv, subject_network_wj_stats.csv, subject_network_recovery.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json
from datetime import datetime

# =============================================================================
# CONFIG
# =============================================================================
FORCE_RECOMPUTE = True
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).parent
CORR_DIR = BASE_DIR / "results" / "correlation_matrices"
ATLAS_FILE = (BASE_DIR / "data" / "raw" / "ds006623" / "derivatives" /
              "xcp_d_without_GSR_bandpass_output" / "atlases" /
              "atlas-4S456Parcels" / "atlas-4S456Parcels_dseg.tsv")
OUTPUT_DIR = BASE_DIR / "results" / "wj"
SUPPLEMENT_DIR = OUTPUT_DIR / "manuscript_supplements"

CONDITIONS = ['awake', 'unconscious', 'recovery']
COMPARISONS = [
    ('awake', 'unconscious'),
    ('awake', 'recovery'),
    ('unconscious', 'recovery'),
]

# =============================================================================
# FUNCTIONS
# =============================================================================

def weighted_jaccard(corr_A, corr_B, indices=None):
    """Compute WJ between two correlation matrices (or submatrices)."""
    if indices is not None:
        sub_A = corr_A[np.ix_(indices, indices)]
        sub_B = corr_B[np.ix_(indices, indices)]
    else:
        sub_A = corr_A
        sub_B = corr_B

    idx = np.triu_indices(sub_A.shape[0], k=1)
    vec_A = np.abs(sub_A[idx])
    vec_B = np.abs(sub_B[idx])

    numerator = np.sum(np.minimum(vec_A, vec_B))
    denominator = np.sum(np.maximum(vec_A, vec_B))

    if denominator == 0:
        return 1.0
    return numerator / denominator


def load_subject_corr(subject, condition):
    """Load a cached per-subject correlation matrix."""
    path = CORR_DIR / f"{subject}_{condition}_spearman_corr.npy"
    if not path.exists():
        return None
    return np.load(path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PER-SUBJECT NETWORK-LEVEL WJ ANALYSIS")
    print("=" * 70)

    # Load atlas labels for network assignment
    print("\nLoading atlas labels...")
    atlas = pd.read_csv(ATLAS_FILE, sep='\t')
    atlas['network_label'] = atlas['network_label'].fillna('Subcortical')
    atlas.loc[atlas['network_label'] == 'n/a', 'network_label'] = 'Subcortical'
    networks = sorted(atlas['network_label'].unique())
    print(f"  Networks: {networks}")

    # Build network index map
    network_indices = {}
    for net in networks:
        network_indices[net] = atlas.index[atlas['network_label'] == net].tolist()
        print(f"  {net}: {len(network_indices[net])} ROIs")

    # Discover valid subjects
    corr_files = list(CORR_DIR.glob("sub-*_awake_spearman_corr.npy"))
    subjects = sorted(set(f.name.split('_')[0] for f in corr_files))
    print(f"\nFound {len(subjects)} subjects: {subjects[:5]}...")

    # Compute per-subject, per-network WJ
    print("\nComputing per-subject network WJ...")
    results = []

    for sub in subjects:
        # Load all 3 condition matrices for this subject
        corr = {}
        skip = False
        for cond in CONDITIONS:
            mat = load_subject_corr(sub, cond)
            if mat is None:
                print(f"  WARNING: Missing {sub}_{cond}, skipping subject")
                skip = True
                break
            corr[cond] = mat
        if skip:
            continue

        for cond_a, cond_b in COMPARISONS:
            comp_label = f"{cond_a}_vs_{cond_b}"

            # Global WJ for this subject (sanity check)
            global_wj = weighted_jaccard(corr[cond_a], corr[cond_b])

            for net in networks:
                idx = network_indices[net]
                net_wj = weighted_jaccard(corr[cond_a], corr[cond_b], indices=idx)
                n_pairs = len(idx) * (len(idx) - 1) // 2

                results.append({
                    'subject': sub,
                    'comparison': comp_label,
                    'network': net,
                    'wj': net_wj,
                    'n_rois': len(idx),
                    'n_pairs': n_pairs,
                    'global_wj': global_wj,
                })

    df = pd.DataFrame(results)
    print(f"  Computed {len(df)} subject x network x comparison WJ values")

    # Save raw per-subject network WJ
    raw_path = SUPPLEMENT_DIR / "subject_network_wj.csv"
    df.to_csv(raw_path, index=False)
    print(f"\n  Saved: {raw_path}")

    # ==========================================================================
    # GROUP STATISTICS FROM SUBJECT-LEVEL DISTRIBUTIONS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("GROUP STATISTICS (derived from subject-level distributions)")
    print("=" * 70)

    stats_rows = []
    for comp in [f"{a}_vs_{b}" for a, b in COMPARISONS]:
        print(f"\n--- {comp} ---")
        for net in networks:
            vals = df[(df['comparison'] == comp) & (df['network'] == net)]['wj'].values
            n = len(vals)
            mean_wj = vals.mean()
            sd_wj = vals.std()
            median_wj = np.median(vals)

            # Jackknife SE
            jk_se = np.sqrt((n - 1) / n * np.sum((vals - mean_wj) ** 2))

            # Wilcoxon test: WJ < 1.0
            w_stat, w_p = stats.wilcoxon(vals - 1.0, alternative='less')

            # Effect size: Cohen's d vs 1.0
            d = (1.0 - mean_wj) / sd_wj if sd_wj > 0 else np.nan

            # Count subjects showing reorganization
            n_reorg = np.sum(vals < 1.0)

            print(f"  {net:15s}: mean={mean_wj:.3f} SD={sd_wj:.3f} "
                  f"JK_SE={jk_se:.3f} W={w_stat:.0f} p={w_p:.2e} "
                  f"d={d:.2f} ({n_reorg}/{n} subjects)")

            stats_rows.append({
                'comparison': comp,
                'network': net,
                'mean_wj': round(mean_wj, 4),
                'sd': round(sd_wj, 4),
                'median_wj': round(median_wj, 4),
                'jackknife_se': round(jk_se, 4),
                'min_wj': round(vals.min(), 4),
                'max_wj': round(vals.max(), 4),
                'wilcoxon_W': w_stat,
                'wilcoxon_p': w_p,
                'effect_size_d': round(d, 2),
                'n_subjects_reorganized': int(n_reorg),
                'n_subjects_total': n,
                'n_rois': df[(df['comparison'] == comp) & (df['network'] == net)]['n_rois'].iloc[0],
            })

    stats_df = pd.DataFrame(stats_rows)

    # FDR correction across all tests within each comparison
    from statsmodels.stats.multitest import multipletests
    for comp in stats_df['comparison'].unique():
        mask = stats_df['comparison'] == comp
        p_vals = stats_df.loc[mask, 'wilcoxon_p'].values
        reject, p_fdr, _, _ = multipletests(p_vals, method='fdr_bh')
        stats_df.loc[mask, 'p_fdr'] = p_fdr
        stats_df.loc[mask, 'significant_fdr'] = reject

    stats_path = SUPPLEMENT_DIR / "subject_network_wj_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n  Saved: {stats_path}")

    # ==========================================================================
    # SUBJECT-LEVEL NETWORK RECOVERY FRACTIONS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUBJECT-LEVEL NETWORK RECOVERY FRACTIONS")
    print("=" * 70)

    recovery_rows = []
    for net in networks:
        aw_un = df[(df['comparison'] == 'awake_vs_unconscious') & (df['network'] == net)]
        aw_re = df[(df['comparison'] == 'awake_vs_recovery') & (df['network'] == net)]

        # Merge on subject
        merged = aw_un[['subject', 'wj']].rename(columns={'wj': 'wj_aw_un'}).merge(
            aw_re[['subject', 'wj']].rename(columns={'wj': 'wj_aw_re'}),
            on='subject'
        )
        merged['recovery_fraction'] = (
            (merged['wj_aw_re'] - merged['wj_aw_un']) /
            (1.0 - merged['wj_aw_un'])
        )

        rf = merged['recovery_fraction'].values
        mean_rf = rf.mean()
        sd_rf = rf.std()

        # Wilcoxon: recovery fraction > 0?
        if np.all(rf == 0):
            w_rf, p_rf = np.nan, 1.0
        else:
            w_rf, p_rf = stats.wilcoxon(rf, alternative='greater')

        print(f"  {net:15s}: mean RF={mean_rf:.3f} SD={sd_rf:.3f} "
              f"W={w_rf:.0f} p={p_rf:.3f}")

        recovery_rows.append({
            'network': net,
            'mean_recovery_fraction': round(mean_rf, 4),
            'sd': round(sd_rf, 4),
            'median_recovery_fraction': round(np.median(rf), 4),
            'wilcoxon_W': w_rf,
            'wilcoxon_p': round(p_rf, 4),
            'n_positive': int(np.sum(rf > 0)),
            'n_negative': int(np.sum(rf < 0)),
            'n_subjects': len(rf),
        })

        # Save per-subject recovery fractions
        for _, row in merged.iterrows():
            recovery_rows_detail = {
                'subject': row['subject'],
                'network': net,
                'wj_awake_unconscious': round(row['wj_aw_un'], 4),
                'wj_awake_recovery': round(row['wj_aw_re'], 4),
                'recovery_fraction': round(row['recovery_fraction'], 4),
            }

    recovery_df = pd.DataFrame(recovery_rows)
    recovery_path = SUPPLEMENT_DIR / "subject_network_recovery_stats.csv"
    recovery_df.to_csv(recovery_path, index=False)
    print(f"\n  Saved: {recovery_path}")

    # ==========================================================================
    # COMPARISON: SUBJECT-LEVEL vs GROUP-LEVEL NETWORK WJ
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUBJECT-LEVEL vs GROUP-LEVEL COMPARISON (awake_vs_unconscious)")
    print("=" * 70)

    # Load group-level network results for comparison
    group_net_path = OUTPUT_DIR / "rsn_network_wj.csv"
    if group_net_path.exists():
        group_net = pd.read_csv(group_net_path)
        group_avu = group_net[group_net['comparison'] == 'awake_vs_unconscious']

        print(f"\n  {'Network':15s} {'Subject-level':>14s} {'Group-level':>12s} {'Diff':>8s}")
        print(f"  {'-'*15} {'-'*14} {'-'*12} {'-'*8}")
        for net in networks:
            subj_mean = stats_df[
                (stats_df['comparison'] == 'awake_vs_unconscious') &
                (stats_df['network'] == net)
            ]['mean_wj'].values[0]

            group_row = group_avu[group_avu['network'] == net]
            if len(group_row) > 0:
                group_wj = group_row['wj'].values[0]
                diff = group_wj - subj_mean
                print(f"  {net:15s} {subj_mean:14.3f} {group_wj:12.3f} {diff:+8.3f}")
            else:
                print(f"  {net:15s} {subj_mean:14.3f} {'N/A':>12s}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
