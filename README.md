# Weighted Jaccard Similarity Analysis of Brain Connectivity Under Propofol Anesthesia

**Author:** Drake H. Harbert (D.H.H.)
**Affiliation:** Inner Architecture LLC, Canton, OH
**ORCID:** [0009-0007-7740-3616](https://orcid.org/0009-0007-7740-3616)
**Contact:** Drake@innerarchitecturellc.com

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19006864.svg)](https://doi.org/10.5281/zenodo.19006864)

## Overview

This repository contains the analysis code for quantifying functional connectivity
reorganization during propofol-induced unconsciousness and early recovery using
Weighted Jaccard (WJ) similarity analysis. WJ treats full pairwise correlation
matrices as weighted edge sets and computes their overlap, providing a single scalar
measure of architectural similarity between conditions without requiring binarization,
threshold selection, or hypothesis-driven connection filtering.

## Dataset

Data are from the Michigan Human Anesthesia fMRI Dataset-1, publicly available on
OpenNeuro:

- **Accession:** [ds006623](https://openneuro.org/datasets/ds006623)
- **Reference:** Huang et al. (2025). *Scientific Data* 12, 185. https://doi.org/10.1038/s41597-025-06442-2
- **Subjects:** 26 healthy adults (24 used after exclusion), graded propofol sedation
- **Preprocessing:** XCP-D pipeline with 4S456Parcels atlas (456 ROIs), no global signal regression, bandpass filtered

## Reproduction

### 1. Obtain the data

Download ds006623 from OpenNeuro and place the XCP-D preprocessed derivatives in
`data/raw/ds006623/derivatives/`.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the main pipeline

```bash
python brain_connectivity_wj_pipeline.py
```

This produces all primary results including:
- Per-subject Spearman correlation matrices (456 x 456) for each condition
- Subject-level and group-level WJ values for all pairwise condition comparisons
- Permutation testing (10,000 iterations) with FDR correction
- Bootstrap confidence intervals (10,000 iterations)
- Complete robustness battery (split-half, jackknife, Pearson sensitivity, length matching, threshold sensitivity, network-level permutation)
- All figures (Figures 1-11)

### 4. Run subject-level network analysis

```bash
python compute_subject_network_wj.py
```

Computes within-network WJ for each subject individually across all 8 canonical
resting-state networks, with group statistics derived from the subject-level
distributions.

### 5. Generate manuscript supplements

```bash
python brain_connectivity_manuscript_supplements_pipeline.py
```

Generates supplementary tables and formatted outputs for manuscript submission.

### 6. Generate submission-ready figures and tables

```bash
python generate_submission_figures_tables.py
```

### 7. Convert manuscript to .docx

```bash
python convert_manuscript_to_docx.py
```

## Key Findings

- **Every subject** shows ~50% reduction in connectivity similarity during propofol unconsciousness (mean WJ = 0.499, SD = 0.043)
- **Recovery is minimal:** only 3.8% of lost similarity recovers during early behavioral recovery
- **All 8 canonical networks** show significant similarity reduction (all p < 0.001, FDR-corrected)
- **Only the default mode network** shows significant early recovery (7.7%, p = 0.021)
- **Dose-response:** propofol concentration at loss of responsiveness correlates with dissimilarity magnitude (rho = -0.591, p = 0.002)
- **Somatomotor-visual interface** concentrates the largest individual edge disruptions (90% of top 50 edges)

## File Descriptions

| File | Description |
|------|-------------|
| `brain_connectivity_wj_pipeline.py` | Main analysis pipeline — reproduces all primary results, figures, and robustness analyses |
| `compute_subject_network_wj.py` | Subject-level within-network WJ analysis across 8 canonical networks |
| `brain_connectivity_manuscript_supplements_pipeline.py` | Supplementary tables (demographics, dose-response, edge statistics, cross-domain comparison) |
| `generate_submission_figures_tables.py` | Publication-formatted figures and tables for submission |
| `convert_manuscript_to_docx.py` | Converts manuscript .txt to formatted .docx (NeuroImage style) |

## Methodology

The Weighted Jaccard index between two correlation matrices is computed as:

**WJ(A, B) = Σ min(w_A, w_B) / Σ max(w_A, w_B)**

where w_A and w_B are the absolute correlation values for each edge. WJ = 1 indicates
identical architectures; WJ = 0 indicates complete topological independence.

All analyses use:
- Spearman rank correlation (Pearson sensitivity analysis included)
- FDR (Benjamini-Hochberg) correction across all tested comparisons
- Fixed random seed (42) for reproducibility
- Subject-level analysis as primary, group-averaged as supplementary

## License

MIT License

## Citation

If you use this code, please cite:

Harbert, D.H. (2026). Weighted Jaccard Similarity Analysis Reveals Incomplete Recovery
of Functional Connectivity Architecture Following Propofol-Induced Unconsciousness.
*[Journal]*, [volume], [pages].

Code archive: https://doi.org/10.5281/zenodo.19006864
