# Lexical Familiarity Predicts Processing Depth for Nonliteral Language in Large Language Models

> **Anonymity Notice:** This repository is fully anonymized for double-blind peer review. All author identities, affiliations, and specific institutional acknowledgments have been removed.

## Overview

This repository contains the datasets, code, and analysis results for the experiments from our paper, *"Lexical Familiarity Predicts Processing Depth for Nonliteral Language in Large Language Models."* Our research investigates how Large Language Models (LLMs) process language that deviates from literal, standard usage. By analyzing layer-wise representations using Sparse Autoencoders (SAEs), we establish a "lexical familiarity gradient" across five categories of nonliteral language.

The main experiments use **Gemma-3-12B-IT** with Gemma-Scope-2 SAEs. We additionally include model-generalization experiments on **Gemma-3-12B-PT** (pre-trained base) and **Qwen3.5-9B-Base** (with Qwen-Scope SAEs) to verify that the lexical familiarity gradient is not specific to a single model or to instruction tuning.

## Repository Structure

```
.
├── data/
│   ├── idioms.csv                          # Idioms dataset with position tags
│   ├── metaphors.csv                       # Metaphors dataset with position tags
│   ├── slang_neologisms.csv                # Neologisms with position tags
│   ├── slang_semantic_shift.csv            # Semantic shift pairs with position tags
│   ├── slang_constructional.csv            # Constructional slang with position tags
│   ├── literal_paraphrase.csv              # Baseline literal language data
│   └── identical_sentence_pair.csv         # Control pairs
├── scripts/
│   ├── README.md                           # Script usage instructions
│   ├── Gemma-12B-IT/                       # Main experiments (Gemma-3-12B-IT + Gemma-Scope-2)
│   │   ├── run.py                          # Main SAE feature extraction and analysis
│   │   ├── lf_full.py                      # Lexical familiarity score (full, tokenizer-based)
│   │   ├── lf_simplified.py                # Lexical familiarity score (simplified)
│   │   ├── normal_approx_ci.py             # 95% CI computation for results
│   │   └── span_length.py                  # Divergence span length confound analysis
│   ├── Gemma-12B-PT/                       # Generalization: Gemma-3-12B pre-trained base
│   │   └── run_pt.py                       # SAE analysis on Gemma-3-12B-PT
│   └── Qwen-9B-Base/                       # Generalization: Qwen3.5-9B-Base
│       └── run_qwen.py                     # SAE analysis on Qwen3.5-9B-Base (Qwen-Scope SAEs)
├── results/
│   ├── Gemma-12B-IT/                       # Main results (per category + CI + LF analysis)
│   │   ├── idiom/, metaphor/, neo/, semantic/, const/
│   │   ├── identical/, literal-paraphrase/
│   │   ├── feature_ratio/
│   │   ├── ci/                             # Cosine distances with 95% CIs, paper tables/figures
│   │   └── lexical_familiarity/            # Per-item LF scores, regression figure, LaTeX table
│   ├── Gemma-12B-PT/                       # Same layout as above, for Gemma-3-12B-PT
│   └── Qwen-9B-Base/                       # Same layout as above, for Qwen3.5-9B-Base
├── requirements.txt                        # Python package dependencies
└── README.md
```

Within each model's `results/<category>/` folder you will find layer-wise SAE feature activations, cosine-distance curves, and (for PT/Qwen) per-layer checkpoint CSVs and diagnostic plots produced by the corresponding `run_*.py` script.

## Data Format

Each dataset CSV file contains sentence pairs with position annotations marking the divergence span between figurative and literal versions:

- **Figurative/nonliteral text**: The slang, idiom, metaphor, or figurative expression
- **Literal text**: The literal or paraphrased equivalent
- **Position tags**: Start and end word indices of the relevant words/phrases in both versions (as lists of `[start, end)` tuples)
- **Segment information**: The specific figurative words/phrases and their literal counterparts

Column names vary by category (e.g., `idiomatic`/`normal` for idioms, `metaphorical`/`normal` for metaphors, `gen_z`/`normal` for slang).

## Scripts

### Main experiments — `scripts/Gemma-12B-IT/`

#### `run.py` — Main SAE Analysis

The primary analysis script that:

1. **Loads datasets** from the `data/` directory
2. **Extracts SAE features** from Gemma-Scope-2 Sparse Autoencoders for each sentence pair
3. **Computes layer-wise representations** across all 48 layers of the model
4. **Analyzes feature distributions** to identify the lexical familiarity gradient
5. **Generates results** and saves them to `results/Gemma-12B-IT/`

#### `lf_full.py` — Lexical Familiarity (Full)

Computes the lexical familiarity (LF) score for each item using the actual Gemma tokenizer. The LF score combines:

- **Subword fragmentation index**: number of subword tokens / number of whitespace words (higher = less familiar)
- **Token frequency percentile**: average frequency rank of span tokens (higher = more frequent)
- **Combined LF score**: `LF = −z(fragmentation) + z(frequency)` (higher = more familiar)

Also runs regression analysis (`peak_layer ~ lf_score`) and generates publication-quality figures and LaTeX tables.

#### `lf_simplified.py` — Lexical Familiarity (Simplified)

A streamlined version of `lf_full.py` using the same tokenizer-based computation but with a simpler interface. Suitable for quick reproduction of the key regression results.

#### `normal_approx_ci.py` — Confidence Intervals

Computes 95% confidence intervals for layer-wise cosine distances using the normal approximation formula (`CI = mean ± 1.96 × SE`). Generates CI tables and LaTeX output for the paper.

#### `span_length.py` — Span Length Confound Analysis

Checks whether divergence span length (in words) is a confound for the observed peak cosine distances. Runs correlation and regression analyses to verify that the lexical familiarity gradient holds independently of span length.

### Model-generalization experiments

#### `scripts/Gemma-12B-PT/run_pt.py`

Re-runs the token-level SAE analysis on **Gemma-3-12B-PT** (the pre-trained base model, no instruction tuning) using the same Gemma-Scope-2 SAEs. Outputs are written to `results/Gemma-12B-PT/`, including per-layer checkpoint CSVs and feature-activation comparison plots.

#### `scripts/Qwen-9B-Base/run_qwen.py`

Replicates the analysis on **Qwen3.5-9B-Base** using the Top-K residual Qwen-Scope SAE release `qwen-scope-3.5-9b-base-w64k-l100` (K=100, width 65,536; HF: `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100`). Iterates over all datasets in a single run so the model is loaded only once. No HF token is required. Outputs are written to `results/Qwen-9B-Base/`.

## Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended for GPU acceleration)
- Hugging Face Transformers library
- SAE-Lens library for working with Sparse Autoencoders
- NumPy, Pandas, SciPy, statsmodels for data processing and statistics
- Matplotlib & Seaborn for visualization
- A Hugging Face API token (required to access Gemma models; not required for Qwen)

### Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your Hugging Face token** (Gemma only):
   Create a `.env` file in the repository root:
   ```
   HF_TOKEN=your_huggingface_token_here
   ```
   A token can be obtained from https://huggingface.co/settings/tokens

3. **Run the main SAE analysis (Gemma-3-12B-IT):**
   ```bash
   cd scripts/Gemma-12B-IT
   python run.py
   ```

4. **Compute lexical familiarity scores:**
   ```bash
   # Full analysis (recommended for reproduction)
   python scripts/Gemma-12B-IT/lf_full.py --data-dir ./data --output ./results/Gemma-12B-IT/lexical_familiarity

   # Simplified analysis
   python scripts/Gemma-12B-IT/lf_simplified.py --data-dir ./data --output ./results/Gemma-12B-IT/lexical_familiarity
   ```

5. **Compute confidence intervals:**
   ```bash
   python scripts/Gemma-12B-IT/normal_approx_ci.py
   ```

6. **(Optional) Run the model-generalization experiments:**
   ```bash
   # Gemma-3-12B-PT
   python scripts/Gemma-12B-PT/run_pt.py

   # Qwen3.5-9B-Base
   python scripts/Qwen-9B-Base/run_qwen.py
   ```

## Results

Each `results/<model>/` directory contains pre-computed outputs with the same layout:

- **Per-category folders** (`idiom/`, `metaphor/`, `neo/`, `semantic/`, `const/`, `identical/`, `literal-paraphrase/`): layer-wise SAE feature activations and cosine distances between figurative and literal representations.
- **`feature_ratio/`**: figurative-to-literal feature ratio analyses.
- **`ci/`**: cosine distance curves with 95% confidence intervals, CI overlap analysis, and LaTeX-formatted tables (Table 2 and Table 3 from the paper).
- **`lexical_familiarity/`** (Gemma-12B-IT) / **`lexical_familarity/`** (PT, Qwen): per-item LF scores, regression figures, and the LaTeX table for the lexical familiarity analysis section.

### Key Findings (Gemma-3-12B-IT, from `results/Gemma-12B-IT/lexical_familiarity/README.md`)

| Category | LF Score | Peak Layer | N |
|---|---|---|---|
| Idiom | +1.28 | L1 | 823 |
| Construction† | +0.44 | L7 | 37 |
| Metaphor | +0.45 | L8 | 625 |
| Semantic Shift | +0.02 | L9 | 1002 |
| Neologism | −1.37 | L41 | 1000 |

Lexical familiarity score significantly predicts peak divergence layer (β = −5.88, p < .001, R² = 0.342). At the category level, mean LF score correlates near-perfectly with peak layer (r = −0.95). The same gradient is reproduced on Gemma-3-12B-PT and Qwen3.5-9B-Base — see `results/Gemma-12B-PT/` and `results/Qwen-9B-Base/`.
