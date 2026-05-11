#!/usr/bin/env python3
"""
Lexical Familiarity Score Computation and Regression Analysis (Qwen3.5-9B).

Differences:
  - Tokenizer is Qwen/Qwen3.5-9B (no HF token required).
  - peak_map is auto-derived from run_qwen.py outputs in results/qwen/

Run after run_qwen.py has produced per-category CSVs in --results-dir.
"""

import argparse
import ast
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')


# =============================================================================
# TOKENIZER SETUP
# =============================================================================

def load_tokenizer(model_id: str = "Qwen/Qwen3.5-9B"):
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    return tokenizer


# =============================================================================
# LEXICAL FAMILIARITY COMPUTATION  (identical math to scripts/lf_full.py)
# =============================================================================

def compute_subword_fragmentation(text: str, positions: List[Tuple[int, int]],
                                  tokenizer) -> float:
    words = text.split()
    span_words = []
    for start, end in positions:
        span_words.extend(words[start:end])
    if not span_words:
        return 1.0
    span_text = ' '.join(span_words)
    n_whitespace_words = len(span_words)
    tokens = tokenizer.tokenize(span_text)
    n_subword_tokens = len(tokens)
    if n_whitespace_words == 0:
        return 1.0
    return n_subword_tokens / n_whitespace_words


def compute_token_frequency_score(text: str, positions: List[Tuple[int, int]],
                                  tokenizer) -> float:
    words = text.split()
    span_words = []
    for start, end in positions:
        span_words.extend(words[start:end])
    if not span_words:
        return 50.0
    span_text = ' '.join(span_words)
    token_ids = tokenizer.encode(span_text, add_special_tokens=False)
    if not token_ids:
        return 50.0
    vocab_size = tokenizer.vocab_size
    percentiles = [100 - (tid / vocab_size * 100) for tid in token_ids]
    return float(np.mean(percentiles))


def compute_lf_score(fragmentation: float, frequency: float,
                     frag_mean: float, frag_std: float,
                     freq_mean: float, freq_std: float) -> float:
    z_frag = (fragmentation - frag_mean) / frag_std if frag_std > 0 else 0
    z_freq = (frequency - freq_mean) / freq_std if freq_std > 0 else 0
    return -z_frag + z_freq


# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

def load_and_process_category(filepath: Path, category: str, tokenizer,
                              text_col: str, pos_col: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    records = []
    for idx, row in df.iterrows():
        try:
            positions = ast.literal_eval(row[pos_col])
            text = str(row[text_col])
            frag = compute_subword_fragmentation(text, positions, tokenizer)
            freq = compute_token_frequency_score(text, positions, tokenizer)
            records.append({
                'category': category,
                'item_idx': idx,
                'text': text[:100],
                'fragmentation': frag,
                'frequency_score': freq,
            })
        except Exception:
            continue
    return pd.DataFrame(records)


CATEGORY_CONFIGS = {
    'idiom':          {'file': 'idiom_baseline.csv',       'text_col': 'idiomatic',     'pos_col': 'idiom_positions'},
    'metaphor':       {'file': 'metaphor_baseline.csv',    'text_col': 'metaphorical',  'pos_col': 'metaphor_positions'},
    'semantic_shift': {'file': 'genz_dataset_tagged.csv',  'text_col': 'gen_z',         'pos_col': 'slang_positions'},
    'neologism':      {'file': 'neologism_tagged.csv',     'text_col': 'gen_z',         'pos_col': 'slang_positions'},
    'construction':   {'file': 'constructions_tagged.csv', 'text_col': 'gen_z',         'pos_col': 'slang_positions'},
}


def load_all_categories(data_dir: Path, tokenizer) -> pd.DataFrame:
    all_data = []
    for cat_name, config in CATEGORY_CONFIGS.items():
        filepath = data_dir / config['file']
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping {cat_name}")
            continue
        print(f"Processing {cat_name}...")
        cat_df = load_and_process_category(filepath, cat_name, tokenizer,
                                           config['text_col'], config['pos_col'])
        all_data.append(cat_df)
        print(f"  Loaded {len(cat_df)} items")

    combined = pd.concat(all_data, ignore_index=True)

    frag_mean, frag_std = combined['fragmentation'].mean(), combined['fragmentation'].std()
    freq_mean, freq_std = combined['frequency_score'].mean(), combined['frequency_score'].std()

    combined['lf_score'] = combined.apply(
        lambda row: compute_lf_score(
            row['fragmentation'], row['frequency_score'],
            frag_mean, frag_std, freq_mean, freq_std,
        ), axis=1,
    )
    combined['z_fragmentation'] = (combined['fragmentation'] - frag_mean) / frag_std
    combined['z_frequency'] = (combined['frequency_score'] - freq_mean) / freq_std

    return combined


# =============================================================================
# PEAK LAYER ASSIGNMENT  (auto-derived from run_qwen outputs)
# =============================================================================

def derive_peak_map_from_results(results_dir: Path, metric: str = 'cosine_dist') -> Dict[str, int]:
    """
    Read the latest run_qwen output for each category and find the peak layer
    on the requested metric.
    """
    peak_map: Dict[str, int] = {}
    for cat in CATEGORY_CONFIGS.keys():
        files = sorted(results_dir.glob(f"{cat}_qwen_*.csv"))
        if not files:
            print(f"Warning: no run_qwen CSV found for category '{cat}' in {results_dir}")
            continue
        latest = files[-1]
        df = pd.read_csv(latest)
        df_metric = df[df['metric'] == metric]
        if df_metric.empty:
            print(f"Warning: metric '{metric}' missing in {latest.name}")
            continue
        peak_layer = int(df_metric.loc[df_metric['mean'].idxmax(), 'layer'])
        peak_map[cat] = peak_layer
        print(f"  {cat}: peak layer = L{peak_layer} (from {latest.name})")
    return peak_map


def assign_peak_layers(df: pd.DataFrame, peak_map: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df['peak_layer'] = df['category'].map(peak_map)
    return df


# =============================================================================
# REGRESSION ANALYSIS  (identical to lf_full.py)
# =============================================================================

def run_regression_analysis(df: pd.DataFrame) -> Dict:
    results = {}

    print("\n" + "=" * 60)
    print("MODEL 1: peak_layer ~ category")
    print("=" * 60)
    model1 = smf.ols('peak_layer ~ C(category)', data=df).fit()
    print(model1.summary().tables[0])
    print(model1.summary().tables[1])
    results['model1_r2'] = model1.rsquared
    results['model1_aic'] = model1.aic

    print("\n" + "=" * 60)
    print("MODEL 2: peak_layer ~ lf_score")
    print("=" * 60)
    model2 = smf.ols('peak_layer ~ lf_score', data=df).fit()
    print(model2.summary().tables[0])
    print(model2.summary().tables[1])
    results['model2_r2'] = model2.rsquared
    results['model2_aic'] = model2.aic
    results['lf_coef'] = model2.params['lf_score']
    results['lf_pvalue'] = model2.pvalues['lf_score']

    print("\n" + "=" * 60)
    print("MODEL 3: peak_layer ~ category + lf_score")
    print("=" * 60)
    model3 = smf.ols('peak_layer ~ C(category) + lf_score', data=df).fit()
    print(model3.summary().tables[0])
    print(model3.summary().tables[1])
    results['model3_r2'] = model3.rsquared
    results['model3_aic'] = model3.aic
    results['lf_coef_with_cat'] = model3.params['lf_score']
    results['lf_pvalue_with_cat'] = model3.pvalues['lf_score']

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<30} {'R²':<10} {'AIC':<12}")
    print("-" * 52)
    print(f"{'Category only':<30} {results['model1_r2']:.4f}    {results['model1_aic']:.1f}")
    print(f"{'LF_score only':<30} {results['model2_r2']:.4f}    {results['model2_aic']:.1f}")
    print(f"{'Category + LF_score':<30} {results['model3_r2']:.4f}    {results['model3_aic']:.1f}")

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_lf_analysis(df: pd.DataFrame, output_dir: Path):
    colors = {
        'idiom': '#1f77b4', 'construction': '#ff7f0e', 'metaphor': '#2ca02c',
        'semantic_shift': '#d62728', 'neologism': '#9467bd',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    category_order = ['idiom', 'construction', 'metaphor', 'semantic_shift', 'neologism']
    df_ordered = df[df['category'].isin(category_order)]

    ax1 = axes[0, 0]
    sns.boxplot(data=df_ordered, x='category', y='lf_score',
                order=category_order, palette=colors, ax=ax1)
    ax1.set_xlabel('Category', fontsize=12)
    ax1.set_ylabel('Lexical Familiarity Score', fontsize=12)
    ax1.set_title('LF Score Distribution by Category (Qwen3.5-9B)', fontsize=14)
    ax1.set_xticklabels(['Idiom', 'Constr.†', 'Metaphor', 'Sem. Shift', 'Neologism'], rotation=15)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax2 = axes[0, 1]
    for cat in category_order:
        cat_data = df[df['category'] == cat]
        ax2.scatter(cat_data['lf_score'], cat_data['peak_layer'],
                    c=colors[cat], label=cat.replace('_', ' ').title(),
                    alpha=0.6, s=30)
    slope, intercept, r, p, se = stats.linregress(df['lf_score'], df['peak_layer'])
    x_line = np.linspace(df['lf_score'].min(), df['lf_score'].max(), 100)
    ax2.plot(x_line, intercept + slope * x_line, 'k--', linewidth=2,
             label=f'Regression (r={r:.2f}, p={p:.3g})')
    ax2.set_xlabel('Lexical Familiarity Score', fontsize=12)
    ax2.set_ylabel('Peak Divergence Layer', fontsize=12)
    ax2.set_title('Peak Layer vs. Lexical Familiarity (Qwen3.5-9B)', fontsize=14)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    sns.boxplot(data=df_ordered, x='category', y='fragmentation',
                order=category_order, palette=colors, ax=ax3)
    ax3.set_xlabel('Category', fontsize=12)
    ax3.set_ylabel('Subword Fragmentation Index', fontsize=12)
    ax3.set_title('Tokenizer Fragmentation by Category', fontsize=14)
    ax3.set_xticklabels(['Idiom', 'Constr.†', 'Metaphor', 'Sem. Shift', 'Neologism'], rotation=15)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='1:1 token ratio')

    ax4 = axes[1, 1]
    sns.boxplot(data=df_ordered, x='category', y='frequency_score',
                order=category_order, palette=colors, ax=ax4)
    ax4.set_xlabel('Category', fontsize=12)
    ax4.set_ylabel('Token Frequency Percentile', fontsize=12)
    ax4.set_title('Token Frequency by Category', fontsize=14)
    ax4.set_xticklabels(['Idiom', 'Constr.†', 'Metaphor', 'Sem. Shift', 'Neologism'], rotation=15)

    plt.tight_layout()
    plt.savefig(output_dir / 'lf_analysis_plots_qwen.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'lf_analysis_plots_qwen.png'}")

    fig, ax = plt.subplots(figsize=(8, 6))
    cat_means = df.groupby('category').agg({
        'lf_score': ['mean', 'std', 'count'],
        'peak_layer': 'first',
    }).reset_index()
    cat_means.columns = ['category', 'lf_mean', 'lf_std', 'n', 'peak_layer']
    cat_means['lf_se'] = cat_means['lf_std'] / np.sqrt(cat_means['n'])

    for _, row in cat_means.iterrows():
        cat = row['category']
        ax.errorbar(row['lf_mean'], row['peak_layer'],
                    xerr=1.96 * row['lf_se'], fmt='o', markersize=12, capsize=5,
                    color=colors[cat], label=cat.replace('_', ' ').title())

    slope, intercept, r, p, se = stats.linregress(cat_means['lf_mean'], cat_means['peak_layer'])
    x_line = np.linspace(cat_means['lf_mean'].min() - 0.5, cat_means['lf_mean'].max() + 0.5, 100)
    ax.plot(x_line, intercept + slope * x_line, 'k--', linewidth=2, alpha=0.7)

    ax.set_xlabel('Mean Lexical Familiarity Score (± 95% CI)', fontsize=12)
    ax.set_ylabel('Peak Divergence Layer', fontsize=12)
    ax.set_title(f'Lexical Familiarity Predicts Processing Depth (Qwen3.5-9B)\n(r = {r:.2f})',
                 fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    for _, row in cat_means.iterrows():
        ax.annotate(f"L{int(row['peak_layer'])}",
                    xy=(row['lf_mean'], row['peak_layer']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / 'lf_peak_layer_figure_qwen.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'lf_peak_layer_figure_qwen.png'}")


def generate_latex_table(df: pd.DataFrame, results: Dict, output_dir: Path):
    summary = df.groupby('category').agg({
        'fragmentation': 'mean',
        'frequency_score': 'mean',
        'lf_score': 'mean',
        'peak_layer': 'first',
    }).reset_index()
    summary = summary.sort_values('peak_layer')

    latex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"Category & Fragmentation & Frequency & LF Score & Peak \\", r"\midrule",
    ]
    for _, row in summary.iterrows():
        name = row['category'].replace('_', ' ').title()
        if row['category'] == 'construction':
            name += r'$^\dagger$'
        latex.append(f"{name} & {row['fragmentation']:.2f} & {row['frequency_score']:.1f} & "
                     f"{row['lf_score']:+.2f} & L{int(row['peak_layer'])} \\\\")
    latex.extend([
        r"\midrule",
        f"\\multicolumn{{5}}{{l}}{{Regression: peak layer $\\sim$ LF score, "
        f"$r^2 = {results['model2_r2']:.3f}$, $p = {results['lf_pvalue']:.3g}$}} \\\\",
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Lexical familiarity metrics by category (Qwen3.5-9B). "
        r"Fragmentation = subword tokens per word; Frequency = token frequency percentile; "
        r"LF Score = combined z-score (higher = more familiar).}",
        r"\label{tab:lf-scores-qwen}", r"\end{table}",
    ])
    out = output_dir / 'lf_table_qwen.tex'
    with open(out, 'w') as f:
        f.write('\n'.join(latex))
    print(f"Saved: {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description='LF analysis for Qwen3.5-9B')
    parser.add_argument('--data-dir', type=Path, default=Path('/home/tyleryeh47/sae_slang/data'))
    parser.add_argument('--results-dir', type=Path, default=Path('/home/tyleryeh47/sae_slang/results/qwen'),
                        help='Directory with run_qwen.py outputs (used to derive peak_map).')
    parser.add_argument('--output-dir', type=Path, default=Path('/home/tyleryeh47/sae_slang/results/qwen/lexical_familiarity'))
    parser.add_argument('--model', '-m', default='Qwen/Qwen3.5-9B')
    parser.add_argument('--peak-metric', default='cosine_dist',
                        choices=['cosine_dist', 'l2_dist', 'l1_dist'],
                        help='Metric used to identify the peak layer per category.')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Lexical Familiarity Analysis (Qwen3.5-9B)")
    print("=" * 60)

    tokenizer = load_tokenizer(args.model)

    print("\nLoading and processing categories...")
    df = load_all_categories(args.data_dir, tokenizer)

    print(f"\nDeriving peak_map from {args.results_dir} (metric={args.peak_metric})...")
    peak_map = derive_peak_map_from_results(args.results_dir, metric=args.peak_metric)
    if not peak_map:
        raise RuntimeError(f"No peaks derived; run run_qwen.py first into {args.results_dir}")
    df = assign_peak_layers(df, peak_map)
    df = df.dropna(subset=['peak_layer'])

    df.to_csv(args.output_dir / 'lf_scores_all_items_qwen.csv', index=False)
    print(f"\nSaved: {args.output_dir / 'lf_scores_all_items_qwen.csv'}")

    print("\n" + "=" * 60)
    print("CATEGORY SUMMARY")
    print("=" * 60)
    summary = df.groupby('category').agg({
        'fragmentation': ['mean', 'std'],
        'frequency_score': ['mean', 'std'],
        'lf_score': ['mean', 'std'],
        'peak_layer': 'first',
        'item_idx': 'count',
    }).round(3)
    print(summary)

    results = run_regression_analysis(df)

    print("\nGenerating visualizations...")
    plot_lf_analysis(df, args.output_dir)

    generate_latex_table(df, results, args.output_dir)


if __name__ == '__main__':
    main()
