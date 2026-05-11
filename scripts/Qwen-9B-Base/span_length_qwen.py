#!/usr/bin/env python3
"""
Span Length Confound Analysis (Qwen3.5-9B).

peak_map and cosine_map are auto-derived
from run_qwen.py outputs; the rest of the analysis (span length stats,
correlations, optional LF covariate regression) is unchanged.
"""

import argparse
import ast
import warnings
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings('ignore')


CATEGORY_CONFIGS = {
    'idiom':          ('idiom_baseline.csv',       'idiom_positions'),
    'metaphor':       ('metaphor_baseline.csv',    'metaphor_positions'),
    'semantic_shift': ('genz_dataset_tagged.csv',  'slang_positions'),
    'neologism':      ('neologism_tagged.csv',     'slang_positions'),
    'construction':   ('constructions_tagged.csv', 'slang_positions'),
}


def compute_span_length_words(positions: list) -> int:
    return sum(end - start for start, end in positions)


def load_category_spans(filepath: Path, pos_col: str, category: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    records = []
    for idx, row in df.iterrows():
        try:
            positions = ast.literal_eval(row[pos_col])
            records.append({
                'category': category, 'item_idx': idx,
                'span_length_words': compute_span_length_words(positions),
                'n_spans': len(positions),
            })
        except Exception:
            continue
    return pd.DataFrame(records)


def derive_peak_and_cosine_maps(results_dir: Path) -> (Dict[str, int], Dict[str, float]):
    peak_map: Dict[str, int] = {}
    cosine_map: Dict[str, float] = {}
    for cat in CATEGORY_CONFIGS.keys():
        files = sorted(results_dir.glob(f"{cat}_qwen_*.csv"))
        if not files:
            print(f"Warning: no run_qwen CSV for '{cat}' in {results_dir}")
            continue
        latest = files[-1]
        df = pd.read_csv(latest)
        cos_df = df[df['metric'] == 'cosine_dist']
        if cos_df.empty:
            print(f"Warning: cosine_dist missing in {latest.name}")
            continue
        peak_idx = cos_df['mean'].idxmax()
        peak_map[cat] = int(cos_df.loc[peak_idx, 'layer'])
        cosine_map[cat] = float(cos_df.loc[peak_idx, 'mean'])
        print(f"  {cat}: peak L{peak_map[cat]}, cosine={cosine_map[cat]:.3f} (from {latest.name})")
    return peak_map, cosine_map


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('/home/tyleryeh47/sae_slang/data'))
    parser.add_argument('--results-dir', type=Path, default=Path('/home/tyleryeh47/sae_slang/results/qwen'),
                        help='Directory with run_qwen.py outputs.')
    parser.add_argument('--output-dir', type=Path, default=Path('/home/tyleryeh47/sae_slang/results/qwen/span_length'))
    parser.add_argument('--lf-csv', type=Path, default=None,
                        help='Optional path to lf_scores_simplified_qwen.csv for covariate regression.')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Span Length Confound Analysis (Qwen3.5-9B)")
    print("=" * 60)

    all_data = []
    for cat, (fname, pos_col) in CATEGORY_CONFIGS.items():
        fpath = args.data_dir / fname
        if not fpath.exists():
            print(f"[skip] {cat}: {fpath} not found")
            continue
        cat_df = load_category_spans(fpath, pos_col, cat)
        all_data.append(cat_df)
        print(f"Loaded {cat}: {len(cat_df)} items")

    df = pd.concat(all_data, ignore_index=True)

    print(f"\nDeriving peak/cosine maps from {args.results_dir}...")
    peak_map, cosine_map = derive_peak_and_cosine_maps(args.results_dir)
    if not peak_map:
        raise RuntimeError(f"No peaks derived; run run_qwen.py first into {args.results_dir}")

    df['peak_layer'] = df['category'].map(peak_map)
    df['peak_cosine'] = df['category'].map(cosine_map)
    df = df.dropna(subset=['peak_layer', 'peak_cosine'])

    print("\n" + "=" * 60)
    print("SPAN LENGTH BY CATEGORY")
    print("=" * 60)
    summary = df.groupby('category').agg({
        'span_length_words': ['mean', 'std', 'min', 'max'],
        'n_spans': 'mean',
        'peak_layer': 'first',
        'peak_cosine': 'first',
        'item_idx': 'count',
    }).round(2)
    summary.columns = ['mean_len', 'std_len', 'min_len', 'max_len',
                       'mean_n_spans', 'peak_layer', 'peak_cosine', 'n']
    summary = summary.sort_values('peak_layer')
    print(summary)

    cat_means = df.groupby('category').agg({
        'span_length_words': 'mean',
        'peak_layer': 'first',
        'peak_cosine': 'first',
    }).reset_index()

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    r1, p1 = stats.pearsonr(cat_means['span_length_words'], cat_means['peak_layer'])
    print(f"Span length vs Peak layer:  r = {r1:.3f}, p = {p1:.3f}")
    r2, p2 = stats.pearsonr(cat_means['span_length_words'], cat_means['peak_cosine'])
    print(f"Span length vs Peak cosine: r = {r2:.3f}, p = {p2:.3f}")
    if p2 < 0.05:
        print("\n[!] CONFOUND DETECTED: Span length correlates with peak cosine distance")
    else:
        print("\n[ok] No significant confound: Span length does not correlate with peak cosine")

    if args.lf_csv is not None and args.lf_csv.exists():
        print("\n" + "=" * 60)
        print("REGRESSION WITH SPAN LENGTH COVARIATE")
        print("=" * 60)
        lf_df = pd.read_csv(args.lf_csv)
        merged = lf_df.merge(df[['category', 'item_idx', 'span_length_words']],
                             on=['category', 'item_idx'], how='left')
        merged = merged.dropna(subset=['span_length_words'])

        m1 = smf.ols('peak_layer ~ lf_score', data=merged).fit()
        m2 = smf.ols('peak_layer ~ lf_score + span_length_words', data=merged).fit()
        m3 = smf.ols('peak_layer ~ span_length_words', data=merged).fit()

        print(f"\n{'Model':<35} {'R²':<10} {'AIC':<12}")
        print("-" * 57)
        print(f"{'LF score only':<35} {m1.rsquared:.4f}    {m1.aic:.1f}")
        print(f"{'Span length only':<35} {m3.rsquared:.4f}    {m3.aic:.1f}")
        print(f"{'LF score + Span length':<35} {m2.rsquared:.4f}    {m2.aic:.1f}")
        print(f"\nLF + Span coefficients:")
        print(f"  lf_score:          beta = {m2.params['lf_score']:.3f}, "
              f"p = {m2.pvalues['lf_score']:.3g}")
        print(f"  span_length_words: beta = {m2.params['span_length_words']:.3f}, "
              f"p = {m2.pvalues['span_length_words']:.3g}")

        merged.to_csv(args.output_dir / 'lf_scores_with_span_qwen.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cat_order = [c for c in ['idiom', 'construction', 'metaphor', 'semantic_shift', 'neologism']
                 if c in cat_means['category'].values]
    colors = {
        'idiom': '#1f77b4', 'construction': '#ff7f0e', 'metaphor': '#2ca02c',
        'semantic_shift': '#d62728', 'neologism': '#9467bd',
    }
    means = [cat_means[cat_means['category'] == c]['span_length_words'].values[0] for c in cat_order]
    peaks = [int(peak_map[c]) for c in cat_order]

    ax1 = axes[0]
    ax1.bar(range(len(cat_order)), means, color=[colors[c] for c in cat_order])
    ax1.set_xticks(range(len(cat_order)))
    ax1.set_xticklabels([f"{c.replace('_', ' ').title()}\n(L{p})" for c, p in zip(cat_order, peaks)])
    ax1.set_ylabel('Mean Span Length (words)')
    ax1.set_title('Divergence Span Length by Category (Qwen3.5-9B)')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = axes[1]
    for _, row in cat_means.iterrows():
        cat = row['category']
        ax2.scatter(row['span_length_words'], row['peak_cosine'],
                    s=150, c=colors[cat], label=cat.replace('_', ' ').title(),
                    edgecolors='white', linewidth=1.5)
    ax2.set_xlabel('Mean Span Length (words)')
    ax2.set_ylabel('Peak Cosine Distance')
    ax2.set_title(f'Span Length vs Peak Cosine (r = {r2:.2f}, p = {p2:.2f})')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output_dir / 'span_length_analysis_qwen.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {args.output_dir / 'span_length_analysis_qwen.png'}")

    summary.to_csv(args.output_dir / 'span_length_summary_qwen.csv')
    print(f"Saved: {args.output_dir / 'span_length_summary_qwen.csv'}")


if __name__ == '__main__':
    main()
