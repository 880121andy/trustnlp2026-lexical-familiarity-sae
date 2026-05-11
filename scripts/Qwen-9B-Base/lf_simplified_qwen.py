#!/usr/bin/env python3
"""
Lexical Familiarity Analysis - Simplified (Qwen3.5-9B).

Uses the Qwen tokenizer for subword
fragmentation / token frequency stats, and auto-derives peak layers from
run_qwen.py outputs in --results-dir.
"""

import argparse
import ast
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')


CATEGORY_CONFIGS = {
    'idiom':          ('idiom_baseline.csv',       'idiomatic',    'idiom_positions'),
    'metaphor':       ('metaphor_baseline.csv',    'metaphorical', 'metaphor_positions'),
    'semantic_shift': ('genz_dataset_tagged.csv',  'gen_z',        'slang_positions'),
    'neologism':      ('neologism_tagged.csv',     'gen_z',        'slang_positions'),
    'construction':   ('constructions_tagged.csv', 'gen_z',        'slang_positions'),
}


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
    return len(tokens) / n_whitespace_words


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
    percentiles = [100 * (1 - tid / vocab_size) for tid in token_ids]
    return float(np.mean(percentiles))


def load_category(filepath: Path, category: str, text_col: str, pos_col: str,
                  tokenizer) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    records = []
    for idx, row in df.iterrows():
        try:
            positions = ast.literal_eval(row[pos_col])
            text = str(row[text_col])
            frag = compute_subword_fragmentation(text, positions, tokenizer)
            freq = compute_token_frequency_score(text, positions, tokenizer)
            records.append({
                'category': category, 'item_idx': idx,
                'fragmentation': frag, 'frequency_score': freq,
            })
        except Exception:
            continue
    return pd.DataFrame(records)


def derive_peak_map(results_dir: Path, metric: str = 'cosine_dist') -> Dict[str, int]:
    peak_map: Dict[str, int] = {}
    for cat in CATEGORY_CONFIGS.keys():
        files = sorted(results_dir.glob(f"{cat}_qwen_*.csv"))
        if not files:
            print(f"Warning: no run_qwen CSV for '{cat}' in {results_dir}")
            continue
        latest = files[-1]
        df = pd.read_csv(latest)
        df_metric = df[df['metric'] == metric]
        if df_metric.empty:
            print(f"Warning: metric '{metric}' missing in {latest.name}")
            continue
        peak_map[cat] = int(df_metric.loc[df_metric['mean'].idxmax(), 'layer'])
        print(f"  {cat}: peak layer = L{peak_map[cat]} (from {latest.name})")
    return peak_map


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('/home/tyleryeh47/sae_slang/data'))
    parser.add_argument('--results-dir', type=Path, default=Path('/home/tyleryeh47/sae_slang/results/qwen'),
                        help='Directory with run_qwen.py outputs.')
    parser.add_argument('--output-dir', type=Path, default=Path('/home/tyleryeh47/sae_slang/results/qwen/lexical_familiarity'))
    parser.add_argument('--model', '-m', default='Qwen/Qwen3.5-9B')
    parser.add_argument('--peak-metric', default='cosine_dist',
                        choices=['cosine_dist', 'l2_dist', 'l1_dist'])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Lexical Familiarity Analysis - Simplified (Qwen3.5-9B)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    all_data = []
    for cat, (fname, text_col, pos_col) in CATEGORY_CONFIGS.items():
        fpath = args.data_dir / fname
        if not fpath.exists():
            print(f"[skip] {cat}: {fpath} not found")
            continue
        print(f"Loading {cat}...")
        cat_df = load_category(fpath, cat, text_col, pos_col, tokenizer)
        all_data.append(cat_df)
        print(f"  {len(cat_df)} items")

    df = pd.concat(all_data, ignore_index=True)

    frag_mean, frag_std = df['fragmentation'].mean(), df['fragmentation'].std()
    freq_mean, freq_std = df['frequency_score'].mean(), df['frequency_score'].std()
    df['z_frag'] = (df['fragmentation'] - frag_mean) / frag_std
    df['z_freq'] = (df['frequency_score'] - freq_mean) / freq_std
    df['lf_score'] = -df['z_frag'] + df['z_freq']

    print(f"\nDeriving peak_map from {args.results_dir} (metric={args.peak_metric})...")
    peak_map = derive_peak_map(args.results_dir, metric=args.peak_metric)
    if not peak_map:
        raise RuntimeError(f"No peaks derived; run run_qwen.py first into {args.results_dir}")
    df['peak_layer'] = df['category'].map(peak_map)
    df = df.dropna(subset=['peak_layer'])

    print("\n" + "=" * 60)
    print("CATEGORY SUMMARY")
    print("=" * 60)
    summary = df.groupby('category').agg({
        'fragmentation': 'mean',
        'frequency_score': 'mean',
        'lf_score': 'mean',
        'peak_layer': 'first',
        'item_idx': 'count',
    }).rename(columns={'item_idx': 'n'})
    summary = summary.sort_values('peak_layer')
    print(summary.round(3))

    print("\n" + "=" * 60)
    print("REGRESSION: peak_layer ~ lf_score")
    print("=" * 60)
    model = smf.ols('peak_layer ~ lf_score', data=df).fit()
    print(f"Coefficient: {model.params['lf_score']:.3f}")
    print(f"p-value:     {model.pvalues['lf_score']:.3g}")
    print(f"R²:          {model.rsquared:.4f}")

    cat_stats = df.groupby('category').agg({
        'lf_score': ['mean', 'std', 'count'],
        'peak_layer': 'first',
    }).reset_index()
    cat_stats.columns = ['category', 'lf_mean', 'lf_std', 'n', 'peak_layer']
    cat_stats['lf_se'] = cat_stats['lf_std'] / np.sqrt(cat_stats['n'])

    r, p = stats.pearsonr(cat_stats['lf_mean'], cat_stats['peak_layer'])
    print(f"\nCategory-level correlation: r = {r:.3f}, p = {p:.4f}")

    colors = {
        'idiom': '#1f77b4', 'construction': '#ff7f0e', 'metaphor': '#2ca02c',
        'semantic_shift': '#d62728', 'neologism': '#9467bd',
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in cat_stats.iterrows():
        cat = row['category']
        label = cat.replace('_', ' ').title()
        if cat == 'construction':
            label += '†'
        ax.errorbar(row['lf_mean'], row['peak_layer'],
                    xerr=1.96 * row['lf_se'], fmt='o', markersize=14,
                    capsize=6, capthick=2, color=colors[cat], label=label,
                    markeredgecolor='white', markeredgewidth=1.5)

    x_range = np.linspace(cat_stats['lf_mean'].min() - 0.3, cat_stats['lf_mean'].max() + 0.3, 100)
    slope, intercept = np.polyfit(cat_stats['lf_mean'], cat_stats['peak_layer'], 1)
    ax.plot(x_range, intercept + slope * x_range, 'k--', linewidth=2, alpha=0.6)
    ax.set_xlabel('Lexical Familiarity Score (± 95% CI)', fontsize=12)
    ax.set_ylabel('Peak Divergence Layer', fontsize=12)
    ax.set_title(f'Lexical Familiarity Predicts Processing Depth (Qwen3.5-9B)\n'
                 f'(r = {r:.2f}, p = {p:.3g})', fontsize=14)
    ax.legend(loc='center right', fontsize=10)
    ax.grid(True, alpha=0.3)
    for _, row in cat_stats.iterrows():
        ax.annotate(f"L{int(row['peak_layer'])}",
                    xy=(row['lf_mean'], row['peak_layer']),
                    xytext=(8, 0), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.output_dir / 'lf_regression_plot_qwen.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {args.output_dir / 'lf_regression_plot_qwen.png'}")

    df.to_csv(args.output_dir / 'lf_scores_simplified_qwen.csv', index=False)
    print(f"Saved: {args.output_dir / 'lf_scores_simplified_qwen.csv'}")


if __name__ == '__main__':
    main()
