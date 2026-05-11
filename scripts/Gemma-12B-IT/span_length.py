#!/usr/bin/env python3
"""
Span Length Confound Analysis

1. Mean divergence-span length per category (in words and tokens)
2. Correlation between span length and peak cosine distance
3. Regression with span length as covariate
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import ast
import warnings
warnings.filterwarnings('ignore')


def compute_span_length_words(positions: list) -> int:
    """Compute total span length in words from position tuples."""
    total = 0
    for start, end in positions:
        total += (end - start)
    return total


def load_category_spans(filepath: Path, pos_col: str, category: str) -> pd.DataFrame:
    """Load span lengths for a category."""
    df = pd.read_csv(filepath)

    records = []
    for idx, row in df.iterrows():
        try:
            positions = ast.literal_eval(row[pos_col])
            span_len = compute_span_length_words(positions)
            n_spans = len(positions)  # Number of discontinuous spans

            records.append({
                'category': category,
                'item_idx': idx,
                'span_length_words': span_len,
                'n_spans': n_spans
            })
        except:
            continue

    return pd.DataFrame(records)


def main():
    data_dir = Path('/mnt/project')
    output_dir = Path('/mnt/user-data/outputs')

    print("=" * 60)
    print("Span Length Confound Analysis")
    print("=" * 60)

    # Load all categories
    configs = {
        'idiom': ('idiom_baseline.csv', 'idiom_positions'),
        'metaphor': ('metaphor_baseline.csv', 'metaphor_positions'),
        'semantic_shift': ('genz_dataset_tagged.csv', 'slang_positions'),
        'neologism': ('neologism_tagged.csv', 'slang_positions'),
        'construction': ('constructions_tagged.csv', 'slang_positions'),
    }

    all_data = []
    for cat, (fname, pos_col) in configs.items():
        fpath = data_dir / fname
        if fpath.exists():
            cat_df = load_category_spans(fpath, pos_col, cat)
            all_data.append(cat_df)
            print(f"Loaded {cat}: {len(cat_df)} items")

    df = pd.concat(all_data, ignore_index=True)

    # Add peak layers
    peak_map = {'idiom': 1, 'construction': 7,
                'metaphor': 8, 'semantic_shift': 9, 'neologism': 41}
    df['peak_layer'] = df['category'].map(peak_map)

    # Add peak cosine values from paper (Table 2)
    cosine_map = {'idiom': 0.830, 'construction': 0.659, 'metaphor': 0.474,
                  'semantic_shift': 0.478, 'neologism': 0.857}
    df['peak_cosine'] = df['category'].map(cosine_map)

    # Summary by category
    print("\n" + "=" * 60)
    print("SPAN LENGTH BY CATEGORY")
    print("=" * 60)

    summary = df.groupby('category').agg({
        'span_length_words': ['mean', 'std', 'min', 'max'],
        'n_spans': 'mean',
        'peak_layer': 'first',
        'peak_cosine': 'first',
        'item_idx': 'count'
    }).round(2)
    summary.columns = ['mean_len', 'std_len', 'min_len', 'max_len',
                       'mean_n_spans', 'peak_layer', 'peak_cosine', 'n']
    summary = summary.sort_values('peak_layer')
    print(summary)

    # Category-level correlations
    cat_means = df.groupby('category').agg({
        'span_length_words': 'mean',
        'peak_layer': 'first',
        'peak_cosine': 'first'
    }).reset_index()

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    # Span length vs peak layer
    r1, p1 = stats.pearsonr(
        cat_means['span_length_words'], cat_means['peak_layer'])
    print(f"\nSpan length vs Peak layer:  r = {r1:.3f}, p = {p1:.3f}")

    # Span length vs peak cosine
    r2, p2 = stats.pearsonr(
        cat_means['span_length_words'], cat_means['peak_cosine'])
    print(f"Span length vs Peak cosine: r = {r2:.3f}, p = {p2:.3f}")

    if p2 < 0.05:
        print("\n⚠️  CONFOUND DETECTED: Span length correlates with peak cosine distance")
    else:
        print(
            "\n✓ No significant confound: Span length does not correlate with peak cosine")

    # Load LF scores if available and add span length as covariate
    lf_path = output_dir / 'lf_scores_simplified.csv'
    if lf_path.exists():
        print("\n" + "=" * 60)
        print("REGRESSION WITH SPAN LENGTH COVARIATE")
        print("=" * 60)

        lf_df = pd.read_csv(lf_path)

        # Merge span lengths
        merged = lf_df.merge(df[['category', 'item_idx', 'span_length_words']],
                             on=['category', 'item_idx'], how='left')
        merged = merged.dropna(subset=['span_length_words'])

        # Model 1: LF only (baseline)
        m1 = smf.ols('peak_layer ~ lf_score', data=merged).fit()

        # Model 2: LF + span length
        m2 = smf.ols('peak_layer ~ lf_score + span_length_words',
                     data=merged).fit()

        # Model 3: Span length only
        m3 = smf.ols('peak_layer ~ span_length_words', data=merged).fit()

        print(f"\n{'Model':<35} {'R²':<10} {'AIC':<12}")
        print("-" * 57)
        print(f"{'LF score only':<35} {m1.rsquared:.4f}    {m1.aic:.1f}")
        print(f"{'Span length only':<35} {m3.rsquared:.4f}    {m3.aic:.1f}")
        print(f"{'LF score + Span length':<35} {m2.rsquared:.4f}    {m2.aic:.1f}")

        print(f"\nLF score + Span length model coefficients:")
        print(f"  lf_score:          β = {m2.params['lf_score']:.3f}, p = {
              m2.pvalues['lf_score']:.2e}")
        print(f"  span_length_words: β = {m2.params['span_length_words']:.3f}, p = {
              m2.pvalues['span_length_words']:.2e}")

        if m2.pvalues['lf_score'] < 0.05 and m2.pvalues['span_length_words'] >= 0.05:
            print("\n✓ LF score remains significant; span length is NOT a confound")
        elif m2.pvalues['lf_score'] < 0.05 and m2.pvalues['span_length_words'] < 0.05:
            print("\n⚠️  Both significant: span length explains additional variance")
            print("   Consider reporting both in the paper")
        else:
            print("\n⚠️  LF score weakened by span length control")

        # Save merged data
        merged.to_csv(output_dir / 'lf_scores_with_span.csv', index=False)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Span length by category
    ax1 = axes[0]
    cat_order = ['idiom', 'construction',
                 'metaphor', 'semantic_shift', 'neologism']
    colors = {'idiom': '#1f77b4', 'construction': '#ff7f0e', 'metaphor': '#2ca02c',
              'semantic_shift': '#d62728', 'neologism': '#9467bd'}

    means = [cat_means[cat_means['category'] == c]
             ['span_length_words'].values[0] for c in cat_order]
    bars = ax1.bar(range(len(cat_order)), means, color=[
                   colors[c] for c in cat_order])
    ax1.set_xticks(range(len(cat_order)))
    ax1.set_xticklabels(['Idiom\n(L1)', 'Constr.†\n(L7)', 'Metaphor\n(L8)',
                         'Sem.Shift\n(L9)', 'Neologism\n(L41)'])
    ax1.set_ylabel('Mean Span Length (words)')
    ax1.set_title('Divergence Span Length by Category')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Span length vs peak cosine
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
    plt.savefig(output_dir / 'span_length_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir / 'span_length_analysis.png'}")

    # LaTeX table
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Category & Mean & SD & Range & Peak & N \\",
        r"\midrule"
    ]

    for _, row in summary.iterrows():
        name = row.name.replace('_', ' ').title()
        if row.name == 'construction':
            name += r'$^\dagger$'
        latex.append(f"{name} & {row['mean_len']:.1f} & {row['std_len']:.1f} & "
                     f"{int(row['min_len'])}--{int(row['max_len'])} & "
                     f"L{int(row['peak_layer'])} & {int(row['n'])} \\\\")

    latex.extend([
        r"\midrule",
        f"\\multicolumn{{6}}{{l}}{{Span length vs. peak cosine: $r = {
            r2:.2f}$, $p = {p2:.2f}$}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Mean divergence span length (in words) by category.}",
        r"\label{tab:span-length}",
        r"\end{table}"
    ])

    with open(output_dir / 'span_length_table.tex', 'w') as f:
        f.write('\n'.join(latex))
    print(f"Saved: {output_dir / 'span_length_table.tex'}")

    # Save summary
    summary.to_csv(output_dir / 'span_length_summary.csv')
    print(f"Saved: {output_dir / 'span_length_summary.csv'}")

    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"""
Mean span lengths: Idiom ({summary.loc['idiom', 'mean_len']:.1f} words) >
Metaphor ({summary.loc['metaphor', 'mean_len']:.1f}) >
Neologism ({summary.loc['neologism', 'mean_len']:.1f}) >
Semantic shift ({summary.loc['semantic_shift', 'mean_len']:.1f})

Correlation with peak cosine: r = {r2:.2f}, p = {p2:.2f}
{"→ NOT a significant confound" if p2 >=
        0.05 else "→ Potential confound, add as covariate"}
""")


if __name__ == '__main__':
    main()
