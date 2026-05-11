"""
Normal Approximation Confidence Intervals

Simple script to compute 95% CIs from aggregated results (mean, std, n).
No bootstrap required — valid for large sample sizes (n > 30).

Formula: CI = mean ± 1.96 × (std / sqrt(n))
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_ci(mean: float, std: float, n: int, ci_level: float = 0.95) -> tuple:
    """
    Compute CI using normal approximation.

    Returns: (lower, upper, standard_error)
    """
    z = 1.96 if ci_level == 0.95 else 2.576  # 95% or 99%
    se = std / np.sqrt(n)
    lower = mean - z * se
    upper = mean + z * se
    return lower, upper, se


def add_cis_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add CI columns to dataframe with mean, std, n columns."""
    df = df.copy()
    results = df.apply(lambda row: compute_ci(
        row['mean'], row['std'], row['n']), axis=1)
    df['ci_lower'] = [r[0] for r in results]
    df['ci_upper'] = [r[1] for r in results]
    df['se'] = [r[2] for r in results]
    return df


def main():
    data_dir = Path('/mnt/project')
    output_dir = Path('/mnt/user-data/outputs')

    # File mapping: name -> (filename, n_samples)
    files = {
        'neologism': 'neo.csv',
        'semantic_shift': 'semantic_20260129.csv',
        'construction': 'const.csv',
        'metaphor': 'metaphor.csv',
        'idiom': 'idiom_20260226.csv',
        'literal_paraphrase': 'literalparaphrase_20260227.csv',
    }

    print("=" * 60)
    print("Normal Approximation CIs (95%)")
    print("=" * 60)

    all_results = []

    for name, filename in files.items():
        filepath = data_dir / filename
        if not filepath.exists():
            continue

        df = pd.read_csv(filepath)

        # Filter to cosine_dist, layers 0-47
        df_cos = df[(df['metric'] == 'cosine_dist')
                    & (df['layer'] <= 47)].copy()
        df_cos = df_cos[['layer', 'mean', 'std', 'n']].sort_values('layer')

        # Add CIs
        df_cos = add_cis_to_df(df_cos)
        df_cos['category'] = name

        # Find peak
        peak_idx = df_cos['mean'].idxmax()
        peak = df_cos.loc[peak_idx]

        print(f"\n{name}:")
        print(f"  n = {int(peak['n'])}")
        print(f"  Peak: L{int(peak['layer'])} = {peak['mean']:.3f} [{
              peak['ci_lower']:.3f}, {peak['ci_upper']:.3f}]")

        all_results.append(df_cos)

    # Combine all
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output_dir / 'all_categories_with_ci.csv', index=False)
    print(f"\nSaved: {output_dir / 'all_categories_with_ci.csv'}")

    # Generate LaTeX table for peaks
    print("\n" + "=" * 60)
    print("LaTeX Table (Peak Values with CIs)")
    print("=" * 60)

    latex_lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Category & Peak & Cosine Distance & 95\% CI \\",
        r"\midrule",
    ]

    order = ['idiom', 'literal_paraphrase', 'construction',
             'metaphor', 'semantic_shift', 'neologism']

    for cat in order:
        cat_df = combined[combined['category'] == cat]
        if cat_df.empty:
            continue
        peak_idx = cat_df['mean'].idxmax()
        peak = cat_df.loc[peak_idx]

        name = cat.replace('_', ' ').title()
        if cat == 'construction':
            name += r'$^\dagger$'
        if cat == 'literal_paraphrase':
            name = 'Lit. Paraphrase'

        latex_lines.append(
            f"{name} & L{int(peak['layer'])} & {peak['mean']:.3f} & "
            f"[{peak['ci_lower']:.3f}, {peak['ci_upper']:.3f}] \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}"
    ])

    print('\n'.join(latex_lines))

    with open(output_dir / 'peak_ci_table.tex', 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"\nSaved: {output_dir / 'peak_ci_table.tex'}")


if __name__ == '__main__':
    main()
