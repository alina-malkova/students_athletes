"""
Forest plot of §3.8 retention coefficients by sport.

Visualizes the conflict and polstab_drop coefficients on intl-athlete
retention separately for each sport, with 95% CIs (country-clustered
SEs). Reads output/retention_own_shock_by_sport.csv.
"""
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')


def main():
    df = pd.read_csv(os.path.join(OUTPUT, 'retention_own_shock_by_sport.csv'))
    df = df[df['shock'].isin(['any_conflict', 'polstab_drop'])].copy()
    df['ci_lo'] = df['beta'] - 1.96 * df['se']
    df['ci_hi'] = df['beta'] + 1.96 * df['se']
    sport_order = (df[df['shock'] == 'any_conflict']
                     .sort_values('beta')['sport'].tolist())

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    y = list(range(len(sport_order)))
    colors = {'any_conflict': '#c0392b', 'polstab_drop': '#2c3e50'}
    labels = {'any_conflict': 'Armed conflict (UCDP)',
              'polstab_drop': 'Political-stability drop'}
    offset = {'any_conflict': -0.18, 'polstab_drop': +0.18}

    for sv in ['any_conflict', 'polstab_drop']:
        sub = df[df['shock'] == sv].set_index('sport').reindex(sport_order)
        ys = [yi + offset[sv] for yi in y]
        ax.errorbar(sub['beta'], ys,
                    xerr=[sub['beta'] - sub['ci_lo'],
                          sub['ci_hi'] - sub['beta']],
                    fmt='o', color=colors[sv], capsize=3, markersize=7,
                    label=labels[sv], lw=1.4)

    ax.axvline(0, color='black', linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(sport_order)
    ax.set_xlabel(r'$\beta$ on Pr(return to same school)', fontsize=11)
    ax.set_title('Home-country shock effects on intl athlete retention\n'
                  'by sport (year FE, country-clustered SE)', fontsize=11)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(-0.18, 0.30)
    fig.tight_layout()
    out = os.path.join(OUTPUT, 'fig_retention_by_sport.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
