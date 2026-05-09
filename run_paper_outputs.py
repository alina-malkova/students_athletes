"""
Generate Table 1 (descriptive stats) and paper-ready figures.

- output/table1_descriptives.csv  -- by income tier
- output/table1_descriptives.tex  -- LaTeX, optional
- output/fig_coef_forest.png      -- coefficient forest plot across the 5 main specs
- output/fig_log_athletes_per_cohort.png  -- final scatter, annotated
- output/fig_top20_countries_panel.png    -- side-by-side: count vs cohort-rate
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')
os.makedirs(OUTPUT, exist_ok=True)


def main():
    warnings.filterwarnings('ignore')
    df = pd.read_csv(os.path.join(RAW_DATA, 'analysis_dataset.csv'),
                     low_memory=False)

    intl = df[(df['is_international'] == 1) & df['country_code'].notna()]
    panel = (intl.groupby('country_code')
                  .agg(athletes=('school_name', 'count'),
                       gdp_per_capita_ppp=('gdp_per_capita_ppp', 'first'),
                       population=('population', 'first'),
                       pop_15_24=('pop_15_24_avg_2018_22', 'first'),
                       political_stability=('political_stability', 'first'),
                       econ_shocks_2010_23=('econ_shocks_2010_23', 'first'),
                       disaster_deaths_2010_23=('disaster_deaths_2010_23', 'first'),
                       years_with_conflict_2010_23=('years_with_conflict_2010_23', 'first'))
                  .reset_index())
    panel = panel.dropna(subset=['gdp_per_capita_ppp', 'population']).copy()
    panel['athletes_per_million']     = 1e6 * panel['athletes'] / panel['population']
    panel['athletes_per_million_15_24'] = 1e6 * panel['athletes'] / panel['pop_15_24']

    # ---------- Table 1: descriptive stats by income tier ----------
    # Quartile cutoffs of GDP per capita PPP
    qs = panel['gdp_per_capita_ppp'].quantile([0.25, 0.5, 0.75]).tolist()
    def tier(g):
        if g < qs[0]: return '1. Low (Q1)'
        if g < qs[1]: return '2. Lower-middle (Q2)'
        if g < qs[2]: return '3. Upper-middle (Q3)'
        return '4. High (Q4)'
    panel['income_tier'] = panel['gdp_per_capita_ppp'].apply(tier)

    rows = []
    for tier_label, sub in list(panel.groupby('income_tier')) + [('All', panel)]:
        rows.append({
            'Tier':                tier_label,
            'Countries':           len(sub),
            'Athletes (sum)':      int(sub['athletes'].sum()),
            'Athletes per country (mean)':  round(sub['athletes'].mean(), 1),
            'GDP/cap PPP (median)':  int(sub['gdp_per_capita_ppp'].median()),
            'Pop, mil (median)':    round(sub['population'].median() / 1e6, 1),
            '15-24 pop, mil (median)': round(sub['pop_15_24'].median() / 1e6, 1),
            'Athletes / million pop': round(sub['athletes_per_million'].median(), 2),
            'Athletes / million 15-24': round(sub['athletes_per_million_15_24'].median(), 2),
            'Pol. stab. (median)':   round(sub['political_stability'].median(), 2),
            'Econ shocks 2010-23 (mean)': round(sub['econ_shocks_2010_23'].mean(), 2),
            'Disaster deaths 2010-23 (median)': int(sub['disaster_deaths_2010_23'].median()),
            'Years w/ conflict 2010-23 (mean)': round(sub['years_with_conflict_2010_23'].mean(), 1),
        })
    table1 = pd.DataFrame(rows).set_index('Tier')
    table1.to_csv(os.path.join(OUTPUT, 'table1_descriptives.csv'))
    print("=== Table 1: Descriptive statistics by income tier ===")
    print(table1.T.to_string())

    # ---------- Coefficient forest plot ----------
    # Re-fit the 5 main specs to extract coefs + CIs
    p = panel.copy()
    p['log_athletes']    = np.log(p['athletes'])
    p['log_gdp_pc']      = np.log(p['gdp_per_capita_ppp'])
    p['log_pop']         = np.log(p['population'])
    p['log_pop_15_24']   = np.log(p['pop_15_24'])
    p['log_athletes_per_million']     = np.log(p['athletes_per_million'])
    p['log_athletes_per_million_15_24'] = np.log(p['athletes_per_million_15_24'])
    p['log_disaster_deaths_2010_23']  = np.log1p(p['disaster_deaths_2010_23'])

    rhs_full = ['log_gdp_pc', 'political_stability',
                'econ_shocks_2010_23', 'log_disaster_deaths_2010_23']

    def fit(y_col, drop_top20=False):
        sub = p.dropna(subset=[y_col] + rhs_full).copy()
        if drop_top20:
            top = sub.nlargest(20, 'athletes')['country_code']
            sub = sub[~sub['country_code'].isin(top)]
        X = sm.add_constant(sub[rhs_full])
        return sm.OLS(sub[y_col], X).fit(cov_type='HC1')

    specs = {
        'OLS rate (total pop)':     fit('log_athletes_per_million'),
        'OLS rate (15-24 cohort)':  fit('log_athletes_per_million_15_24'),
        'OLS rate, drop top-20':    fit('log_athletes_per_million', drop_top20=True),
    }

    # Poisson and NegBin with offset(log_pop_15_24) and log_pop
    from statsmodels.discrete.discrete_model import Poisson, NegativeBinomial
    sub = p.dropna(subset=rhs_full + ['log_pop_15_24']).copy()
    Xc = sm.add_constant(sub[rhs_full])
    specs['Poisson (offset 15-24)'] = Poisson(sub['athletes'], Xc, offset=sub['log_pop_15_24']).fit(disp=False)
    try:
        specs['NegBin (offset 15-24)'] = NegativeBinomial(sub['athletes'], Xc, offset=sub['log_pop_15_24']).fit(disp=False)
    except Exception:
        pass

    # Build coefficient matrix for forest plot
    coefs_to_show = ['log_gdp_pc', 'political_stability',
                     'econ_shocks_2010_23', 'log_disaster_deaths_2010_23']
    forest_data = []
    for label, m in specs.items():
        for c in coefs_to_show:
            if c in m.params.index:
                est = m.params[c]
                se  = m.bse[c]
                forest_data.append({'spec': label, 'coef': c,
                                    'est': est, 'lo': est - 1.96 * se,
                                    'hi': est + 1.96 * se})
    fdf = pd.DataFrame(forest_data)

    # Plot
    fig, axes = plt.subplots(1, len(coefs_to_show), figsize=(15, 5), sharey=True)
    spec_order = list(specs.keys())
    y = list(range(len(spec_order)))
    for ax, c in zip(axes, coefs_to_show):
        sub = fdf[fdf['coef'] == c].set_index('spec').loc[spec_order]
        ax.errorbar(sub['est'], y, xerr=[sub['est'] - sub['lo'], sub['hi'] - sub['est']],
                    fmt='o', color='steelblue', capsize=3)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(spec_order, fontsize=9)
        ax.set_title(c, fontsize=10)
        ax.grid(True, axis='x', alpha=0.3)
    fig.suptitle('Coefficient stability across specifications (95% CI, HC1 SEs)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'fig_coef_forest.png'), dpi=150)
    plt.close(fig)
    print(f"\nWrote fig_coef_forest.png")

    # ---------- Final paper scatter: log_athletes_per_million_15_24 vs log_gdp ----------
    sub = p.dropna(subset=['log_gdp_pc', 'log_athletes_per_million_15_24',
                            'political_stability'])
    fig, ax = plt.subplots(figsize=(10, 7))
    # Color by political stability (quartiles)
    sub['ps_q'] = pd.qcut(sub['political_stability'], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
    colors = {'Q1 (low)': '#d73027', 'Q2': '#fc8d59', 'Q3': '#91bfdb', 'Q4 (high)': '#4575b4'}
    for q, dd in sub.groupby('ps_q'):
        ax.scatter(dd['log_gdp_pc'], dd['log_athletes_per_million_15_24'],
                   c=colors[q], label=f'Pol. stability {q}', alpha=0.75, s=50,
                   edgecolor='white')
    # Annotate top 12 senders
    for _, row in sub.sort_values('athletes', ascending=False).head(12).iterrows():
        ax.annotate(row['country_code'],
                    (row['log_gdp_pc'], row['log_athletes_per_million_15_24']),
                    xytext=(4, 4), textcoords='offset points', fontsize=9)
    ax.set_xlabel('log(home GDP per capita PPP, 2023)', fontsize=11)
    ax.set_ylabel('log(athletes per million age 15-24, 2018-22 avg)', fontsize=11)
    ax.set_title('NCAA international athletes per young-person, by home-country GDP and stability',
                 fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'fig_log_athletes_per_cohort.png'), dpi=150)
    plt.close(fig)
    print(f"Wrote fig_log_athletes_per_cohort.png")

    # ---------- Top 20 panel: count vs cohort-rate ----------
    top20 = sub.nlargest(20, 'athletes').sort_values('athletes', ascending=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 7))
    a1.barh(top20['country_code'], top20['athletes'], color='steelblue')
    a1.set_xlabel('Athletes (count)')
    a1.set_title('Top 20 by athlete count')
    rate20 = sub.nlargest(20, 'athletes_per_million_15_24').sort_values(
        'athletes_per_million_15_24', ascending=True)
    a2.barh(rate20['country_code'], rate20['athletes_per_million_15_24'], color='darkorange')
    a2.set_xlabel('Athletes per million age 15-24')
    a2.set_title('Top 20 by per-cohort rate')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'fig_top20_countries_panel.png'), dpi=150)
    plt.close(fig)
    print(f"Wrote fig_top20_countries_panel.png")


if __name__ == '__main__':
    main()
