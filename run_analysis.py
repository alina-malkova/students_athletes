"""
Country-level regression analysis on the international NCAA athlete data.

Question: do home-country economic conditions and shocks predict the
number of athletes a country sends to U.S. NCAA programs?

Builds a country-panel from analysis_dataset.csv (one row per source
country) and runs OLS specifications. Saves regression tables and plots
to output/.
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


def country_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate athlete-level data to country level."""
    intl = df[(df['is_international'] == 1) &
              (df['country_code'].notna())].copy()
    agg = (intl.groupby('country_code')
                .agg(athletes=('school_name', 'count'),
                     soccer_athletes=('sport', lambda s: (s == 'Soccer').sum()),
                     track_athletes=('sport', lambda s: (s == 'Track & Field').sum()),
                     gdp_per_capita_ppp=('gdp_per_capita_ppp', 'first'),
                     population=('population', 'first'),
                     pop_15_24=('pop_15_24_avg_2018_22', 'first'),
                     unemployment=('unemployment', 'first'),
                     gdp_growth=('gdp_growth', 'first'),
                     political_stability=('political_stability', 'first'),
                     rule_of_law=('rule_of_law', 'first'),
                     control_corruption=('control_corruption', 'first'),
                     years_with_conflict_2010_23=('years_with_conflict_2010_23', 'first'),
                     disaster_deaths_2010_23=('disaster_deaths_2010_23', 'first'),
                     econ_shocks_2010_23=('econ_shocks_2010_23', 'first'),
                     stringency_2020_21=('stringency_2020_21', 'first'))
                .reset_index())
    return agg


def add_log_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel['log_athletes'] = np.log(panel['athletes'])
    panel['log_gdp_pc']   = np.log(panel['gdp_per_capita_ppp'])
    panel['log_pop']      = np.log(panel['population'])
    panel['log_pop_15_24'] = np.log(panel['pop_15_24'])
    # rate per million total pop and rate per million 15-24 year olds
    panel['athletes_per_million']     = 1e6 * panel['athletes'] / panel['population']
    panel['athletes_per_million_15_24'] = 1e6 * panel['athletes'] / panel['pop_15_24']
    panel['log_athletes_per_million'] = np.log(panel['athletes_per_million'])
    panel['log_athletes_per_million_15_24'] = np.log(panel['athletes_per_million_15_24'])
    panel['log_disaster_deaths_2010_23'] = np.log1p(panel['disaster_deaths_2010_23'])
    return panel


def run_ols(y, X, label, fh):
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc, missing='drop').fit(cov_type='HC1')
    fh.write(f"\n{'=' * 70}\n{label}\n{'=' * 70}\n")
    fh.write(model.summary().as_text())
    fh.write('\n')
    return model


def main():
    warnings.filterwarnings('ignore')
    df = pd.read_csv(os.path.join(RAW_DATA, 'analysis_dataset.csv'),
                     low_memory=False)
    print(f"Analysis dataset: {df.shape}")

    panel = country_panel(df)
    panel = add_log_outcomes(panel)
    panel = panel.dropna(subset=['log_gdp_pc'])
    print(f"Country panel: {len(panel)} countries with at least one international athlete")
    print(panel.describe().round(2).to_string())

    panel.to_csv(os.path.join(OUTPUT, 'country_panel.csv'), index=False)

    # ---------- Regressions ----------
    out_path = os.path.join(OUTPUT, 'regression_results.txt')
    fh = open(out_path, 'w')
    fh.write(f"Regression results -- N = {len(panel)} countries\n")
    fh.write(f"Outcome: log(athletes from country in NCAA D-I rosters)\n")
    fh.write(f"Robust SEs (HC1).\n")

    y = panel['log_athletes']

    # Spec 1: GDP only (no population)
    m1 = run_ols(y, panel[['log_gdp_pc']], 'Spec 1: log_athletes ~ log_gdp_pc', fh)

    # Spec 2: + log_pop (population control)
    m2 = run_ols(y, panel[['log_gdp_pc', 'log_pop']],
                 'Spec 2: + log_pop (population control)', fh)

    # Spec 3: + governance + shocks + disasters (with pop control)
    X3 = panel[['log_gdp_pc', 'log_pop', 'political_stability',
                'econ_shocks_2010_23', 'log_disaster_deaths_2010_23']]
    m3 = run_ols(y, X3,
                 'Spec 3: + pop_control + political_stability + shocks + log(disaster deaths)', fh)

    # Spec 4: outcome = athletes per million population (rate, no scale confound)
    panel_rate = panel.dropna(subset=['log_athletes_per_million']).copy()
    Xr = panel_rate[['log_gdp_pc', 'political_stability',
                     'econ_shocks_2010_23', 'log_disaster_deaths_2010_23']]
    m4 = run_ols(panel_rate['log_athletes_per_million'], Xr,
                 'Spec 4: log(athletes per million) ~ log_gdp + governance + shocks',
                 fh)

    # Spec 4b: outcome = athletes per million age 15-24 (cohort-rate, the
    # right denominator for college-age athletes)
    panel_cohort = panel.dropna(subset=['log_athletes_per_million_15_24']).copy()
    Xc = panel_cohort[['log_gdp_pc', 'political_stability',
                       'econ_shocks_2010_23', 'log_disaster_deaths_2010_23']]
    m4b = run_ols(panel_cohort['log_athletes_per_million_15_24'], Xc,
                  'Spec 4b: log(athletes per million age 15-24)',
                  fh)

    # Spec 5: track-only with population control
    panel_track = panel.dropna(subset=['log_pop']).copy()
    panel_track['log_track'] = np.log(panel_track['track_athletes'].replace(0, np.nan))
    sub = panel_track.dropna(subset=['log_track'])
    m5 = run_ols(sub['log_track'],
                 sub[['log_gdp_pc', 'log_pop', 'political_stability',
                      'econ_shocks_2010_23', 'log_disaster_deaths_2010_23']],
                 'Spec 5: track-only outcome with population control', fh)

    # Spec 6: years_with_conflict instead of econ shocks
    X6 = panel[['log_gdp_pc', 'log_pop', 'political_stability',
                'years_with_conflict_2010_23', 'log_disaster_deaths_2010_23']]
    m6 = run_ols(y, X6, 'Spec 6: + years_with_conflict (pop-controlled)', fh)

    fh.close()
    print(f"\nWrote {out_path}")

    # ---------- Console summary ----------
    print("\n=== Headline coefficients (HC1 robust SEs) ===")
    for name, m in [('Spec 1 (GDP only)', m1),
                    ('Spec 2 (+log_pop)', m2),
                    ('Spec 3 (+governance/shocks)', m3),
                    ('Spec 4 (rate / total pop)', m4),
                    ('Spec 4b (rate / 15-24 cohort)', m4b),
                    ('Spec 5 (track only)', m5),
                    ('Spec 6 (+conflict)', m6)]:
        print(f"\n{name}:  R^2 = {m.rsquared:.3f}, n = {int(m.nobs)}")
        for k, v in m.params.items():
            se = m.bse[k]
            t = m.tvalues[k]
            p = m.pvalues[k]
            stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
            print(f"  {k:35s} {v:8.3f}  (SE {se:.3f}, t = {t:5.2f}) {stars}")

    # ---------- Sport-specific patterns ----------
    print("\n=== Mean home-country GDP by sport (international athletes only) ===")
    intl = df[df['is_international'] == 1].dropna(subset=['gdp_per_capita_ppp'])
    sport_stats = (intl.groupby('sport')
                       .agg(n=('school_name', 'count'),
                            mean_home_gdp=('gdp_per_capita_ppp', 'mean'),
                            median_home_gdp=('gdp_per_capita_ppp', 'median'),
                            mean_log_gdp_ratio=('log_gdp_ratio', 'mean'))
                       .round(0))
    print(sport_stats.to_string())
    sport_stats.to_csv(os.path.join(OUTPUT, 'home_gdp_by_sport.csv'))

    # ---------- Plots ----------
    print("\n=== Plots ===")

    # Plot 1: log_athletes vs log_gdp_pc with fitted line
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(panel['log_gdp_pc'], panel['log_athletes'],
               s=40, alpha=0.65, color='steelblue', edgecolor='white')
    # Fitted line from m1
    xs = np.linspace(panel['log_gdp_pc'].min(), panel['log_gdp_pc'].max(), 50)
    ys = m1.params['const'] + m1.params['log_gdp_pc'] * xs
    ax.plot(xs, ys, color='red', linewidth=1.5,
            label=f"OLS slope = {m1.params['log_gdp_pc']:.2f} (R^2 = {m1.rsquared:.2f})")
    # Annotate top countries
    for _, row in panel.sort_values('athletes', ascending=False).head(15).iterrows():
        ax.annotate(row['country_code'],
                    (row['log_gdp_pc'], row['log_athletes']),
                    xytext=(4, 4), textcoords='offset points', fontsize=8)
    ax.set_xlabel('log(home GDP per capita PPP, 2023)')
    ax.set_ylabel('log(athletes in NCAA D-I rosters)')
    ax.set_title(f'NCAA international athletes vs home GDP (n = {len(panel)} countries)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'fig_log_athletes_vs_log_gdp.png'), dpi=150)
    plt.close(fig)
    print(f"  wrote fig_log_athletes_vs_log_gdp.png")

    # Plot 2: residuals from spec 1 against political stability
    panel = panel.dropna(subset=['log_gdp_pc']).copy()
    Xc = sm.add_constant(panel[['log_gdp_pc']])
    panel['resid_spec1'] = sm.OLS(panel['log_athletes'], Xc, missing='drop').fit().resid
    sub = panel.dropna(subset=['political_stability'])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(sub['political_stability'], sub['resid_spec1'], alpha=0.6,
               color='darkorange', edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Political stability (WGI)')
    ax.set_ylabel('Residual from log_athletes ~ log_gdp_pc')
    ax.set_title('Do unstable countries send disproportionately many or few athletes?')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'fig_residuals_vs_polstab.png'), dpi=150)
    plt.close(fig)
    print(f"  wrote fig_residuals_vs_polstab.png")


if __name__ == '__main__':
    main()
