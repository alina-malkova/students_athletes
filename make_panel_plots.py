"""
Plots for the country × year × sport panel results:
  - fig_event_study_panel.png: COVID DiD leads/lags
  - fig_polstab_within_vs_between.png: pooled vs TWFE coefficient comparison
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')


def build_fe(panel, fe_cols):
    pieces = []
    for col in fe_cols:
        d = pd.get_dummies(panel[col], prefix=col, drop_first=True).astype(float)
        pieces.append(d)
    return pd.concat(pieces, axis=1)


def main():
    warnings.filterwarnings('ignore')
    panel = pd.read_csv(os.path.join(RAW_DATA, 'country_year_sport_panel.csv'))
    panel = panel[panel['country_code'] != 'USA'].copy()

    fe = build_fe(panel, ['country_code', 'year', 'sport'])
    rhs_lead = ['treat_x_y2020', 'treat_x_y2021', 'treat_x_y2022', 'treat_x_y2023']
    X = pd.concat([panel[rhs_lead], fe], axis=1)
    X = sm.add_constant(X)
    sub = X.dropna().index
    m = sm.OLS(panel.loc[sub, 'log_athletes'], X.loc[sub]).fit(
        cov_type='cluster', cov_kwds={'groups': panel.loc[sub, 'country_code']})

    # Plot 1: event-study leads/lags
    years = [2019, 2020, 2021, 2022, 2023]
    coefs = [0.0]
    lo = [0.0]; hi = [0.0]
    for y in [2020, 2021, 2022, 2023]:
        col = f'treat_x_y{y}'
        b = m.params[col]; se = m.bse[col]
        coefs.append(b); lo.append(b - 1.96 * se); hi.append(b + 1.96 * se)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(years, coefs,
                yerr=[[c - l for c, l in zip(coefs, lo)],
                      [h - c for c, h in zip(coefs, hi)]],
                fmt='o-', color='steelblue', capsize=4, markersize=9, lw=1.5)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.axvspan(2019.5, 2020.5, color='red', alpha=0.10, label='COVID year 2020')
    ax.set_xticks(years)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(r'Treated$_s \times \mathbb{1}\{t = k\}$ coefficient', fontsize=12)
    ax.set_title('COVID event study: soccer + volleyball vs other sports\n'
                  '(country × year × sport panel, country/year/sport FE, '
                  'SE clustered at country)', fontsize=11)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'fig_event_study_panel.png'), dpi=150)
    plt.close(fig)
    print(f'wrote fig_event_study_panel.png')

    # Plot 2: pooled vs TWFE coefficients for political_stability and friends
    rhs_p = ['log_pop_15_24', 'log_gdp_pc', 'political_stability',
             'log_disaster_events']
    sport_dums = pd.get_dummies(panel['sport'], prefix='sport',
                                 drop_first=True).astype(float)
    Xp = pd.concat([panel[rhs_p], sport_dums], axis=1)
    Xp = sm.add_constant(Xp)
    subp = Xp.dropna().index
    mp = sm.OLS(panel.loc[subp, 'log_athletes'], Xp.loc[subp]).fit(
        cov_type='cluster', cov_kwds={'groups': panel.loc[subp, 'country_code']})

    rhs_tw = ['political_stability', 'log_disaster_events',
              'gdp_shock', 'currency_crisis']
    Xt = pd.concat([panel[rhs_tw], fe], axis=1)
    Xt = sm.add_constant(Xt)
    subt = Xt.dropna().index
    mt = sm.OLS(panel.loc[subt, 'log_athletes'], Xt.loc[subt]).fit(
        cov_type='cluster', cov_kwds={'groups': panel.loc[subt, 'country_code']})

    # Side-by-side bar with CIs
    labels = ['log_pop_15_24', 'log_gdp_pc', 'political_stability',
              'log_disaster_events']
    pooled = []; tw = []; pooled_ci = []; tw_ci = []
    for k in labels:
        if k in mp.params.index:
            pooled.append(mp.params[k])
            pooled_ci.append(1.96 * mp.bse[k])
        else:
            pooled.append(np.nan); pooled_ci.append(np.nan)
        if k in mt.params.index:
            tw.append(mt.params[k]); tw_ci.append(1.96 * mt.bse[k])
        else:
            tw.append(0.0); tw_ci.append(0.0)
    xpos = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(xpos - width/2, pooled, width, yerr=pooled_ci,
            color='steelblue', alpha=0.85, label='Pooled OLS (with sport FE)',
            capsize=4)
    ax.bar(xpos + width/2, tw, width, yerr=tw_ci, color='darkorange',
            alpha=0.85, label='TWFE (country + year + sport FE)', capsize=4)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Coefficient (95% CI)')
    ax.set_title('Pooled OLS vs TWFE on the country×year×sport panel\n'
                  'Between-country variation absorbs the cross-sectional macro effects')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'fig_pooled_vs_twfe.png'), dpi=150)
    plt.close(fig)
    print(f'wrote fig_pooled_vs_twfe.png')


if __name__ == '__main__':
    main()
