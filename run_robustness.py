"""
Robustness checks for the country-level NCAA international-athlete model.

(1) Drop top-20 sender countries; refit Spec 4 (rate outcome).
(2) Separate models for track and soccer outcomes.
(3) Heckman two-step selection model: who sends athletes at all?
(4) Poisson and Negative Binomial count models with log_pop offset.

Output: output/robustness_results.txt + a few plots.
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit, Poisson, NegativeBinomial
import matplotlib.pyplot as plt

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')
os.makedirs(OUTPUT, exist_ok=True)


def fit_print(label, model, fh):
    fh.write(f"\n{'=' * 72}\n{label}\n{'=' * 72}\n")
    fh.write(model.summary().as_text())
    fh.write('\n')


def coef_row(name, m, vars):
    parts = [name, f"n={int(m.nobs)}"]
    for v in vars:
        if v in m.params.index:
            est = m.params[v]
            se  = m.bse[v]
            p   = m.pvalues[v]
            stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
            parts.append(f"{v}={est:+.3f}({se:.3f}){stars}")
    return '  ' + '  '.join(parts)


def country_panel(df: pd.DataFrame) -> pd.DataFrame:
    intl = df[(df['is_international'] == 1) & df['country_code'].notna()].copy()
    p = (intl.groupby('country_code')
             .agg(athletes=('school_name', 'count'),
                  soccer_athletes=('sport', lambda s: (s == 'Soccer').sum()),
                  track_athletes=('sport', lambda s: (s == 'Track & Field').sum()),
                  gdp_per_capita_ppp=('gdp_per_capita_ppp', 'first'),
                  population=('population', 'first'),
                  political_stability=('political_stability', 'first'),
                  rule_of_law=('rule_of_law', 'first'),
                  control_corruption=('control_corruption', 'first'),
                  years_with_conflict_2010_23=('years_with_conflict_2010_23', 'first'),
                  disaster_deaths_2010_23=('disaster_deaths_2010_23', 'first'),
                  econ_shocks_2010_23=('econ_shocks_2010_23', 'first'))
             .reset_index())
    return p


def add_logs(p):
    p = p.copy()
    p['log_athletes'] = np.log(p['athletes'])
    p['log_gdp_pc']   = np.log(p['gdp_per_capita_ppp'])
    p['log_pop']      = np.log(p['population'])
    p['athletes_per_million']     = 1e6 * p['athletes'] / p['population']
    p['log_athletes_per_million'] = np.log(p['athletes_per_million'])
    p['log_disaster_deaths_2010_23'] = np.log1p(p['disaster_deaths_2010_23'])
    return p


def main():
    warnings.filterwarnings('ignore')
    df = pd.read_csv(os.path.join(RAW_DATA, 'analysis_dataset.csv'),
                     low_memory=False)
    panel = add_logs(country_panel(df)).dropna(subset=['log_gdp_pc', 'log_pop'])
    print(f"Country panel: {len(panel)} countries")

    # Macro side -- to build the Heckman selection sample
    macro = pd.read_csv(os.path.join(RAW_DATA, 'macro_combined.csv'))
    pop = pd.read_csv(os.path.join(RAW_DATA, 'wdi_population.csv'))
    polstab = pd.read_csv(os.path.join(RAW_DATA, 'political_stability.csv'))
    polstab = polstab.rename(columns={'iso_code': 'country_code'})

    macro_2023 = macro[macro['year'] == 2023][['country_code', 'gdp_per_capita_ppp', 'unemployment']]
    pop_2023 = pop[pop['year'] == 2023][['country_code', 'population']]
    ps_2023 = polstab[polstab['year'] == 2023][['country_code', 'political_stability']]

    universe = macro_2023.merge(pop_2023, on='country_code').merge(ps_2023, on='country_code', how='left')
    universe['log_gdp_pc'] = np.log(universe['gdp_per_capita_ppp'])
    universe['log_pop']    = np.log(universe['population'])
    universe['sends_athletes'] = universe['country_code'].isin(panel['country_code']).astype(int)
    print(f"Heckman universe: {len(universe)} countries, "
          f"{universe['sends_athletes'].sum()} sending athletes")

    fh = open(os.path.join(OUTPUT, 'robustness_results.txt'), 'w')
    fh.write('NCAA international-athlete model: robustness checks\n')

    summary_lines = []

    # ------------------------------------------------------------
    # (1) Outlier robustness: drop top-20 sender countries
    # ------------------------------------------------------------
    print('\n--- (1) Outlier robustness: drop top-20 sender countries ---')
    top20 = panel.nlargest(20, 'athletes')['country_code'].tolist()
    sub = panel[~panel['country_code'].isin(top20)].copy()
    print(f"  remaining countries: {len(sub)} (dropped {len(top20)} biggest senders)")

    # Rate spec (matches Spec 4 of run_analysis.py)
    Xr_full = sm.add_constant(panel.dropna(subset=[
        'log_gdp_pc', 'political_stability', 'econ_shocks_2010_23',
        'log_disaster_deaths_2010_23'])[['log_gdp_pc', 'political_stability',
        'econ_shocks_2010_23', 'log_disaster_deaths_2010_23']])
    yr_full = panel.loc[Xr_full.index, 'log_athletes_per_million']
    m_full = sm.OLS(yr_full, Xr_full).fit(cov_type='HC1')

    Xr_sub = sm.add_constant(sub.dropna(subset=[
        'log_gdp_pc', 'political_stability', 'econ_shocks_2010_23',
        'log_disaster_deaths_2010_23'])[['log_gdp_pc', 'political_stability',
        'econ_shocks_2010_23', 'log_disaster_deaths_2010_23']])
    yr_sub = sub.loc[Xr_sub.index, 'log_athletes_per_million']
    m_sub = sm.OLS(yr_sub, Xr_sub).fit(cov_type='HC1')

    fit_print('(1) Rate spec on full sample',                  m_full, fh)
    fit_print('(1) Rate spec, top-20 senders dropped',         m_sub,  fh)

    summary_lines.append('### (1) Outlier robustness (rate outcome)')
    rvars = ['log_gdp_pc', 'political_stability',
             'econ_shocks_2010_23', 'log_disaster_deaths_2010_23']
    summary_lines.append(coef_row('full     ', m_full, rvars))
    summary_lines.append(coef_row('-top20   ', m_sub,  rvars))

    # ------------------------------------------------------------
    # (2) Sport-specific models
    # ------------------------------------------------------------
    print('\n--- (2) Sport-specific models ---')
    p_track = panel[panel['track_athletes'] > 0].copy()
    p_track['log_track']     = np.log(p_track['track_athletes'])
    p_track['log_track_per_million'] = np.log(1e6 * p_track['track_athletes']
                                              / p_track['population'])
    p_soccer = panel[panel['soccer_athletes'] > 0].copy()
    p_soccer['log_soccer']   = np.log(p_soccer['soccer_athletes'])
    p_soccer['log_soccer_per_million'] = np.log(1e6 * p_soccer['soccer_athletes']
                                                / p_soccer['population'])
    print(f"  track panel:  {len(p_track)} countries")
    print(f"  soccer panel: {len(p_soccer)} countries")

    base_rhs = ['log_gdp_pc', 'log_pop', 'political_stability',
                'log_disaster_deaths_2010_23']

    def ols_on(p, y_col):
        X = sm.add_constant(p.dropna(subset=base_rhs)[base_rhs])
        y = p.loc[X.index, y_col]
        return sm.OLS(y, X).fit(cov_type='HC1')

    m_track_count    = ols_on(p_track,  'log_track')
    m_track_rate     = ols_on(p_track,  'log_track_per_million')
    m_soccer_count   = ols_on(p_soccer, 'log_soccer')
    m_soccer_rate    = ols_on(p_soccer, 'log_soccer_per_million')

    fit_print('(2) Track  count outcome',  m_track_count,  fh)
    fit_print('(2) Track  rate  outcome',  m_track_rate,   fh)
    fit_print('(2) Soccer count outcome',  m_soccer_count, fh)
    fit_print('(2) Soccer rate  outcome',  m_soccer_rate,  fh)

    summary_lines.append('\n### (2) Sport-specific models')
    sport_vars = ['log_gdp_pc', 'log_pop', 'political_stability',
                  'log_disaster_deaths_2010_23']
    summary_lines.append(coef_row('track  count ', m_track_count,  sport_vars))
    summary_lines.append(coef_row('track  rate  ', m_track_rate,   ['log_gdp_pc',
                                  'political_stability', 'log_disaster_deaths_2010_23']))
    summary_lines.append(coef_row('soccer count ', m_soccer_count, sport_vars))
    summary_lines.append(coef_row('soccer rate  ', m_soccer_rate,  ['log_gdp_pc',
                                  'political_stability', 'log_disaster_deaths_2010_23']))

    # ------------------------------------------------------------
    # (3) Heckman two-step selection model
    # ------------------------------------------------------------
    print('\n--- (3) Heckman two-step selection model ---')
    sel_X = sm.add_constant(universe.dropna(subset=[
        'log_gdp_pc', 'log_pop', 'political_stability'
    ])[['log_gdp_pc', 'log_pop', 'political_stability']])
    sel_y = universe.loc[sel_X.index, 'sends_athletes']
    probit = Probit(sel_y, sel_X).fit(disp=False)
    fit_print('(3) Probit selection: P(sends athletes)', probit, fh)

    # Inverse Mills Ratio
    from scipy.stats import norm
    universe['_xb'] = sm.add_constant(universe[['log_gdp_pc', 'log_pop',
                                                 'political_stability']]).dot(probit.params)
    universe['imr'] = norm.pdf(universe['_xb']) / norm.cdf(universe['_xb'])

    # Outcome equation on senders, with IMR
    senders = universe.merge(panel[['country_code', 'log_athletes',
                                     'log_athletes_per_million']],
                              on='country_code', how='inner')
    Xo = sm.add_constant(senders.dropna(subset=[
        'log_gdp_pc', 'log_pop', 'political_stability', 'imr'
    ])[['log_gdp_pc', 'log_pop', 'political_stability', 'imr']])
    yo = senders.loc[Xo.index, 'log_athletes']
    m_outcome = sm.OLS(yo, Xo).fit(cov_type='HC1')
    fit_print('(3) Outcome equation: log_athletes with IMR', m_outcome, fh)

    summary_lines.append('\n### (3) Heckman selection')
    summary_lines.append(coef_row('selection (probit)', probit,
                                  ['log_gdp_pc', 'log_pop', 'political_stability']))
    summary_lines.append(coef_row('outcome with IMR  ', m_outcome,
                                  ['log_gdp_pc', 'log_pop', 'political_stability', 'imr']))

    # ------------------------------------------------------------
    # (4) Poisson + Negative Binomial with log_pop as offset
    # ------------------------------------------------------------
    print('\n--- (4) Poisson + Negative Binomial ---')
    rhs_count = ['log_gdp_pc', 'political_stability',
                 'econ_shocks_2010_23', 'log_disaster_deaths_2010_23']
    pcount = panel.dropna(subset=rhs_count + ['log_pop']).copy()
    Xc = sm.add_constant(pcount[rhs_count])
    offset = pcount['log_pop']
    yc = pcount['athletes']

    poisson  = Poisson(yc, Xc, offset=offset).fit(disp=False)
    fit_print('(4) Poisson: athletes ~ ... + offset(log_pop)', poisson, fh)

    try:
        negbin = NegativeBinomial(yc, Xc, offset=offset).fit(disp=False)
        fit_print('(4) Negative Binomial: athletes ~ ... + offset(log_pop)',
                  negbin, fh)
    except Exception as e:
        negbin = None
        fh.write(f"\n(4) Negative Binomial: failed -- {e}\n")

    summary_lines.append('\n### (4) Count models with log_pop offset')
    summary_lines.append(coef_row('Poisson         ', poisson, rhs_count))
    if negbin is not None:
        summary_lines.append(coef_row('NegativeBinomial', negbin, rhs_count))

    # ------------------------------------------------------------
    fh.close()

    # Console summary
    print('\n' + '\n'.join(summary_lines))
    print(f"\nFull tables written to {os.path.join(OUTPUT, 'robustness_results.txt')}")


if __name__ == '__main__':
    main()
