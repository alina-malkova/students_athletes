"""
Main results for the revised paper, after the sublinear-scaling
finding.

Two-margin design:

  Extensive margin (probit on universe of 214 countries):
      Pr(country sends any athletes) on log_gdp_pc + log_pop + polstab

  Intensive margin (count model among 102 sender countries):
      log(athletes) on log_gdp_pc + log_pop_15_24 + polstab + shocks

We deliberately do NOT use the rate (athletes / pop) outcome because
that specification imposes a unit elasticity of athletes wrt
population. The data shows this elasticity is roughly 0.33 -- so
imposing 1.0 hides the dominant intensive-margin fact.

Output: output/main_results.txt
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')


def fit_print(label, model, fh):
    fh.write(f"\n{'=' * 76}\n{label}\n{'=' * 76}\n")
    fh.write(model.summary().as_text() + '\n')


def coef_brief(m, vars_):
    out = []
    for v in vars_:
        if v in m.params.index:
            b = m.params[v]
            se = m.bse[v]
            p = m.pvalues[v]
            stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
            out.append(f"{v}={b:+.3f}({se:.3f}){stars}")
    return out


def build_senders_panel(df: pd.DataFrame) -> pd.DataFrame:
    intl = df[(df['is_international'] == 1) & df['country_code'].notna()].copy()
    p = (intl.groupby('country_code')
             .agg(athletes=('school_name', 'count'),
                  gdp_per_capita_ppp=('gdp_per_capita_ppp', 'first'),
                  population=('population', 'first'),
                  pop_15_24=('pop_15_24_avg_2018_22', 'first'),
                  political_stability=('political_stability', 'first'),
                  econ_shocks_2010_23=('econ_shocks_2010_23', 'first'),
                  disaster_deaths_2010_23=('disaster_deaths_2010_23', 'first'),
                  disaster_events_2010_23=('disaster_events_2010_23', 'first'),
                  years_with_conflict_2010_23=('years_with_conflict_2010_23', 'first'),
                  refugees_plus_asylum_2010_23_total=('refugees_plus_asylum_2010_23_total', 'first'))
             .reset_index())
    p['log_athletes']      = np.log(p['athletes'])
    p['log_gdp_pc']        = np.log(p['gdp_per_capita_ppp'])
    p['log_pop']           = np.log(p['population'])
    p['log_pop_15_24']     = np.log(p['pop_15_24'])
    p['log_athletes_per_million_15_24'] = np.log(1e6 * p['athletes'] / p['pop_15_24'])
    p['log_disaster_deaths_2010_23']    = np.log1p(p['disaster_deaths_2010_23'])
    p['log_disaster_events_2010_23']    = np.log1p(p['disaster_events_2010_23'])
    p['log_refugees_2010_23']           = np.log1p(p['refugees_plus_asylum_2010_23_total'])
    return p


def build_universe() -> pd.DataFrame:
    macro = pd.read_csv(os.path.join(RAW_DATA, 'macro_combined.csv'))
    pop_total = pd.read_csv(os.path.join(RAW_DATA, 'wdi_population.csv'))
    cohort = pd.read_csv(os.path.join(RAW_DATA, 'wdi_population_15_24.csv'))
    polstab = pd.read_csv(os.path.join(RAW_DATA, 'political_stability.csv')).rename(
        columns={'iso_code': 'country_code'})
    m23 = macro[macro['year'] == 2023][['country_code', 'gdp_per_capita_ppp']]
    p23 = pop_total[pop_total['year'] == 2023][['country_code', 'population']]
    ps23 = polstab[polstab['year'] == 2023][['country_code', 'political_stability']]
    u = m23.merge(p23, on='country_code').merge(ps23, on='country_code', how='left'
        ).merge(cohort, on='country_code', how='left')
    u['log_gdp_pc']    = np.log(u['gdp_per_capita_ppp'])
    u['log_pop']       = np.log(u['population'])
    u['log_pop_15_24'] = np.log(u['pop_15_24_avg_2018_22'])
    return u


def main():
    warnings.filterwarnings('ignore')
    df = pd.read_csv(os.path.join(RAW_DATA, 'analysis_dataset.csv'), low_memory=False)
    panel = build_senders_panel(df)
    universe = build_universe()
    universe['sends_athletes'] = universe['country_code'].isin(panel['country_code']).astype(int)

    fh = open(os.path.join(OUTPUT, 'main_results.txt'), 'w')
    fh.write('MAIN RESULTS (after sublinear-scaling correction)\n')
    fh.write(f'  Senders panel:  n = {len(panel)}\n')
    fh.write(f'  Universe:       n = {len(universe)}\n')
    fh.write(f'  Senders / universe: {universe["sends_athletes"].sum()} / {len(universe)}\n')

    # ----------------------------------------------------------------
    # EXTENSIVE MARGIN: probit on universe
    # ----------------------------------------------------------------
    print('=== Extensive margin: probit on universe ===')
    rhs_ext = ['log_gdp_pc', 'log_pop', 'political_stability']
    sub_e = universe.dropna(subset=rhs_ext)
    Xe = sm.add_constant(sub_e[rhs_ext])
    probit = Probit(sub_e['sends_athletes'], Xe).fit(disp=False)
    fit_print('Extensive margin (probit): Pr(sends any athletes)', probit, fh)
    me = probit.get_margeff(method='dydx').summary_frame()
    fh.write('\n=== Probit average marginal effects ===\n' + me.to_string() + '\n')
    for k, v in probit.params.items():
        se = probit.bse[k]; p = probit.pvalues[k]
        stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
        print(f'  {k:25s} {v:+.3f} (SE {se:.3f}) {stars}')
    print(f'  pseudo R^2 = {probit.prsquared:.3f}, n = {int(probit.nobs)}')

    # ----------------------------------------------------------------
    # INTENSIVE MARGIN: count model with free log_pop_15_24
    # Series of specs adding controls
    # ----------------------------------------------------------------
    print('\n=== Intensive margin: count model (free pop elasticity) ===')

    def fit_count(extra=[]):
        rhs = ['log_pop_15_24', 'log_gdp_pc', 'political_stability'] + extra
        sub = panel.dropna(subset=['log_athletes'] + rhs).copy()
        X = sm.add_constant(sub[rhs])
        return sm.OLS(sub['log_athletes'], X).fit(cov_type='HC1'), rhs, sub

    # Spec (1): minimal -- size, wealth, stability
    m1, rhs1, sub1 = fit_count(extra=[])
    fit_print('(1) Count model: log_athletes ~ log_pop_15_24 + log_gdp_pc + polstab',
              m1, fh)

    # Spec (2): + disaster events
    m2, rhs2, sub2 = fit_count(extra=['log_disaster_events_2010_23'])
    fit_print('(2) + log(1+disaster events)', m2, fh)

    # Spec (3): full battery of shocks
    m3, rhs3, sub3 = fit_count(extra=['log_disaster_events_2010_23',
                                       'econ_shocks_2010_23',
                                       'years_with_conflict_2010_23',
                                       'log_refugees_2010_23'])
    fit_print('(3) + econ_shocks + conflict + refugees', m3, fh)

    # Spec (4): drop top-20 senders
    rhs4 = rhs2
    sub4 = panel.dropna(subset=['log_athletes'] + rhs4).copy()
    top20 = sub4.nlargest(20, 'athletes')['country_code']
    sub4 = sub4[~sub4['country_code'].isin(top20)]
    X4 = sm.add_constant(sub4[rhs4])
    m4 = sm.OLS(sub4['log_athletes'], X4).fit(cov_type='HC1')
    fit_print('(4) Drop top-20 senders', m4, fh)

    # Spec (5): show how the spurious rate model overstated polstab
    rhs5 = ['log_gdp_pc', 'political_stability', 'log_disaster_events_2010_23']
    sub5 = panel.dropna(subset=['log_athletes_per_million_15_24'] + rhs5).copy()
    X5 = sm.add_constant(sub5[rhs5])
    m5 = sm.OLS(sub5['log_athletes_per_million_15_24'], X5).fit(cov_type='HC1')
    fit_print('(5) Rate model with implicit unit-elasticity (the previous headline)',
              m5, fh)

    # Print headline table
    print()
    headline_cols = ['log_pop_15_24', 'log_gdp_pc', 'political_stability',
                     'log_disaster_events_2010_23', 'econ_shocks_2010_23',
                     'years_with_conflict_2010_23', 'log_refugees_2010_23']
    for label, m in [('(1) Spec base', m1), ('(2) +disasters', m2),
                     ('(3) +shocks  ', m3), ('(4) Drop top-20', m4),
                     ('(5) Rate (old)', m5)]:
        n = int(m.nobs); r2 = m.rsquared
        print(f'\n{label}  n={n}  R^2={r2:.3f}')
        for k in m.params.index:
            if k == 'const':
                continue
            v = m.params[k]; se = m.bse[k]; p = m.pvalues[k]
            stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
            print(f'    {k:35s} {v:+.3f} (SE {se:.3f}) {stars}')

    fh.close()
    print(f'\nWrote {os.path.join(OUTPUT, "main_results.txt")}')


if __name__ == '__main__':
    main()
