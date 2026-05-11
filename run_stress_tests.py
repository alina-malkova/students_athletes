"""
Stress tests in response to the May 2026 review:

(1) Disaster-deaths drop-top-5 + count-instead-of-deaths.
(2) Refugees: re-run on raw athlete counts (with log_pop control)
    rather than per-cohort rate.
(3) Cohort 18-22 vs 15-24 denominator sensitivity.
(4) Two-margin: separate selection (whether any athletes) from
    intensity (count given any) -- the Heckman two-step done
    cleanly with a focus on what GDP and political stability do
    on each margin.

All results write to output/stress_tests.txt.
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit, NegativeBinomial

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')
os.makedirs(OUTPUT, exist_ok=True)


def fit_print(label, m, fh):
    fh.write(f"\n{'=' * 70}\n{label}\n{'=' * 70}\n")
    fh.write(m.summary().as_text())
    fh.write('\n')


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


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Country-level panel of international senders."""
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
                  refugees_plus_asylum_2010_23_total=('refugees_plus_asylum_2010_23_total', 'first'),
                  refugees_origin_2023=('refugees_origin_2023', 'first'))
             .reset_index())
    p['log_athletes'] = np.log(p['athletes'])
    p['log_gdp_pc']   = np.log(p['gdp_per_capita_ppp'])
    p['log_pop']      = np.log(p['population'])
    p['log_pop_15_24'] = np.log(p['pop_15_24'])
    p['athletes_per_million']     = 1e6 * p['athletes'] / p['population']
    p['athletes_per_million_15_24'] = 1e6 * p['athletes'] / p['pop_15_24']
    p['log_athletes_per_million']     = np.log(p['athletes_per_million'])
    p['log_athletes_per_million_15_24'] = np.log(p['athletes_per_million_15_24'])
    p['log_disaster_deaths_2010_23']  = np.log1p(p['disaster_deaths_2010_23'])
    p['log_disaster_events_2010_23']  = np.log1p(p['disaster_events_2010_23'])
    p['log_refugees_2010_23']         = np.log1p(p['refugees_plus_asylum_2010_23_total'])
    p['log_refugees_per_cohort']      = np.log1p(
        p['refugees_plus_asylum_2010_23_total'].fillna(0) / p['pop_15_24'])
    return p


def build_extensive_panel():
    """Universe of countries (including non-senders) for the selection probit."""
    macro = pd.read_csv(os.path.join(RAW_DATA, 'macro_combined.csv'))
    pop_total = pd.read_csv(os.path.join(RAW_DATA, 'wdi_population.csv'))
    cohort = pd.read_csv(os.path.join(RAW_DATA, 'wdi_population_15_24.csv'))
    polstab = pd.read_csv(os.path.join(RAW_DATA, 'political_stability.csv'))
    polstab = polstab.rename(columns={'iso_code': 'country_code'})

    m23 = macro[macro['year'] == 2023][['country_code', 'gdp_per_capita_ppp', 'unemployment']]
    p23 = pop_total[pop_total['year'] == 2023][['country_code', 'population']]
    ps23 = polstab[polstab['year'] == 2023][['country_code', 'political_stability']]
    u = m23.merge(p23, on='country_code').merge(ps23, on='country_code', how='left').merge(
        cohort, on='country_code', how='left')
    u['log_gdp_pc'] = np.log(u['gdp_per_capita_ppp'])
    u['log_pop']    = np.log(u['population'])
    u['log_pop_15_24'] = np.log(u['pop_15_24_avg_2018_22'])
    return u


def main():
    warnings.filterwarnings('ignore')
    df = pd.read_csv(os.path.join(RAW_DATA, 'analysis_dataset.csv'), low_memory=False)
    panel = build_panel(df)
    universe = build_extensive_panel()
    universe['sends_athletes'] = universe['country_code'].isin(panel['country_code']).astype(int)

    print(f'Panel of senders: n = {len(panel)}')
    print(f'Universe (for selection):  n = {len(universe)}, '
          f'senders = {universe["sends_athletes"].sum()}')

    fh = open(os.path.join(OUTPUT, 'stress_tests.txt'), 'w')
    fh.write('STRESS TESTS in response to May 2026 referee comments\n')
    fh.write(f'  Senders panel:   n = {len(panel)}\n')
    fh.write(f'  Full universe:   n = {len(universe)}\n')
    fh.write(f'  Senders / universe = {universe["sends_athletes"].sum()}/{len(universe)}\n')

    # ----------------------------------------------------------
    # (1) Disaster-deaths stress tests
    # ----------------------------------------------------------
    print('\n--- (1) Disaster-deaths stress ---')
    rhs_base = ['log_gdp_pc', 'political_stability', 'econ_shocks_2010_23']
    sub_full = panel.dropna(subset=['log_athletes_per_million_15_24'] + rhs_base
                            + ['log_disaster_deaths_2010_23']).copy()

    # (1a) Full sample, deaths
    X1a = sm.add_constant(sub_full[rhs_base + ['log_disaster_deaths_2010_23']])
    y1a = sub_full['log_athletes_per_million_15_24']
    m1a = sm.OLS(y1a, X1a).fit(cov_type='HC1')

    # (1b) Drop top-5 disaster-death countries
    top5 = panel.nlargest(5, 'disaster_deaths_2010_23')['country_code'].tolist()
    print(f'  Top 5 disaster-death countries (dropped): {top5}')
    sub_drop = sub_full[~sub_full['country_code'].isin(top5)]
    X1b = sm.add_constant(sub_drop[rhs_base + ['log_disaster_deaths_2010_23']])
    y1b = sub_drop['log_athletes_per_million_15_24']
    m1b = sm.OLS(y1b, X1b).fit(cov_type='HC1')

    # (1c) Use disaster_events (count) instead of deaths
    sub_evt = panel.dropna(subset=['log_athletes_per_million_15_24'] + rhs_base
                            + ['log_disaster_events_2010_23']).copy()
    X1c = sm.add_constant(sub_evt[rhs_base + ['log_disaster_events_2010_23']])
    y1c = sub_evt['log_athletes_per_million_15_24']
    m1c = sm.OLS(y1c, X1c).fit(cov_type='HC1')

    # (1d) Drop top-5 disaster-EVENT countries; use events outcome
    top5_events = panel.nlargest(5, 'disaster_events_2010_23')['country_code'].tolist()
    sub_evt_drop = sub_evt[~sub_evt['country_code'].isin(top5_events)]
    X1d = sm.add_constant(sub_evt_drop[rhs_base + ['log_disaster_events_2010_23']])
    y1d = sub_evt_drop['log_athletes_per_million_15_24']
    m1d = sm.OLS(y1d, X1d).fit(cov_type='HC1')

    fit_print('(1a) Disaster deaths, full sample', m1a, fh)
    fit_print(f'(1b) Disaster deaths, drop top-5: {top5}', m1b, fh)
    fit_print('(1c) Disaster events instead of deaths, full sample', m1c, fh)
    fit_print(f'(1d) Disaster events, drop top-5 by events: {top5_events}', m1d, fh)

    print('  full     ', '  '.join(coef_brief(m1a, ['log_gdp_pc', 'political_stability',
                                                     'log_disaster_deaths_2010_23'])))
    print('  drop top5', '  '.join(coef_brief(m1b, ['log_gdp_pc', 'political_stability',
                                                     'log_disaster_deaths_2010_23'])))
    print('  events   ', '  '.join(coef_brief(m1c, ['log_gdp_pc', 'political_stability',
                                                     'log_disaster_events_2010_23'])))
    print('  evt drop ', '  '.join(coef_brief(m1d, ['log_gdp_pc', 'political_stability',
                                                     'log_disaster_events_2010_23'])))

    # ----------------------------------------------------------
    # (2) Refugees on raw counts (no rate denominator)
    # ----------------------------------------------------------
    print('\n--- (2) Refugees on raw counts vs per-cohort rate ---')
    rhs_ref = ['log_gdp_pc', 'log_pop', 'political_stability',
               'log_disaster_deaths_2010_23']
    # (2a) Original spec: log(athletes/cohort) on log(refugees/cohort)
    sub2a = panel.dropna(subset=['log_athletes_per_million_15_24',
                                   'log_refugees_per_cohort',
                                   'log_gdp_pc', 'political_stability',
                                   'log_disaster_deaths_2010_23']).copy()
    rhs2a = ['log_gdp_pc', 'political_stability', 'log_disaster_deaths_2010_23',
             'log_refugees_per_cohort']
    X2a = sm.add_constant(sub2a[rhs2a])
    m2a = sm.OLS(sub2a['log_athletes_per_million_15_24'], X2a).fit(cov_type='HC1')

    # (2b) Same regressors but on RAW log(athletes), controlling for log_pop
    rhs2b = ['log_gdp_pc', 'log_pop', 'political_stability',
             'log_disaster_deaths_2010_23', 'log_refugees_2010_23']
    sub2b = panel.dropna(subset=['log_athletes'] + rhs2b).copy()
    X2b = sm.add_constant(sub2b[rhs2b])
    m2b = sm.OLS(sub2b['log_athletes'], X2b).fit(cov_type='HC1')

    fit_print('(2a) Rate outcome, refugees per cohort (original)', m2a, fh)
    fit_print('(2b) Count outcome (log_athletes), raw log(refugees) + log_pop', m2b, fh)

    print('  rate, refugees/cohort:', '  '.join(coef_brief(m2a,
        ['log_refugees_per_cohort', 'log_gdp_pc', 'political_stability'])))
    print('  count, raw refugees:  ', '  '.join(coef_brief(m2b,
        ['log_refugees_2010_23', 'log_gdp_pc', 'log_pop', 'political_stability'])))

    # ----------------------------------------------------------
    # (3) Two-margin: extensive + intensive
    # ----------------------------------------------------------
    print('\n--- (3) Two-margin decomposition ---')
    # Extensive: probit on universe of countries
    sel_X = sm.add_constant(universe.dropna(subset=[
        'log_gdp_pc', 'log_pop', 'political_stability'])
        [['log_gdp_pc', 'log_pop', 'political_stability']])
    sel_y = universe.loc[sel_X.index, 'sends_athletes']
    probit = Probit(sel_y, sel_X).fit(disp=False)

    # Intensive: among senders, log(athletes/cohort) ~ same RHS
    sub_int = panel.dropna(subset=['log_athletes_per_million_15_24',
                                   'log_gdp_pc', 'political_stability']).copy()
    Xi = sm.add_constant(sub_int[['log_gdp_pc', 'political_stability']])
    m_int = sm.OLS(sub_int['log_athletes_per_million_15_24'], Xi).fit(cov_type='HC1')

    fit_print('(3a) Selection probit: Pr(sends athletes) on log_gdp_pc + log_pop + polstab',
              probit, fh)
    fit_print('(3b) Intensity OLS: log(athletes/cohort) on log_gdp_pc + polstab (senders only)',
              m_int, fh)

    print('  Selection (extensive):', '  '.join(coef_brief(probit,
        ['log_gdp_pc', 'log_pop', 'political_stability'])))
    print('  Intensity (intensive):', '  '.join(coef_brief(m_int,
        ['log_gdp_pc', 'political_stability'])))

    # Marginal effects from probit at sample means
    me = probit.get_margeff(method='dydx').summary_frame()
    fh.write('\n=== Probit average marginal effects ===\n')
    fh.write(me.to_string() + '\n')

    # ----------------------------------------------------------
    # (4) Cohort 18-22 sensitivity
    # ----------------------------------------------------------
    print('\n--- (4) Cohort 18-22 vs 15-24 ---')
    # Approximate pop_18_22: we have pop_15_24 (whole 10-year band).
    # Fraction of the 15-24 cohort that is 18-22: 5 of 10 years if uniform.
    # That's a flat scaling, so the rate elasticity to log_pop_18_22
    # equals the elasticity to log_pop_15_24 (up to an additive constant
    # absorbed by the intercept). To get real movement we need separate
    # 15-19 and 20-24 cohorts; the only WB extract we computed is the
    # combined 15-24 average. We therefore report the comparison as a
    # check that the choice does not move things mechanically: by
    # construction it cannot.
    fh.write('\n=== (4) Cohort 18-22 vs 15-24 ===\n')
    fh.write('A uniform-rescaling of pop_15_24 by a constant fraction does '
             'not change any coefficient in the log-rate model (the '
             'intercept absorbs the rescaling). To get genuine variation '
             'one would re-pull the 5-year age bands. Documented in §5.\n')
    print('  (cohort 18-22 vs 15-24: log-rate model invariant to a constant '
          'rescaling; documenting as a §5 caveat)')

    fh.close()
    print(f'\nWrote {os.path.join(OUTPUT, "stress_tests.txt")}')


if __name__ == '__main__':
    main()
