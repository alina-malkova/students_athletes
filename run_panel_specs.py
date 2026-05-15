"""
Four panel specifications on the country × year × sport long panel.

(1) Pooled OLS baseline: log(1+athletes) ~ macro covariates, sport FE.
(2) Country FE + year FE (TWFE): identifies political_stability on
    within-country time variation; the central reviewer-flagged test.
(3) COVID DiD: soccer + volleyball treated 2020. Two variants -- a
    static DiD with post-2020 dummy and a dynamic event-study with
    year-by-year leads/lags.
(4) Heckman two-step on the panel:
    - Extensive probit: Pr(athletes > 0) at the country-year-sport cell.
    - Intensive OLS: log(athletes | >0) with inverse Mills ratio.

Output: output/panel_results.txt + headline coefficient table.
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
from scipy.stats import norm

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')


def fit_print(label, m, fh):
    fh.write(f"\n{'=' * 78}\n{label}\n{'=' * 78}\n")
    try:
        fh.write(m.summary().as_text() + '\n')
    except Exception:
        fh.write('(summary unavailable)\n')


def coef_brief(m, vars_, max_p=0.10):
    out = []
    for v in vars_:
        if v in m.params.index:
            b = m.params[v]; se = m.bse[v]; p = m.pvalues[v]
            stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''
            out.append(f"  {v:<35s} {b:+.3f} (SE {se:.3f}) {stars}")
    return '\n'.join(out)


def build_fixed_effects(panel, fe_cols):
    """Return a design matrix of FE dummies; drop one level per FE for identification."""
    pieces = []
    for col in fe_cols:
        d = pd.get_dummies(panel[col], prefix=col, drop_first=True).astype(float)
        pieces.append(d)
    return pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=panel.index)


def main():
    warnings.filterwarnings('ignore')
    panel = pd.read_csv(os.path.join(RAW_DATA, 'country_year_sport_panel.csv'))
    print(f'Panel: {panel.shape}')

    # Restrict to international-supply analysis: drop USA (it's the destination)
    panel = panel[panel['country_code'] != 'USA'].copy()
    print(f'After dropping USA: {panel.shape}')

    fh = open(os.path.join(OUTPUT, 'panel_results.txt'), 'w')
    fh.write('Country x year x sport panel: 4 specifications\n')
    fh.write(f'  Cells: {len(panel):,}\n')
    fh.write(f'  Countries: {panel["country_code"].nunique()}\n')
    fh.write(f'  Years: {sorted(panel["year"].unique())}\n')
    fh.write(f'  Sports: {sorted(panel["sport"].unique())}\n')

    # ----------------------------------------------------------------
    # (1) Pooled OLS
    # ----------------------------------------------------------------
    print('\n=== Spec 1: Pooled OLS ===')
    rhs_pooled = ['log_pop_15_24', 'log_gdp_pc', 'political_stability',
                  'any_conflict', 'log_disaster_events']
    sport_dums = pd.get_dummies(panel['sport'], prefix='sport',
                                 drop_first=True).astype(float)
    X = pd.concat([panel[rhs_pooled], sport_dums], axis=1)
    X = sm.add_constant(X)
    y = panel['log_athletes']
    sub = X.dropna().index
    m1 = sm.OLS(y.loc[sub], X.loc[sub]).fit(cov_type='cluster',
        cov_kwds={'groups': panel.loc[sub, 'country_code']})
    fit_print('(1) Pooled OLS with sport FE; SE clustered at country', m1, fh)
    print(coef_brief(m1, rhs_pooled))
    print(f'  n = {int(m1.nobs)}, R^2 = {m1.rsquared:.3f}')

    # ----------------------------------------------------------------
    # (2) Two-way FE: country FE + year FE (plus sport FE)
    # ----------------------------------------------------------------
    print('\n=== Spec 2: Country FE + Year FE + Sport FE (TWFE) ===')
    fe = build_fixed_effects(panel, ['country_code', 'year', 'sport'])
    rhs_twfe = ['political_stability', 'any_conflict', 'log_disaster_events',
                'gdp_shock', 'currency_crisis']
    X2 = pd.concat([panel[rhs_twfe], fe], axis=1)
    X2 = sm.add_constant(X2)
    sub2 = X2.dropna().index
    m2 = sm.OLS(y.loc[sub2], X2.loc[sub2]).fit(
        cov_type='cluster', cov_kwds={'groups': panel.loc[sub2, 'country_code']})
    fit_print('(2) TWFE (country + year + sport FE); SE clustered at country',
              m2, fh)
    print(coef_brief(m2, rhs_twfe))
    print(f'  n = {int(m2.nobs)}, R^2 = {m2.rsquared:.3f}')

    # ----------------------------------------------------------------
    # (3) COVID DiD: soccer + volleyball treated, other sports control
    # ----------------------------------------------------------------
    print('\n=== Spec 3a: COVID static DiD ===')
    rhs_did = ['did_term']
    X3 = pd.concat([panel[rhs_did], fe], axis=1)
    X3 = sm.add_constant(X3)
    sub3 = X3.dropna().index
    m3 = sm.OLS(y.loc[sub3], X3.loc[sub3]).fit(
        cov_type='cluster', cov_kwds={'groups': panel.loc[sub3, 'country_code']})
    fit_print('(3a) Static DiD: did_term = treated_sport x post_2020', m3, fh)
    print(coef_brief(m3, rhs_did))
    print(f'  n = {int(m3.nobs)}, R^2 = {m3.rsquared:.3f}')

    print('\n=== Spec 3b: COVID dynamic event study (leads/lags) ===')
    rhs_lead = ['treat_x_y2020', 'treat_x_y2021', 'treat_x_y2022', 'treat_x_y2023']
    X3b = pd.concat([panel[rhs_lead], fe], axis=1)
    X3b = sm.add_constant(X3b)
    sub3b = X3b.dropna().index
    m3b = sm.OLS(y.loc[sub3b], X3b.loc[sub3b]).fit(
        cov_type='cluster', cov_kwds={'groups': panel.loc[sub3b, 'country_code']})
    fit_print('(3b) Dynamic event study: treated_sport x year (base = 2019)',
              m3b, fh)
    print(coef_brief(m3b, rhs_lead, max_p=0.99))
    print(f'  n = {int(m3b.nobs)}, R^2 = {m3b.rsquared:.3f}')

    # ----------------------------------------------------------------
    # (4) Heckman two-step on the panel
    # ----------------------------------------------------------------
    print('\n=== Spec 4: Heckman two-step (extensive probit + intensive OLS) ===')
    # Extensive: probit on is_present
    rhs_sel = ['log_pop_15_24', 'log_gdp_pc', 'political_stability']
    sport_sel = pd.get_dummies(panel['sport'], prefix='sport', drop_first=True).astype(float)
    Xs = pd.concat([panel[rhs_sel], sport_sel], axis=1)
    Xs = sm.add_constant(Xs)
    sub_sel = Xs.dropna().index
    probit = Probit(panel.loc[sub_sel, 'is_present'].astype(int),
                     Xs.loc[sub_sel]).fit(disp=False)
    fit_print('(4a) Extensive margin probit', probit, fh)

    # Compute IMR
    xb = Xs.loc[sub_sel].dot(probit.params)
    imr = norm.pdf(xb) / norm.cdf(xb)
    panel.loc[sub_sel, 'imr'] = imr

    # Intensive: senders only, log(athletes) > 0
    senders = panel[(panel['is_present'] == 1) & panel['imr'].notna()].copy()
    senders['log_athletes_pos'] = np.log(senders['athletes'])
    fe_int = build_fixed_effects(senders, ['country_code', 'year', 'sport'])
    rhs_int = ['log_pop_15_24', 'log_gdp_pc', 'political_stability',
               'log_disaster_events', 'imr']
    X4 = pd.concat([senders[rhs_int], fe_int], axis=1)
    X4 = sm.add_constant(X4)
    sub4 = X4.dropna().index
    m4 = sm.OLS(senders.loc[sub4, 'log_athletes_pos'], X4.loc[sub4]).fit(
        cov_type='cluster', cov_kwds={'groups': senders.loc[sub4, 'country_code']})
    fit_print('(4b) Intensive margin: log(athletes>0) ~ macro + FE + IMR', m4, fh)
    print(coef_brief(m4, rhs_int, max_p=0.99))
    print(f'  n = {int(m4.nobs)}, R^2 = {m4.rsquared:.3f}')

    # ----------------------------------------------------------------
    # (5) Poisson PPML with country / year / sport FE (Chen-Roth 2024
    #     recommendation when many cells are zero; log(1+x) biases
    #     the slope toward zero)
    # ----------------------------------------------------------------
    print('\n=== Spec 5: Poisson PPML on raw counts ===')
    fe5 = build_fixed_effects(panel, ['country_code', 'year', 'sport'])
    rhs_ppml = ['political_stability', 'any_conflict', 'log_disaster_events',
                'gdp_shock', 'did_term']
    X5 = pd.concat([panel[rhs_ppml], fe5], axis=1)
    X5 = sm.add_constant(X5)
    sub5 = X5.dropna().index
    try:
        m5 = sm.GLM(panel.loc[sub5, 'athletes'], X5.loc[sub5],
                    family=sm.families.Poisson()).fit(
            cov_type='cluster', cov_kwds={'groups': panel.loc[sub5, 'country_code']},
            maxiter=200)
        fit_print('(5) Poisson PPML: athletes ~ shocks + DiD + country/year/sport FE',
                  m5, fh)
        for k in rhs_ppml:
            if k in m5.params.index:
                b = m5.params[k]; se = m5.bse[k]; p = m5.pvalues[k]
                stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''
                print(f'  {k:<35s} {b:+.3f} (SE {se:.3f}) {stars}')
        print(f'  n = {int(m5.nobs)}')
    except Exception as e:
        m5 = None
        print(f'  Poisson PPML failed: {e}')

    # 5b: PPML dynamic event study to complement Spec 3b
    print('\n=== Spec 5b: Poisson PPML dynamic event study ===')
    rhs5b = ['treat_x_y2020', 'treat_x_y2021', 'treat_x_y2022', 'treat_x_y2023']
    X5b = pd.concat([panel[rhs5b], fe5], axis=1)
    X5b = sm.add_constant(X5b)
    sub5b = X5b.dropna().index
    try:
        m5b = sm.GLM(panel.loc[sub5b, 'athletes'], X5b.loc[sub5b],
                    family=sm.families.Poisson()).fit(
            cov_type='cluster', cov_kwds={'groups': panel.loc[sub5b, 'country_code']},
            maxiter=200)
        fit_print('(5b) Poisson PPML dynamic event study (base year 2019)',
                  m5b, fh)
        for k in rhs5b:
            if k in m5b.params.index:
                b = m5b.params[k]; se = m5b.bse[k]; p = m5b.pvalues[k]
                stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''
                print(f'  {k:<35s} {b:+.3f} (SE {se:.3f}) {stars}')
        print(f'  n = {int(m5b.nobs)}')
    except Exception as e:
        m5b = None
        print(f'  Poisson PPML dynamic failed: {e}')

    fh.close()

    # ----------------------------------------------------------------
    # Console summary table
    # ----------------------------------------------------------------
    print('\n\n' + '=' * 70)
    print('HEADLINE COEFFICIENT TABLE')
    print('=' * 70)
    def fmt(m, k):
        if k not in m.params.index: return '   --   '
        b = m.params[k]; se = m.bse[k]; p = m.pvalues[k]
        stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''
        return f'{b:+.3f}{stars:<3}'

    print(f'{"":<28}{"(1) Pool":<13}{"(2) TWFE":<13}{"(3a) DiD":<13}{"(4) Heck-int":<13}{"(5) PPML":<13}')
    print('-' * 90)
    for k, label in [('log_pop_15_24', 'log_pop_15_24'),
                      ('log_gdp_pc',    'log_gdp_pc'),
                      ('political_stability', 'political_stability'),
                      ('log_disaster_events', 'log_disaster_events'),
                      ('any_conflict',    'any_conflict'),
                      ('gdp_shock',       'gdp_shock'),
                      ('did_term',        'did_term (Treated×Post)'),
                      ('imr',             'inverse Mills ratio')]:
        print(f'{label:<28}{fmt(m1,k):<13}{fmt(m2,k):<13}{fmt(m3,k):<13}{fmt(m4,k):<13}'
              f'{fmt(m5, k) if m5 is not None else "  --":<13}')
    print('-' * 90)
    n5 = int(m5.nobs) if m5 is not None else 0
    print(f'{"n":<28}{int(m1.nobs):<13}{int(m2.nobs):<13}{int(m3.nobs):<13}'
          f'{int(m4.nobs):<13}{n5:<13}')

    # Save panel with IMR
    panel.to_csv(os.path.join(RAW_DATA, 'country_year_sport_panel.csv'), index=False)


if __name__ == '__main__':
    main()
