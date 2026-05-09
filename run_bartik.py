"""
Bartik-style shift-share design for the international NCAA athlete panel.

Logic
-----
For each country c, intl track athletes are distributed across event
groups (Sprints, Distance, Throws, ...). Different countries specialize
(Kenya in distance, Jamaica in sprints, Poland in throwing). Two
quantities matter:

  shares  s_{c,e}   the fraction of c's intl track athletes competing
                    in event group e.

  shifts  Delta_e   the leave-one-out aggregate intl supply in event
                    group e -- sum over all countries except c.

  Bartik  B_c = sum_e s_{c,e} * log(Delta_e)

A country with composition tilted toward high-flow events gets a high
B_c. The Bartik instrument is the part of c's predicted athlete count
that comes from "the average country's response to your event mix"
rather than country-specific factors.

We use B_c three ways:
  (i)   covariate alongside GDP, population, political stability,
        shocks;
  (ii)  IV for log(athletes) in 2SLS;
  (iii) decomposition: predicted (Bartik) vs residual.

Output: output/bartik_results.txt + a coefficient table appended to
output/robustness_results.txt.
"""
from __future__ import annotations

import os
import re
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')
os.makedirs(OUTPUT, exist_ok=True)


# ---------- Position → event group --------------------------------------

EVENT_PATTERNS = [
    ('PoleVault',    r'\bpole\s*vault|^pv$'),
    ('Hurdles',      r'\bhurdles?\b|\bhurd\b'),
    ('Multi',        r'\bmulti|\bdecath|\bheptath'),
    ('Sprints',      r'\bsprints?\b|^spr$|^s$|^sprint$'),
    ('MidDistance',  r'\bmid[\s\-]*dist|middle\s*distance|800|1500|mile|^md$'),
    ('Distance',     r'\bdistance\b|\bdist\.?\b|\b5k\b|\b10k\b|3000|^dis$|^d$'),
    ('XC',           r'\bxc\b|cross[\s\-]*country'),
    ('Jumps',        r'\bjumps?\b|high\s*jump|long\s*jump|triple\s*jump|^lj$|^hj$|^tj$'),
    ('Throws',       r'\bthrows?\b|javelin|shot\s*put|discus|hammer|^t$|^th$|^shot$'),
]


def classify_event(pos):
    if pd.isna(pos) or not str(pos).strip():
        return 'Unknown'
    s = str(pos).lower().strip()
    for grp, pat in EVENT_PATTERNS:
        if re.search(pat, s):
            return grp
    return 'Unknown'


def main():
    warnings.filterwarnings('ignore')

    df = pd.read_csv(os.path.join(RAW_DATA, 'analysis_dataset.csv'),
                     low_memory=False)

    # Track only -- events are track-specific.
    track = df[(df['sport'] == 'Track & Field') &
               (df['is_international'] == 1) &
               df['country_code'].notna()].copy()
    track['event_group'] = track['position'].apply(classify_event)
    print(f"Intl track athletes: {len(track)}; with event group:")
    print(track['event_group'].value_counts().to_string())

    # Drop the Unknown event group when computing shares; otherwise it
    # dominates for countries whose schools don't surface position data.
    # Also restrict to countries with at least MIN_N intl track athletes
    # so shares are not pathological 1-of-1 cells.
    MIN_N = 5
    track = track[track['event_group'] != 'Unknown']
    keep_countries = track.groupby('country_code').size()
    keep_countries = keep_countries[keep_countries >= MIN_N].index
    track = track[track['country_code'].isin(keep_countries)]
    print(f"\nAfter MIN_N={MIN_N} filter: {len(track)} athletes "
          f"from {track['country_code'].nunique()} countries")

    # ---------- Build shares matrix s_{c,e} ----------
    counts = (track.groupby(['country_code', 'event_group'])
                   .size().unstack(fill_value=0))
    # event totals per country
    totals = counts.sum(axis=1)
    shares = counts.div(totals, axis=0)
    print(f"\nShares matrix: {shares.shape} (countries x events)")

    # ---------- Leave-one-out shifts Delta_e ----------
    # Total intl track athletes in event e from all countries.
    event_totals = counts.sum(axis=0)
    # LOO: subtract country c's contribution to each event
    shifts_loo = pd.DataFrame(np.tile(event_totals.values, (len(counts), 1)),
                              index=counts.index, columns=counts.columns) - counts
    # Use log(shifts + 1) to keep additive form
    log_shifts = np.log1p(shifts_loo)

    # ---------- Bartik exposure B_c = sum_e s_{c,e} * log(Delta_e^LOO) ----------
    bartik = (shares.values * log_shifts.values).sum(axis=1)
    bartik_df = pd.DataFrame({
        'country_code':   counts.index,
        'bartik':         bartik,
        'track_athletes': totals.values,
    })
    print(f"\nBartik exposure: mean = {bartik.mean():.3f}, "
          f"sd = {bartik.std():.3f}, range = [{bartik.min():.3f}, {bartik.max():.3f}]")

    # Show top/bottom 5 countries by Bartik
    show = bartik_df.merge(track.groupby('country_code').first().reset_index()[
        ['country_code', 'gdp_per_capita_ppp', 'political_stability']],
        on='country_code', how='left')
    print("\nTop 10 by Bartik exposure:")
    print(show.sort_values('bartik', ascending=False).head(10).round(3).to_string(index=False))
    print("\nBottom 10 by Bartik exposure:")
    print(show.sort_values('bartik').head(10).round(3).to_string(index=False))

    # ---------- Build country panel ----------
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
    panel = panel.merge(bartik_df[['country_code', 'bartik']], on='country_code', how='left')

    # Logs
    panel['log_athletes']    = np.log(panel['athletes'])
    panel['log_gdp_pc']      = np.log(panel['gdp_per_capita_ppp'])
    panel['log_pop']         = np.log(panel['population'])
    panel['log_athletes_per_million']     = np.log(1e6 * panel['athletes'] / panel['population'])
    panel['log_athletes_per_million_15_24'] = np.log(1e6 * panel['athletes'] / panel['pop_15_24'])
    panel['log_disaster_deaths_2010_23'] = np.log1p(panel['disaster_deaths_2010_23'])
    panel = panel.dropna(subset=['log_gdp_pc', 'log_pop', 'bartik'])
    print(f"\nPanel for analysis: {len(panel)} countries")

    panel.to_csv(os.path.join(OUTPUT, 'bartik_panel.csv'), index=False)

    fh = open(os.path.join(OUTPUT, 'bartik_results.txt'), 'w')
    fh.write("Bartik shift-share results\n")
    fh.write("---------------------------\n")
    fh.write(f"Event groups: {list(counts.columns)}\n")
    fh.write(f"Countries:    {len(panel)}\n")
    fh.write(f"Track-event shares × log(LOO event totals).\n\n")

    # ---------- (i) Bartik as covariate ----------
    print("\n--- (i) Bartik as covariate ---")
    for outcome, label in [('log_athletes',                       'count'),
                           ('log_athletes_per_million',           'rate (total pop)'),
                           ('log_athletes_per_million_15_24',     'rate (15-24 cohort)')]:
        rhs = ['bartik', 'log_gdp_pc', 'log_pop', 'political_stability',
               'econ_shocks_2010_23', 'log_disaster_deaths_2010_23']
        if outcome.startswith('log_athletes_per_million'):
            rhs = [r for r in rhs if r != 'log_pop']
        sub = panel.dropna(subset=[outcome] + rhs).copy()
        X = sm.add_constant(sub[rhs])
        m = sm.OLS(sub[outcome], X).fit(cov_type='HC1')
        fh.write(f"\n=== Bartik as covariate ({label} outcome) ===\n")
        fh.write(m.summary().as_text() + '\n')
        print(f"\n{label} outcome  R^2={m.rsquared:.3f}  n={int(m.nobs)}")
        for k, v in m.params.items():
            se = m.bse[k]; p = m.pvalues[k]
            stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''
            print(f"  {k:35s} {v:8.3f} (SE {se:.3f}) {stars}")

    # ---------- (ii) 2SLS: instrument log_athletes with bartik ----------
    print("\n--- (ii) 2SLS: log_athletes_per_million ~ log_gdp_pc + log_pop  IV: bartik ---")
    # Endogenous: log_gdp_pc; instrument: bartik
    # Then control political_stability, etc. exogenously.
    exog = ['log_pop', 'political_stability', 'econ_shocks_2010_23',
            'log_disaster_deaths_2010_23']
    sub = panel.dropna(subset=['log_athletes', 'log_gdp_pc', 'bartik'] + exog).copy()
    Xexog = sm.add_constant(sub[exog])
    # Treat log_gdp_pc as endogenous, instrument with bartik
    endog = sub[['log_gdp_pc']]
    instruments = pd.concat([Xexog, sub[['bartik']]], axis=1)
    full_x = pd.concat([Xexog, endog], axis=1)
    iv_model = IV2SLS(sub['log_athletes'], full_x, instruments).fit()
    fh.write(f"\n=== 2SLS: log_athletes on log_gdp_pc (instrument = bartik) ===\n")
    fh.write(iv_model.summary().as_text() + '\n')
    print(f"\n2SLS log_athletes ~ log_gdp_pc (IV: bartik) + controls")
    for k in iv_model.params.index:
        v  = iv_model.params[k]
        se = iv_model.bse[k]
        p  = iv_model.pvalues[k]
        stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''
        print(f"  {k:35s} {v:8.3f} (SE {se:.3f}) {stars}")

    # First-stage R^2 / F-stat for instrument relevance
    Xfs = sm.add_constant(pd.concat([sub[exog], sub[['bartik']]], axis=1))
    yfs = sub['log_gdp_pc']
    fs = sm.OLS(yfs, Xfs).fit(cov_type='HC1')
    print(f"  First-stage  R^2={fs.rsquared:.3f}  bartik coef={fs.params['bartik']:.3f}  "
          f"t={fs.tvalues['bartik']:.2f}  F={fs.fvalue:.1f}")
    fh.write(f"\nFirst-stage: bartik on log_gdp_pc, R^2={fs.rsquared:.3f}, "
             f"bartik t={fs.tvalues['bartik']:.2f}\n")

    # ---------- (iii) Decomposition: predicted vs residual ----------
    print("\n--- (iii) Sport-mix decomposition: predicted vs residual ---")
    sub2 = panel.dropna(subset=['bartik', 'log_athletes']).copy()
    Xb = sm.add_constant(sub2[['bartik']])
    m_b = sm.OLS(sub2['log_athletes'], Xb).fit(cov_type='HC1')
    sub2['log_athletes_predicted'] = m_b.fittedvalues
    sub2['log_athletes_resid']     = m_b.resid
    print(f"  Bartik-only fit: R^2 = {m_b.rsquared:.3f}, slope = {m_b.params['bartik']:.3f}")
    fh.write(f"\nDecomposition: log_athletes ~ bartik alone.\n  R^2 = {m_b.rsquared:.3f}\n  "
             f"slope = {m_b.params['bartik']:.3f}\n")

    # Residual against shocks and political stability
    rhs2 = ['political_stability', 'econ_shocks_2010_23', 'log_disaster_deaths_2010_23',
            'years_with_conflict_2010_23']
    sub3 = sub2.dropna(subset=rhs2).copy()
    Xr = sm.add_constant(sub3[rhs2])
    m_r = sm.OLS(sub3['log_athletes_resid'], Xr).fit(cov_type='HC1')
    fh.write("\n=== Residual after Bartik regressed on shocks ===\n")
    fh.write(m_r.summary().as_text() + '\n')
    print(f"\nResidual ~ shocks (after stripping Bartik):  R^2 = {m_r.rsquared:.3f}")
    for k in m_r.params.index:
        v=m_r.params[k]; se=m_r.bse[k]; p=m_r.pvalues[k]
        stars='***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''
        print(f"  {k:35s} {v:8.3f} (SE {se:.3f}) {stars}")

    fh.close()
    print(f"\nWrote {os.path.join(OUTPUT, 'bartik_results.txt')}")


if __name__ == '__main__':
    main()
