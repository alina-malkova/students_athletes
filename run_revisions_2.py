"""
Six checks in response to the second-round review:

(1) Heckman two-step: probit selection + OLS intensive with IMR.
    Key number: what happens to log_pop_15_24 elasticity?
(2) Power analysis: minimum detectable effect (MDE) for the
    political-stability null at observed n and SE.
(3) Quantile regression at tau = 0.25, 0.50, 0.75.
(4) Negative Binomial with log_pop_15_24 as a free covariate
    (NOT offset) -- addresses overdispersion + low-count truncation
    of log(athletes).
(5) Within-event placebo: re-estimate count model within Sprints,
    Mid+Distance, Throws, Jumps.
(6) Region fixed effects: 7-region WB classification + Western/
    Eastern Europe split = 8 regions. Horse-race against macro
    covariates.

Output: output/revisions_2.txt
"""
from __future__ import annotations

import os
import re
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import Probit, NegativeBinomial
from statsmodels.regression.quantile_regression import QuantReg
from scipy.stats import norm

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')


# Region classification (8 regions: 7 WB + WEU/EEU split)
REGION = {
    # Sub-Saharan Africa
    **{c: 'SSA' for c in ['AGO','BDI','BEN','BFA','BWA','CAF','CIV','CMR','COD','COG',
                            'COM','CPV','DJI','ERI','ETH','GAB','GHA','GIN','GMB','GNB',
                            'GNQ','KEN','LBR','LSO','MDG','MLI','MOZ','MRT','MUS','MWI',
                            'NAM','NER','NGA','RWA','SDN','SEN','SLE','SOM','SSD','STP',
                            'SWZ','SYC','TCD','TGO','TZA','UGA','ZAF','ZMB','ZWE']},
    # Middle East & North Africa
    **{c: 'MENA' for c in ['ARE','BHR','DJI','DZA','EGY','IRN','IRQ','ISR','JOR','KWT',
                             'LBN','LBY','MAR','OMN','PSE','QAT','SAU','SYR','TUN','TUR',
                             'YEM']},
    # South Asia
    **{c: 'SA' for c in ['AFG','BGD','BTN','IND','LKA','MDV','NPL','PAK']},
    # East Asia & Pacific
    **{c: 'EAP' for c in ['AUS','BRN','CHN','FJI','HKG','IDN','JPN','KHM','KIR','KOR',
                            'LAO','MAC','MMR','MNG','MYS','NCL','NZL','PHL','PLW','PNG',
                            'PRK','SGP','SLB','THA','TLS','TON','TWN','TUV','VNM','VUT',
                            'WSM']},
    # Latin America & Caribbean
    **{c: 'LAC' for c in ['ARG','ATG','BHS','BLZ','BOL','BRA','BRB','CHL','COL','CRI',
                            'CUB','DMA','DOM','ECU','GRD','GTM','GUY','HND','HTI','JAM',
                            'KNA','LCA','MEX','NIC','PAN','PER','PRY','SLV','SUR','TTO',
                            'URY','VCT','VEN','VGB']},
    # North America
    **{c: 'NA' for c in ['BMU','CAN','USA']},
    # Western Europe
    **{c: 'WEU' for c in ['AND','AUT','BEL','CHE','CYP','DEU','DNK','ESP','FIN','FRA',
                            'FRO','GBR','GIB','GRC','GRL','IMN','IRL','ISL','ITA','LIE',
                            'LUX','MCO','MLT','NLD','NOR','PRT','SMR','SWE','VAT']},
    # Eastern Europe & Central Asia
    **{c: 'EEU' for c in ['ALB','ARM','AZE','BGR','BIH','BLR','CZE','EST','GEO','HRV',
                            'HUN','KAZ','KGZ','LTU','LVA','MDA','MKD','MNE','POL','ROU',
                            'RUS','SRB','SVK','SVN','TJK','TKM','UKR','UZB','XKX']},
}


# Event group classification (copied from run_bartik.py)
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


def build_senders_panel(df: pd.DataFrame) -> pd.DataFrame:
    intl = df[(df['is_international'] == 1) & df['country_code'].notna()].copy()
    p = (intl.groupby('country_code')
             .agg(athletes=('school_name', 'count'),
                  gdp_per_capita_ppp=('gdp_per_capita_ppp', 'first'),
                  population=('population', 'first'),
                  pop_15_24=('pop_15_24_avg_2018_22', 'first'),
                  political_stability=('political_stability', 'first'),
                  disaster_events_2010_23=('disaster_events_2010_23', 'first'))
             .reset_index())
    p['log_athletes']    = np.log(p['athletes'])
    p['log_gdp_pc']      = np.log(p['gdp_per_capita_ppp'])
    p['log_pop']         = np.log(p['population'])
    p['log_pop_15_24']   = np.log(p['pop_15_24'])
    p['log_disaster_events_2010_23'] = np.log1p(p['disaster_events_2010_23'])
    p['region']          = p['country_code'].map(REGION).fillna('OTHER')
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


def fit_print(label, m, fh):
    fh.write(f"\n{'='*78}\n{label}\n{'='*78}\n")
    fh.write(m.summary().as_text() + '\n')


def main():
    warnings.filterwarnings('ignore')
    df = pd.read_csv(os.path.join(RAW_DATA, 'analysis_dataset.csv'), low_memory=False)
    panel = build_senders_panel(df)
    universe = build_universe()
    universe['sends_athletes'] = universe['country_code'].isin(panel['country_code']).astype(int)

    print(f'Senders panel: n = {len(panel)}')
    print(f'Universe: n = {len(universe)}, senders {universe["sends_athletes"].sum()}')

    fh = open(os.path.join(OUTPUT, 'revisions_2.txt'), 'w')
    fh.write(f'Round-2 referee responses\n')
    fh.write(f'  Senders n = {len(panel)}\n')
    fh.write(f'  Universe n = {len(universe)}\n\n')

    # ----------------------------------------------------------------
    # (1) Heckman two-step
    # ----------------------------------------------------------------
    print('\n=== (1) Heckman two-step ===')
    # Stage 1: probit on universe
    sel_rhs = ['log_gdp_pc', 'log_pop', 'political_stability']
    sel = universe.dropna(subset=sel_rhs)
    Xsel = sm.add_constant(sel[sel_rhs])
    probit = Probit(sel['sends_athletes'], Xsel).fit(disp=False)
    # Compute IMR for the sender countries
    sel['xb'] = Xsel @ probit.params
    sel['imr'] = norm.pdf(sel['xb']) / norm.cdf(sel['xb'])

    # Stage 2: intensive-margin count regression with IMR
    senders = sel[sel['country_code'].isin(panel['country_code'])].merge(
        panel[['country_code', 'log_athletes', 'log_pop_15_24',
                'log_disaster_events_2010_23']],
        on='country_code', how='inner', suffixes=('_u', ''))
    rhs2 = ['log_pop_15_24', 'log_gdp_pc', 'political_stability',
            'log_disaster_events_2010_23', 'imr']
    s2 = senders.dropna(subset=['log_athletes'] + rhs2)
    X2 = sm.add_constant(s2[rhs2])
    m_heck = sm.OLS(s2['log_athletes'], X2).fit(cov_type='HC1')

    fit_print('(1a) Selection probit', probit, fh)
    fit_print('(1b) Intensive count + IMR (Heckman 2-step)', m_heck, fh)
    print('\nHeckman intensive-margin coefficients:')
    for k in m_heck.params.index:
        b = m_heck.params[k]; se = m_heck.bse[k]; p = m_heck.pvalues[k]
        stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''
        print(f'  {k:30s} {b:+.3f} (SE {se:.3f}) {stars}')
    print(f'  n = {int(m_heck.nobs)}, R^2 = {m_heck.rsquared:.3f}')

    # ----------------------------------------------------------------
    # (2) Power analysis: MDE for political_stability null
    # ----------------------------------------------------------------
    print('\n=== (2) Minimum detectable effect for political_stability ===')
    # Use the count model SE from the base spec
    base_rhs = ['log_pop_15_24', 'log_gdp_pc', 'political_stability']
    sb = panel.dropna(subset=['log_athletes'] + base_rhs).copy()
    Xb = sm.add_constant(sb[base_rhs])
    mb = sm.OLS(sb['log_athletes'], Xb).fit(cov_type='HC1')
    se_ps = mb.bse['political_stability']
    n = int(mb.nobs)
    # MDE at alpha=0.05 two-sided, power=0.80
    mde_80 = (norm.ppf(0.975) + norm.ppf(0.80)) * se_ps
    mde_50 = norm.ppf(0.975) * se_ps  # 50% power
    fh.write(f'\n=== (2) MDE political_stability ===\n')
    fh.write(f'  Observed SE = {se_ps:.3f}, n = {n}\n')
    fh.write(f'  MDE at 80% power, alpha=0.05 two-sided = {mde_80:.3f}\n')
    fh.write(f'  MDE at 50% power = {mde_50:.3f}\n')
    fh.write(f'  Interpretation: we can reject effects larger than {mde_80:.2f}\n')
    fh.write(f'  with 80% power; effects between 0 and {mde_80:.2f} are within\n')
    fh.write(f'  our power envelope.\n')
    print(f'  SE = {se_ps:.3f}, n = {n}')
    print(f'  MDE @ 80% power = {mde_80:.3f}, @ 50% power = {mde_50:.3f}')

    # ----------------------------------------------------------------
    # (3) Quantile regression
    # ----------------------------------------------------------------
    print('\n=== (3) Quantile regression at tau = 0.25, 0.50, 0.75 ===')
    Xq = sm.add_constant(sb[['log_pop_15_24', 'log_gdp_pc', 'political_stability']])
    yq = sb['log_athletes']
    qresults = {}
    for tau in [0.25, 0.50, 0.75]:
        try:
            mq = QuantReg(yq, Xq).fit(q=tau)
            qresults[tau] = mq
        except Exception as e:
            print(f'  tau={tau}: error {e}')
            continue
    fh.write(f'\n=== (3) Quantile regression ===\n')
    fh.write(f'  Variables: log_pop_15_24, log_gdp_pc, political_stability\n\n')
    for tau, mq in qresults.items():
        fh.write(f'\n--- tau = {tau:.2f} ---\n')
        fh.write(mq.summary().as_text() + '\n')
        print(f'\n  tau = {tau:.2f}:')
        for k in mq.params.index:
            if k == 'const': continue
            b = mq.params[k]; se = mq.bse[k]; p = mq.pvalues[k]
            stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''
            print(f'    {k:25s} {b:+.3f} (SE {se:.3f}) {stars}')

    # ----------------------------------------------------------------
    # (4) Negative Binomial with log_pop_15_24 as free covariate
    # ----------------------------------------------------------------
    print('\n=== (4) Negative Binomial with log_pop_15_24 as free covariate ===')
    rhs_nb = ['log_pop_15_24', 'log_gdp_pc', 'political_stability',
              'log_disaster_events_2010_23']
    snb = panel.dropna(subset=rhs_nb).copy()
    Xnb = sm.add_constant(snb[rhs_nb])
    try:
        nb = NegativeBinomial(snb['athletes'], Xnb).fit(disp=False, maxiter=200)
        fit_print('(4) NegBin: athletes ~ log_pop_15_24 + log_gdp_pc + polstab + disasters',
                  nb, fh)
        print('  NB coefficients (log link):')
        for k in nb.params.index:
            if k == 'alpha': continue
            b = nb.params[k]; se = nb.bse[k]; p = nb.pvalues[k]
            stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''
            print(f'    {k:30s} {b:+.3f} (SE {se:.3f}) {stars}')
    except Exception as e:
        print(f'  NB error: {e}')
        fh.write(f'\nNegBin failed: {e}\n')

    # ----------------------------------------------------------------
    # (5) Within-event-group placebo
    # ----------------------------------------------------------------
    print('\n=== (5) Within-event-group count specs ===')
    intl = df[(df['is_international'] == 1) & df['country_code'].notna()].copy()
    intl['event_group'] = intl['position'].apply(classify_event)
    # Big event groups
    bigs = ['Sprints', 'Distance', 'Throws', 'Jumps']
    for evt in bigs:
        sub = intl[intl['event_group'] == evt]
        cnt = sub.groupby('country_code').size().rename(f'athletes_{evt}').reset_index()
        pe = panel.merge(cnt, on='country_code', how='left')
        pe[f'athletes_{evt}'] = pe[f'athletes_{evt}'].fillna(0)
        pe = pe[pe[f'athletes_{evt}'] >= 1].copy()  # only senders in this event
        pe[f'log_athletes_{evt}'] = np.log(pe[f'athletes_{evt}'])
        rhs = ['log_pop_15_24', 'log_gdp_pc', 'political_stability']
        s = pe.dropna(subset=rhs + [f'log_athletes_{evt}'])
        if len(s) < 15:
            print(f'  {evt}: n={len(s)}, too small to fit')
            continue
        X = sm.add_constant(s[rhs])
        m_e = sm.OLS(s[f'log_athletes_{evt}'], X).fit(cov_type='HC1')
        fit_print(f'(5) Within-event count: {evt}, n = {len(s)}', m_e, fh)
        print(f'\n  {evt} (n={len(s)}, R^2={m_e.rsquared:.3f}):')
        for k in m_e.params.index:
            if k == 'const': continue
            b = m_e.params[k]; se = m_e.bse[k]; p = m_e.pvalues[k]
            stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''
            print(f'    {k:25s} {b:+.3f} (SE {se:.3f}) {stars}')

    # ----------------------------------------------------------------
    # (6) Region fixed effects horse-race
    # ----------------------------------------------------------------
    print('\n=== (6) Region FE horse-race ===')
    rhs_r = ['log_pop_15_24', 'log_gdp_pc', 'political_stability']
    sr = panel.dropna(subset=rhs_r + ['log_athletes']).copy()
    print(f'  region counts in panel: {sr["region"].value_counts().to_dict()}')

    # Spec A: macro only
    Xa = sm.add_constant(sr[rhs_r])
    mA = sm.OLS(sr['log_athletes'], Xa).fit(cov_type='HC1')
    # Spec B: macro + region dummies
    region_dums = pd.get_dummies(sr['region'], prefix='reg', drop_first=True).astype(float)
    Xb_ = sm.add_constant(pd.concat([sr[rhs_r], region_dums], axis=1))
    mB = sm.OLS(sr['log_athletes'], Xb_).fit(cov_type='HC1')
    # Spec C: region dummies only (R^2 ceiling from regions alone)
    Xc = sm.add_constant(region_dums)
    mC = sm.OLS(sr['log_athletes'], Xc).fit(cov_type='HC1')

    fit_print('(6A) Macro only (no region FE)', mA, fh)
    fit_print('(6B) Macro + 8 region FE', mB, fh)
    fit_print('(6C) Region FE only (no macro)', mC, fh)

    print(f'\n  (A) macro only:        R^2 = {mA.rsquared:.3f}')
    print(f'  (B) macro + region FE: R^2 = {mB.rsquared:.3f}')
    print(f'  (C) region FE only:    R^2 = {mC.rsquared:.3f}')
    print(f'\n  Region FE add R^2 = {mB.rsquared - mA.rsquared:.3f} over macro alone.')
    print(f'  Coefficient changes from (A) to (B):')
    for k in rhs_r:
        bA = mA.params[k]; bB = mB.params[k]
        print(f'    {k:25s} {bA:+.3f}  ->  {bB:+.3f}')

    fh.close()
    print(f'\nWrote {os.path.join(OUTPUT, "revisions_2.txt")}')


if __name__ == '__main__':
    main()
