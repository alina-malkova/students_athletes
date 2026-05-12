"""
Expanded sport-event analysis for the paper's §3.5.

Produces:
  (a) Athletes per event group + top-5 source countries per event.
  (b) Regional concentration per event (Herfindahl + share by region).
  (c) Within-event count regressions across all event groups
      (OLS log-count + NegBin where feasible).
  (d) A region x event matrix.
  (e) Country-event specialization: which countries are "outliers"
      for their event mix?

Output: output/event_analysis.txt and output/event_panel.csv
"""
from __future__ import annotations

import os
import re
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')

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


REGION = {
    **{c: 'SSA' for c in ['AGO','BDI','BEN','BFA','BWA','CAF','CIV','CMR','COD','COG',
                            'COM','CPV','DJI','ERI','ETH','GAB','GHA','GIN','GMB','GNB',
                            'GNQ','KEN','LBR','LSO','MDG','MLI','MOZ','MRT','MUS','MWI',
                            'NAM','NER','NGA','RWA','SDN','SEN','SLE','SOM','SSD','STP',
                            'SWZ','SYC','TCD','TGO','TZA','UGA','ZAF','ZMB','ZWE']},
    **{c: 'MENA' for c in ['ARE','BHR','DJI','DZA','EGY','IRN','IRQ','ISR','JOR','KWT',
                             'LBN','LBY','MAR','OMN','PSE','QAT','SAU','SYR','TUN','TUR',
                             'YEM']},
    **{c: 'SA' for c in ['AFG','BGD','BTN','IND','LKA','MDV','NPL','PAK']},
    **{c: 'EAP' for c in ['AUS','BRN','CHN','FJI','HKG','IDN','JPN','KHM','KIR','KOR',
                            'LAO','MAC','MMR','MNG','MYS','NCL','NZL','PHL','PLW','PNG',
                            'PRK','SGP','SLB','THA','TLS','TON','TWN','TUV','VNM','VUT',
                            'WSM']},
    **{c: 'LAC' for c in ['ARG','ATG','BHS','BLZ','BOL','BRA','BRB','CHL','COL','CRI',
                            'CUB','DMA','DOM','ECU','GRD','GTM','GUY','HND','HTI','JAM',
                            'KNA','LCA','MEX','NIC','PAN','PER','PRY','SLV','SUR','TTO',
                            'URY','VCT','VEN','VGB']},
    **{c: 'NA' for c in ['BMU','CAN','USA']},
    **{c: 'WEU' for c in ['AND','AUT','BEL','CHE','CYP','DEU','DNK','ESP','FIN','FRA',
                            'FRO','GBR','GIB','GRC','GRL','IMN','IRL','ISL','ITA','LIE',
                            'LUX','MCO','MLT','NLD','NOR','PRT','SMR','SWE','VAT']},
    **{c: 'EEU' for c in ['ALB','ARM','AZE','BGR','BIH','BLR','CZE','EST','GEO','HRV',
                            'HUN','KAZ','KGZ','LTU','LVA','MDA','MKD','MNE','POL','ROU',
                            'RUS','SRB','SVK','SVN','TJK','TKM','UKR','UZB','XKX']},
}


def country_panel(df):
    intl = df[(df['is_international'] == 1) & df['country_code'].notna()].copy()
    p = (intl.groupby('country_code')
             .agg(athletes_total=('school_name', 'count'),
                  gdp_per_capita_ppp=('gdp_per_capita_ppp', 'first'),
                  pop_15_24=('pop_15_24_avg_2018_22', 'first'),
                  political_stability=('political_stability', 'first'))
             .reset_index())
    p['log_pop_15_24']         = np.log(p['pop_15_24'])
    p['log_gdp_pc']            = np.log(p['gdp_per_capita_ppp'])
    p['region']                = p['country_code'].map(REGION).fillna('OTHER')
    return p


def main():
    warnings.filterwarnings('ignore')
    df = pd.read_csv(os.path.join(RAW_DATA, 'analysis_dataset.csv'), low_memory=False)
    intl = df[(df['is_international'] == 1) & df['country_code'].notna()].copy()
    intl['event_group'] = intl['position'].apply(classify_event)
    intl = intl[intl['event_group'] != 'Unknown'].copy()
    intl['region'] = intl['country_code'].map(REGION).fillna('OTHER')

    panel = country_panel(df)

    fh = open(os.path.join(OUTPUT, 'event_analysis.txt'), 'w')

    # -----------------------------------------------------------
    # (a) Athletes per event + top-5 source countries
    # -----------------------------------------------------------
    print('=== (a) Composition by event group ===')
    fh.write('=== (a) Composition by event group ===\n\n')
    n_total = len(intl)
    print(f'Total intl track athletes with event group: {n_total}')
    fh.write(f'Total intl track athletes with event group: {n_total}\n\n')

    by_evt = intl['event_group'].value_counts()
    print('\nAthletes per event group:')
    fh.write('Event group counts:\n')
    for evt, n in by_evt.items():
        line = f'  {evt:12s}  {n:>4d}  ({100*n/n_total:.1f}%)'
        print(line); fh.write(line + '\n')

    print('\nTop-5 source countries per event:')
    fh.write('\nTop-5 source countries per event:\n')
    for evt in by_evt.index:
        sub = intl[intl['event_group'] == evt]
        top = sub['country_code'].value_counts().head(5)
        share = top / len(sub) * 100
        countries = ', '.join(f'{c} ({n}, {s:.0f}%)' for (c, n), s in
                              zip(top.items(), share.values))
        line = f'  {evt:12s}  {countries}'
        print(line); fh.write(line + '\n')

    # -----------------------------------------------------------
    # (b) Regional concentration per event (Herfindahl)
    # -----------------------------------------------------------
    print('\n\n=== (b) Regional concentration per event ===')
    fh.write('\n\n=== (b) Regional concentration ===\n\n')
    fh.write('Herfindahl index across regions for each event '
             '(0 = perfectly diffuse, 1 = single region).\n\n')
    print('Region shares and Herfindahl per event (HHI normalised to [0,1]):')
    fh.write('  Event        HHI    Top region (share)\n')
    for evt in by_evt.index:
        sub = intl[intl['event_group'] == evt]
        shares = sub['region'].value_counts(normalize=True)
        hhi = (shares ** 2).sum()
        top_reg = shares.idxmax(); top_share = shares.max()
        line = f'  {evt:12s} {hhi:.3f}  {top_reg:5s} ({100*top_share:.0f}%)'
        print(line); fh.write(line + '\n')
    # Also a country-level HHI for context
    print('\nCountry-level Herfindahl per event (concentration among source countries):')
    fh.write('\nCountry-level concentration:\n')
    fh.write('  Event        HHI    Top-3 share\n')
    for evt in by_evt.index:
        sub = intl[intl['event_group'] == evt]
        cshares = sub['country_code'].value_counts(normalize=True)
        chhi = (cshares ** 2).sum()
        top3 = cshares.head(3).sum() * 100
        line = f'  {evt:12s} {chhi:.3f}  {top3:.0f}%'
        print(line); fh.write(line + '\n')

    # -----------------------------------------------------------
    # (c) Region x event matrix
    # -----------------------------------------------------------
    fh.write('\n\n=== (c) Region x event matrix (athlete counts) ===\n')
    mat = pd.crosstab(intl['region'], intl['event_group'])
    fh.write(mat.to_string() + '\n')
    fh.write('\nColumn shares (% of each event from each region):\n')
    fh.write((mat / mat.sum() * 100).round(1).to_string() + '\n')
    print('\nRegion x event column shares (%):')
    print((mat / mat.sum() * 100).round(1).to_string())

    # -----------------------------------------------------------
    # (d) Within-event count regressions for all groups
    # -----------------------------------------------------------
    print('\n\n=== (d) Within-event count regressions ===')
    fh.write('\n\n=== (d) Within-event count regressions ===\n')

    rows = []
    for evt in by_evt.index:
        sub_e = intl[intl['event_group'] == evt]
        cnt = sub_e.groupby('country_code').size().rename('athletes_evt').reset_index()
        pe = panel.merge(cnt, on='country_code', how='left')
        pe['athletes_evt'] = pe['athletes_evt'].fillna(0)
        senders = pe[pe['athletes_evt'] >= 1].copy()
        senders['log_athletes_evt'] = np.log(senders['athletes_evt'])
        rhs = ['log_pop_15_24', 'log_gdp_pc', 'political_stability']
        s = senders.dropna(subset=rhs + ['log_athletes_evt'])
        if len(s) < 12:
            fh.write(f'\n{evt}: n={len(s)} too small\n')
            continue
        # OLS
        X = sm.add_constant(s[rhs])
        m_ols = sm.OLS(s['log_athletes_evt'], X).fit(cov_type='HC1')
        # NegBin on raw athletes_evt
        try:
            X_nb = sm.add_constant(s[rhs])
            m_nb = NegativeBinomial(s['athletes_evt'], X_nb).fit(disp=False, maxiter=200)
        except Exception:
            m_nb = None
        fh.write(f'\n--- {evt}  n={len(s)} ---\n')
        fh.write('OLS log-count:\n')
        for k in m_ols.params.index:
            if k == 'const': continue
            b = m_ols.params[k]; se = m_ols.bse[k]; p = m_ols.pvalues[k]
            stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''
            fh.write(f'  {k:25s} {b:+.3f} (SE {se:.3f}) {stars}\n')
        fh.write(f'  R^2 = {m_ols.rsquared:.3f}\n')
        if m_nb is not None:
            fh.write('NegBin:\n')
            for k in m_nb.params.index:
                if k in ('const', 'alpha'): continue
                b = m_nb.params[k]; se = m_nb.bse[k]; p = m_nb.pvalues[k]
                stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''
                fh.write(f'  {k:25s} {b:+.3f} (SE {se:.3f}) {stars}\n')
        # Save row for headline table
        def fmt(m, k):
            if k not in m.params.index: return ''
            b = m.params[k]; se = m.bse[k]; p = m.pvalues[k]
            stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''
            return f'{b:+.2f}{stars}'
        rows.append({
            'event': evt, 'n': len(s),
            'ols_pop':       fmt(m_ols, 'log_pop_15_24'),
            'ols_gdp':       fmt(m_ols, 'log_gdp_pc'),
            'ols_polstab':   fmt(m_ols, 'political_stability'),
            'ols_R2':        round(m_ols.rsquared, 2),
            'nb_pop':        fmt(m_nb, 'log_pop_15_24') if m_nb else '',
            'nb_gdp':        fmt(m_nb, 'log_gdp_pc') if m_nb else '',
            'nb_polstab':    fmt(m_nb, 'political_stability') if m_nb else '',
        })

    headline = pd.DataFrame(rows)
    fh.write('\n\n=== Headline table: OLS and NegBin coefficients by event ===\n')
    fh.write(headline.to_string(index=False) + '\n')
    print('\nHeadline within-event table:')
    print(headline.to_string(index=False))
    headline.to_csv(os.path.join(OUTPUT, 'event_table.csv'), index=False)

    # -----------------------------------------------------------
    # (e) Country-event specialization: residuals
    # -----------------------------------------------------------
    fh.write('\n\n=== (e) Country-event specialization (residuals) ===\n')
    fh.write('Residual = actual log(athletes in event) minus predicted '
             'from country panel ~ log_pop_15_24 + log_gdp_pc + polstab.\n')
    fh.write('Top 10 positive residuals per event (countries supplying more than '
             'macro covariates predict):\n')
    for evt in ['Sprints', 'Distance', 'Throws', 'Jumps', 'MidDistance']:
        sub_e = intl[intl['event_group'] == evt]
        cnt = sub_e.groupby('country_code').size().rename('athletes_evt').reset_index()
        pe = panel.merge(cnt, on='country_code', how='left')
        pe['athletes_evt'] = pe['athletes_evt'].fillna(0)
        senders = pe[pe['athletes_evt'] >= 1].copy()
        senders['log_athletes_evt'] = np.log(senders['athletes_evt'])
        rhs = ['log_pop_15_24', 'log_gdp_pc', 'political_stability']
        s = senders.dropna(subset=rhs + ['log_athletes_evt'])
        if len(s) < 12: continue
        m = sm.OLS(s['log_athletes_evt'], sm.add_constant(s[rhs])).fit()
        s = s.copy()
        s['resid'] = m.resid
        top = s.nlargest(10, 'resid')[['country_code', 'region', 'athletes_evt', 'resid']]
        fh.write(f'\n--- {evt} ---\n')
        fh.write(top.to_string(index=False) + '\n')

    # Save event panel for paper Figure
    intl[['country_code', 'region', 'event_group']].to_csv(
        os.path.join(OUTPUT, 'event_panel.csv'), index=False)

    fh.close()
    print(f'\nWrote {os.path.join(OUTPUT, "event_analysis.txt")}')


if __name__ == '__main__':
    main()
