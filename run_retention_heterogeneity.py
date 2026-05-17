"""
Heterogeneity extensions to §3.8 retention-vs-shocks:

  (A) Region splits — does the conflict / polstab effect concentrate
      in any of the 8 regions of origin?
  (B) Lagged shocks — does shock_{t-1} predict return_{t,t+1} after
      controlling for shock_t? (Persistence; one-year lag in response.)
  (C) Intra-NCAA transfer — distinguish "drop out of NCAA" from
      "transfer to another U.S. school" by checking whether the
      athlete reappears at *any* school in year t+1.
"""
from __future__ import annotations

import os
import glob
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')

REGION = {  # (subset reused from run_position_analysis.py)
    **{c: 'SSA' for c in ['AGO','BDI','BEN','BFA','BWA','CAF','CIV','CMR','COD','COG',
                           'COM','CPV','DJI','ERI','ETH','GAB','GHA','GIN','GMB','GNB',
                           'GNQ','KEN','LBR','LSO','MDG','MLI','MOZ','MRT','MUS','MWI',
                           'NAM','NER','NGA','RWA','SDN','SEN','SLE','SOM','SSD','STP',
                           'SWZ','SYC','TCD','TGO','TZA','UGA','ZAF','ZMB','ZWE']},
    **{c: 'MENA' for c in ['ARE','BHR','DJI','DZA','EGY','IRN','IRQ','ISR','JOR','KWT',
                            'LBN','LBY','MAR','OMN','PSE','QAT','SAU','SYR','TUN','TUR','YEM']},
    **{c: 'SA' for c in ['AFG','BGD','BTN','IND','LKA','MDV','NPL','PAK']},
    **{c: 'EAP' for c in ['AUS','BRN','CHN','FJI','HKG','IDN','JPN','KHM','KIR','KOR',
                           'LAO','MAC','MMR','MNG','MYS','NCL','NZL','PHL','PLW','PNG',
                           'PRK','SGP','SLB','THA','TLS','TON','TWN','TUV','VNM','VUT','WSM']},
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


def norm_class(c):
    if pd.isna(c): return 'UNK'
    s = str(c).upper().strip().replace('.', '')
    if s in {'FR','SO','JR','FY','R-FR','R-SO','R-JR',
             'FRESHMAN','SOPHOMORE','JUNIOR'}: return 'UNDER'
    if s in {'SR','R-SR','5TH','GR','SENIOR'}: return 'SENIOR'
    return 'UNK'


def load_panel():
    files = sorted(glob.glob(os.path.join(RAW_DATA, '*_rosters_panel.csv')))
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if 'sport' not in df.columns: df['sport'] = 'Track'
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    for col, k in [('name', 'name_k'), ('school_name', 'school_k'),
                    ('gender', 'gender_k'), ('sport', 'sport_k')]:
        panel[k] = (panel[col].fillna('').str.upper()
                     .str.replace(r'\s+', ' ', regex=True).str.strip())
    panel['athlete_id_school'] = (panel['school_k'] + '|' + panel['gender_k']
                                   + '|' + panel['sport_k'] + '|'
                                   + panel['name_k'])
    panel['athlete_id_xfer']   = (panel['gender_k'] + '|' + panel['sport_k']
                                   + '|' + panel['name_k'])  # any school
    panel['class_norm'] = panel['class_year'].apply(norm_class)
    panel['year'] = pd.to_numeric(panel['year'], errors='coerce').astype('Int64')
    panel = panel[panel['year'].notna()].copy()
    panel['year'] = panel['year'].astype(int)
    panel['is_intl'] = ((panel['country_code'].notna()) &
                       (panel['country_code'] != 'USA')).astype(int)
    panel['region']  = panel['country_code'].map(REGION).fillna('OTHER')
    return panel


def add_retention_flags(panel):
    seen_school = panel.groupby('athlete_id_school')['year'].apply(set).to_dict()
    seen_any    = panel.groupby('athlete_id_xfer')['year'].apply(set).to_dict()
    panel['return_same'] = panel.apply(
        lambda r: int((r['year']+1) in seen_school.get(r['athlete_id_school'], set())),
        axis=1)
    panel['return_anywhere'] = panel.apply(
        lambda r: int((r['year']+1) in seen_any.get(r['athlete_id_xfer'], set())),
        axis=1)
    # Transfer = appears next year but at different school
    panel['transferred'] = ((panel['return_anywhere'] == 1) &
                            (panel['return_same'] == 0)).astype(int)
    return panel


def load_shocks():
    p = pd.read_csv(os.path.join(RAW_DATA, 'country_year_sport_panel.csv'))
    cols = ['country_code', 'year', 'gdp_shock', 'currency_crisis',
            'any_conflict', 'political_stability']
    p = p[[c for c in cols if c in p.columns]].drop_duplicates(
        subset=['country_code', 'year'])
    m = p['political_stability'].mean(); s = p['political_stability'].std()
    p['polstab_drop'] = ((p['political_stability'] - m) / s < -1).astype(int)
    # add t-1 lag of shocks
    p = p.sort_values(['country_code', 'year'])
    for sv in ['any_conflict', 'polstab_drop', 'gdp_shock',
               'currency_crisis']:
        if sv in p.columns:
            p[sv + '_lag1'] = p.groupby('country_code')[sv].shift(1)
    return p


def run(sub, sv, outcome, cluster='country_code'):
    sub = sub[sub[sv].notna() & sub[outcome].notna()].copy()
    if len(sub) < 200: return None
    X = pd.concat([
        sub[[sv]].astype(float),
        pd.get_dummies(sub['year'], prefix='y', drop_first=True).astype(float),
        pd.get_dummies(sub['sport'], prefix='sport',
                        drop_first=True).astype(float),
    ], axis=1)
    X = sm.add_constant(X)
    idx = X.dropna().index
    try:
        m = sm.OLS(sub.loc[idx, outcome].astype(float),
                   X.loc[idx]).fit(
              cov_type='cluster',
              cov_kwds={'groups': sub.loc[idx, cluster]})
        b = m.params[sv]; se = m.bse[sv]; p = m.pvalues[sv]
        stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        return {'beta': round(b,4), 'se': round(se,4),
                'p': round(p,4), 'stars': stars, 'N': len(idx)}
    except Exception:
        return None


def main():
    warnings.filterwarnings('ignore')
    print('Loading panel...')
    panel = load_panel()
    panel = add_retention_flags(panel)
    work = panel[panel['class_norm'] == 'UNDER'].copy()
    work = work[work['year'] < 2023].copy()
    shocks = load_shocks()
    work = work.merge(shocks, on=['country_code', 'year'], how='left')
    intl = work[work['is_intl'] == 1].copy()
    print(f'  intl under-classmen N = {len(intl):,}')

    fh = open(os.path.join(OUTPUT, 'retention_heterogeneity.txt'), 'w')
    fh.write('§3.8 retention — heterogeneity extensions\n')
    fh.write('=' * 64 + '\n\n')

    # (A) Region splits ------------------------------------------------
    print('\n--- (A) Region splits (conflict + polstab) ---')
    fh.write('(A) Region splits\n')
    region_rows = []
    for reg in sorted(intl['region'].unique()):
        if reg == 'OTHER': continue
        sub = intl[intl['region'] == reg]
        for sv in ['any_conflict', 'polstab_drop']:
            r = run(sub, sv, 'return_same')
            if r:
                region_rows.append({'region': reg, 'shock': sv, **r})
    rdf = pd.DataFrame(region_rows)
    print(rdf.to_string(index=False))
    fh.write(rdf.to_string(index=False) + '\n\n')
    rdf.to_csv(os.path.join(OUTPUT, 'retention_by_region.csv'), index=False)

    # (B) Lagged shocks -----------------------------------------------
    print('\n--- (B) Lagged shocks (t-1) ---')
    fh.write('(B) Lagged shocks (t-1)\n')
    lag_rows = []
    for sv in ['any_conflict_lag1', 'polstab_drop_lag1']:
        if sv not in intl.columns: continue
        r = run(intl, sv, 'return_same')
        if r:
            lag_rows.append({'shock': sv, **r})
    # contemporaneous + lag joint
    sub = intl.dropna(subset=['any_conflict', 'any_conflict_lag1',
                                'return_same']).copy()
    X = pd.concat([
        sub[['any_conflict', 'any_conflict_lag1']].astype(float),
        pd.get_dummies(sub['year'], prefix='y', drop_first=True).astype(float),
        pd.get_dummies(sub['sport'], prefix='sport',
                        drop_first=True).astype(float),
    ], axis=1)
    X = sm.add_constant(X)
    idx = X.dropna().index
    m = sm.OLS(sub.loc[idx, 'return_same'].astype(float),
                X.loc[idx]).fit(
            cov_type='cluster',
            cov_kwds={'groups': sub.loc[idx, 'country_code']})
    for v in ['any_conflict', 'any_conflict_lag1']:
        b = m.params[v]; se = m.bse[v]; p = m.pvalues[v]
        stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        lag_rows.append({'shock': v + ' (joint)', 'beta': round(b,4),
                          'se': round(se,4), 'p': round(p,4),
                          'stars': stars, 'N': len(idx)})
    ldf = pd.DataFrame(lag_rows)
    print(ldf.to_string(index=False))
    fh.write(ldf.to_string(index=False) + '\n\n')
    ldf.to_csv(os.path.join(OUTPUT, 'retention_lag.csv'), index=False)

    # (C) Intra-NCAA transfers ---------------------------------------
    print('\n--- (C) Transfer vs drop-out decomposition ---')
    fh.write('(C) Transfer vs drop-out decomposition\n')
    # mean rates
    print(f'  Intl mean return_same    : {intl["return_same"].mean():.3f}')
    print(f'  Intl mean return_anywhere: {intl["return_anywhere"].mean():.3f}')
    print(f'  Intl mean transferred    : {intl["transferred"].mean():.3f}')
    fh.write(f'  Intl mean return_same    : {intl["return_same"].mean():.3f}\n')
    fh.write(f'  Intl mean return_anywhere: {intl["return_anywhere"].mean():.3f}\n')
    fh.write(f'  Intl mean transferred    : {intl["transferred"].mean():.3f}\n\n')
    xfer_rows = []
    for sv in ['any_conflict', 'polstab_drop']:
        for out in ['return_same', 'return_anywhere', 'transferred']:
            r = run(intl, sv, out)
            if r:
                xfer_rows.append({'shock': sv, 'outcome': out, **r})
    xdf = pd.DataFrame(xfer_rows)
    print(xdf.to_string(index=False))
    fh.write(xdf.to_string(index=False) + '\n')
    xdf.to_csv(os.path.join(OUTPUT, 'retention_transfer.csv'), index=False)

    fh.close()
    print(f'\nWrote {os.path.join(OUTPUT, "retention_heterogeneity.txt")}')


if __name__ == '__main__':
    main()
