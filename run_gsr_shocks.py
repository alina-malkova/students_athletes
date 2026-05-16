"""
Test whether team graduation rates (GSR) covary with home-country
shock exposure built from the roster panel.

GSR is a 6-year cohort graduation rate.  Cohort year is the freshman-
entering year, so a 2018 cohort graduates by 2024.  Our roster panel
covers 2019–2023.  We therefore use the most recent cohorts
(2014–2018) and compute team-sport-mean exposure across the 5 roster
years.

Spec:  GSR_{team-sport, cohort} = α_{sport} + γ_{cohort}
                                  + β · MeanExposure_{team-sport}
                                  + ε

Since exposure is team-sport-mean (time-invariant in this design),
team-sport FE would absorb it; we use sport FE + cohort FE instead.
SEs clustered at the school.
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


ROSTER_TO_GSR = {
    ('soccer', 'Men'):       'MSO',
    ('soccer', 'Women'):     'WSO',
    ('basketball', 'Men'):   'MBB',
    ('basketball', 'Women'): 'WBB',
    ('volleyball', 'Men'):   'MVB',
    ('volleyball', 'Women'): 'WVB',
    ('icehockey', 'Men'):    'MIH',
    ('icehockey', 'Women'):  'WIH',
    ('swimming', 'Men'):     'MSW',
    ('swimming', 'Women'):   'WSW',
    ('golf', 'Men'):         'MGO',
    ('golf', 'Women'):       'WGO',
    ('tennis', 'Men'):       'MTE',
    ('tennis', 'Women'):     'WTE',
    ('Track', 'Mixed'):      ('MTR', 'WTR'),
}


def load_rosters():
    files = sorted(glob.glob(os.path.join(RAW_DATA, '*_rosters_panel.csv')))
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if 'sport' not in df.columns: df['sport'] = 'Track'
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel['year'] = pd.to_numeric(panel['year'], errors='coerce').astype('Int64')
    panel = panel[panel['year'].notna()].copy()
    panel['year'] = panel['year'].astype(int)
    panel['is_intl'] = ((panel['country_code'].notna()) &
                       (panel['country_code'] != 'USA')).astype(int)
    return panel


def load_shocks():
    p = pd.read_csv(os.path.join(RAW_DATA, 'country_year_sport_panel.csv'))
    cols = ['country_code', 'year', 'gdp_shock', 'currency_crisis',
            'any_conflict', 'political_stability']
    p = p[[c for c in cols if c in p.columns]].drop_duplicates(
        subset=['country_code', 'year'])
    m = p['political_stability'].mean(); s = p['political_stability'].std()
    p['polstab_drop'] = ((p['political_stability'] - m) / s < -1).astype(int)
    return p


def main():
    warnings.filterwarnings('ignore')
    roster = load_rosters()
    shocks = load_shocks()
    cw = pd.read_csv(os.path.join(RAW_DATA,
                     'ncaa_ipeds_crosswalk_verified.csv'))
    cw['key'] = cw['ncaa_name'].str.upper().str.strip()

    # merge shocks
    r = roster.merge(shocks, on=['country_code', 'year'], how='left')
    # expand sport_codes
    def map_codes(row):
        v = ROSTER_TO_GSR.get((row['sport'], row['gender']))
        return list(v) if isinstance(v, tuple) else ([v] if v else [])
    r['sport_codes'] = r.apply(map_codes, axis=1)
    r = r[r['sport_codes'].str.len() > 0].explode('sport_codes').rename(
        columns={'sport_codes': 'sport_code'})
    # unitid
    r['key'] = r['school_name'].str.upper().str.strip()
    r = r.merge(cw[['key', 'ipeds_unitid']], on='key', how='left')
    r = r[r['ipeds_unitid'].notna()].copy()
    r['ipeds_unitid'] = r['ipeds_unitid'].astype(int)

    # team-sport-mean exposure across 2019-2023
    grp = r.groupby(['ipeds_unitid', 'sport_code'])
    mean_share = grp['is_intl'].mean().rename('mean_share_intl')
    intl_only = r[r['is_intl'] == 1]
    intl_grp = intl_only.groupby(['ipeds_unitid', 'sport_code'])
    pieces = [mean_share]
    for sv in ['any_conflict', 'polstab_drop', 'gdp_shock',
               'currency_crisis']:
        if sv not in intl_only.columns: continue
        pieces.append(intl_grp[sv].mean().rename(f'intl_mean_{sv}'))
    expo = pd.concat(pieces, axis=1).reset_index()
    for sv in ['any_conflict', 'polstab_drop', 'gdp_shock',
               'currency_crisis']:
        col = f'intl_mean_{sv}'
        if col in expo.columns:
            expo[f'expo_{sv}'] = expo['mean_share_intl'].fillna(0) * \
                                  expo[col].fillna(0)

    gsr = pd.read_csv(os.path.join(RAW_DATA, 'ncaa_gsr.csv'),
                       low_memory=False)
    gsr = gsr[gsr['sport_code'] != 'ALL'].copy()
    gsr['ipeds_unitid'] = pd.to_numeric(gsr['ipeds_unitid'],
                                          errors='coerce').astype('Int64')
    gsr = gsr[gsr['ipeds_unitid'].notna()].copy()
    gsr['ipeds_unitid'] = gsr['ipeds_unitid'].astype(int)
    gsr = gsr[gsr['cohort_year'] >= 2014].copy()

    df = gsr.merge(expo, on=['ipeds_unitid', 'sport_code'], how='inner')
    df = df[df['gsr'].notna()].copy()
    print(f'Merged team-sport-cohort obs: {len(df):,}')
    print(f'Cohort years: {sorted(df["cohort_year"].unique())}')

    fh = open(os.path.join(OUTPUT, 'gsr_shocks.txt'), 'w')
    fh.write('NCAA GSR vs home-country shock exposure\n')
    fh.write('=' * 64 + '\n')
    fh.write(f'N = {len(df):,}  team-sport-cohort\n')
    fh.write('Exposure is team-sport-mean across roster years 2019-2023.\n\n')

    rows = []
    for v in ['mean_share_intl', 'expo_any_conflict',
              'expo_polstab_drop', 'expo_gdp_shock',
              'expo_currency_crisis']:
        if v not in df.columns: continue
        sub = df[df[v].notna()].copy()
        X = pd.concat([
            sub[[v]].astype(float),
            pd.get_dummies(sub['cohort_year'], prefix='c',
                            drop_first=True).astype(float),
            pd.get_dummies(sub['sport_code'], prefix='sp',
                            drop_first=True).astype(float),
        ], axis=1)
        X = sm.add_constant(X)
        idx = X.dropna().index
        m = sm.OLS(sub.loc[idx, 'gsr'].astype(float),
                   X.loc[idx]).fit(
                cov_type='cluster',
                cov_kwds={'groups': sub.loc[idx, 'ipeds_unitid']})
        b = m.params[v]; se = m.bse[v]; p = m.pvalues[v]
        stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        rows.append({'var': v, 'beta': round(b, 3),
                      'se': round(se, 3), 'p': round(p, 4),
                      'stars': stars, 'N': len(idx)})
        # FGR variant
        if 'fgr' in sub.columns and sub['fgr'].notna().sum() > 100:
            sub_f = sub[sub['fgr'].notna()]
            X_f = X.loc[X.index.isin(sub_f.index)]
            idx_f = X_f.dropna().index
            m_f = sm.OLS(sub_f.loc[idx_f, 'fgr'].astype(float),
                          X_f.loc[idx_f]).fit(
                  cov_type='cluster',
                  cov_kwds={'groups': sub_f.loc[idx_f, 'ipeds_unitid']})
            b = m_f.params[v]; se = m_f.bse[v]; p = m_f.pvalues[v]
            stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
            rows.append({'var': v + ' (FGR)', 'beta': round(b, 3),
                          'se': round(se, 3), 'p': round(p, 4),
                          'stars': stars, 'N': len(idx_f)})
    res = pd.DataFrame(rows)
    print('\n--- GSR / FGR on team-sport-mean exposure ---')
    print(res.to_string(index=False))
    fh.write('--- GSR / FGR on team-sport-mean exposure ---\n')
    fh.write(res.to_string(index=False) + '\n')
    res.to_csv(os.path.join(OUTPUT, 'gsr_shocks_table.csv'), index=False)

    # Variance check
    fh.write(f'\nGSR pooled mean = {df["gsr"].mean():.1f}  '
              f'SD = {df["gsr"].std():.1f}\n')
    fh.write(f'Within-team-sport GSR SD = '
              f'{df.groupby(["ipeds_unitid","sport_code"])["gsr"].std().mean():.1f}\n')
    print(f'\nGSR pooled mean = {df["gsr"].mean():.1f}, SD = {df["gsr"].std():.1f}')

    fh.close()
    print(f'\nWrote {os.path.join(OUTPUT, "gsr_shocks.txt")}')


if __name__ == '__main__':
    main()
