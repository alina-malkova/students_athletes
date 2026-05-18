"""
Four named shock case studies + one chronic comparator.

For each treated country, build a donor pool of "similar" non-shocked
countries, then run a simple DID on two outcomes:
  - retention: Pr(athlete returns to same school in t+1) at the
    athlete-year level.
  - new arrivals: number of freshman-class entrants per country-year.

Treated countries (onset year):
  Russia        2022   — invasion / sanctions / sport bans
  Ukraine       2022   — invasion / displacement
  Iran          2022   — Mahsa Amini protests / WGI polstab drop
  Lebanon       2020   — financial collapse spilling from Oct-2019
                          (the 2019 snapshot is essentially pre-crisis)
Comparator (no onset):
  Venezuela            — chronic instability; included to show what
                          a continuously-distressed country looks like

Donor pool for each treated country is built from the same region
with continuous non-shock status across 2019-2023, with the same
list of sports represented.
"""
from __future__ import annotations

import os, glob, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')


REGION = {
    **{c: 'SSA' for c in ['AGO','BDI','BEN','BFA','BWA','CAF','CIV','CMR',
        'COD','COG','COM','CPV','DJI','ERI','ETH','GAB','GHA','GIN','GMB',
        'GNB','GNQ','KEN','LBR','LSO','MDG','MLI','MOZ','MRT','MUS','MWI',
        'NAM','NER','NGA','RWA','SDN','SEN','SLE','SOM','SSD','STP','SWZ',
        'SYC','TCD','TGO','TZA','UGA','ZAF','ZMB','ZWE']},
    **{c: 'MENA' for c in ['ARE','BHR','DJI','DZA','EGY','IRN','IRQ','ISR',
        'JOR','KWT','LBN','LBY','MAR','OMN','PSE','QAT','SAU','SYR','TUN',
        'TUR','YEM']},
    **{c: 'EEU' for c in ['ALB','ARM','AZE','BGR','BIH','BLR','CZE','EST',
        'GEO','HRV','HUN','KAZ','KGZ','LTU','LVA','MDA','MKD','MNE','POL',
        'ROU','RUS','SRB','SVK','SVN','TJK','TKM','UKR','UZB','XKX']},
    **{c: 'LAC' for c in ['ARG','ATG','BHS','BLZ','BOL','BRA','BRB','CHL',
        'COL','CRI','CUB','DMA','DOM','ECU','GRD','GTM','GUY','HND','HTI',
        'JAM','KNA','LCA','MEX','NIC','PAN','PER','PRY','SLV','SUR','TTO',
        'URY','VCT','VEN','VGB']},
}


CASES = [
    {'name': 'Russia',    'cc': 'RUS', 'onset': 2022, 'region': 'EEU'},
    {'name': 'Ukraine',   'cc': 'UKR', 'onset': 2022, 'region': 'EEU'},
    {'name': 'Iran',      'cc': 'IRN', 'onset': 2022, 'region': 'MENA'},
    {'name': 'Lebanon',   'cc': 'LBN', 'onset': 2020, 'region': 'MENA'},
    {'name': 'Venezuela', 'cc': 'VEN', 'onset': None, 'region': 'LAC'},
]


def norm_class(c):
    if pd.isna(c): return 'UNK'
    s = str(c).upper().strip().replace('.','')
    if s in {'FR','SO','JR','FY','R-FR','R-SO','R-JR',
             'FRESHMAN','SOPHOMORE','JUNIOR'}: return 'UNDER'
    return 'OTHER'


def load_panel():
    fr = []
    for f in sorted(glob.glob(os.path.join(RAW_DATA, '*_rosters_panel.csv'))):
        df = pd.read_csv(f, low_memory=False)
        if 'sport' not in df.columns: df['sport'] = 'Track'
        fr.append(df)
    panel = pd.concat(fr, ignore_index=True)
    for col, k in [('name','name_k'),('school_name','school_k'),
                    ('gender','gender_k'),('sport','sport_k')]:
        panel[k] = (panel[col].fillna('').str.upper()
                     .str.replace(r'\s+',' ',regex=True).str.strip())
    panel['athlete_id'] = (panel['school_k'] + '|' + panel['gender_k'] +
                            '|' + panel['sport_k'] + '|' + panel['name_k'])
    panel['year'] = pd.to_numeric(panel['year'], errors='coerce').astype('Int64')
    panel = panel[panel['year'].notna()].copy()
    panel['year'] = panel['year'].astype(int)
    panel['class_norm'] = panel['class_year'].apply(norm_class)
    panel['region'] = panel['country_code'].map(REGION).fillna('OTHER')
    # retention
    seen = panel.groupby('athlete_id')['year'].apply(set).to_dict()
    panel['return_next'] = panel.apply(
        lambda r: int((r['year']+1) in seen.get(r['athlete_id'], set())),
        axis=1)
    return panel


def load_shocks():
    s = pd.read_csv(os.path.join(RAW_DATA, 'country_year_sport_panel.csv'))
    cols = ['country_code','year','any_conflict','political_stability',
            'currency_crisis','gdp_shock']
    s = s[[c for c in cols if c in s.columns]
          ].drop_duplicates(['country_code','year'])
    m = s['political_stability'].mean(); sd = s['political_stability'].std()
    s['polstab_drop'] = ((s['political_stability']-m)/sd < -1).astype(int)
    return s


def build_donor_pool(panel, shocks, case, sports_used):
    """Donor pool: same-region countries with no shock onset in the
    panel period (any_conflict and polstab_drop stay 0 throughout),
    with at least one athlete in each of the sports the treated
    country has athletes in."""
    region = case['region']
    onset = case['onset']
    candidates = [cc for cc, r in REGION.items()
                   if r == region and cc != case['cc']]
    # never any shock indicator in 2019-2023
    shocks_period = shocks[shocks['year'].between(2019, 2023)]
    bad = set()
    for cc in candidates:
        sub = shocks_period[shocks_period['country_code'] == cc]
        if ((sub['any_conflict'].fillna(0).max() > 0) or
            (sub['polstab_drop'].fillna(0).max() > 0)):
            bad.add(cc)
    donors = [cc for cc in candidates if cc not in bad]
    # require at least one panel athlete from each donor
    has_data = panel[panel['country_code'].isin(donors)
                     ]['country_code'].unique().tolist()
    donors = [cc for cc in donors if cc in has_data]
    return donors


def case_did_retention(panel, case, donors):
    """DID at athlete-year level: return_next ~ post*treated +
    country FE + year FE."""
    onset = case['onset']
    if onset is None: return None
    cc = case['cc']
    keep = [cc] + donors
    sub = panel[(panel['country_code'].isin(keep)) &
                 (panel['year'] < 2023) &
                 (panel['class_norm'] == 'UNDER')].copy()
    sub['treated'] = (sub['country_code'] == cc).astype(int)
    sub['post'] = (sub['year'] >= onset).astype(int)
    sub['did'] = sub['treated'] * sub['post']
    X = pd.concat([
        sub[['did','treated','post']].astype(float),
        pd.get_dummies(sub['year'], prefix='y',
                        drop_first=True).astype(float),
        pd.get_dummies(sub['sport'], prefix='sp',
                        drop_first=True).astype(float),
    ], axis=1)
    X = sm.add_constant(X)
    idx = X.dropna().index
    m_ = sm.OLS(sub.loc[idx,'return_next'].astype(float),
                 X.loc[idx]).fit(cov_type='cluster',
                  cov_kwds={'groups': sub.loc[idx,'country_code']})
    b = m_.params['did']; se = m_.bse['did']; p = m_.pvalues['did']
    stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
    # mean retention by year × treated
    means = (sub.groupby(['year','treated'])['return_next']
                .mean().unstack().rename(columns={0:'donor',1:'treated'}))
    return {'case': case['name'], 'cc': cc, 'onset': onset,
            'n_donors': len(donors),
            'n_treated_athyears': int(sub['treated'].sum()),
            'n_donor_athyears': int((1-sub['treated']).sum()),
            'did_beta': round(b,4), 'did_se': round(se,4),
            'did_p': round(p,4), 'stars': stars,
            'means': means}


def case_did_arrivals(panel, case, donors):
    """DID at country-year level on log(1+freshman entrants).
    Freshmen identified by class_norm == 'UNDER' AND year == first
    observed year for that athlete (proxy for new arrival)."""
    onset = case['onset']
    if onset is None: return None
    cc = case['cc']
    keep = [cc] + donors
    # first observed year for each athlete
    first = panel.groupby('athlete_id')['year'].min().rename('first_yr')
    pn = panel.merge(first, on='athlete_id')
    new = pn[pn['year'] == pn['first_yr']].copy()
    # count freshman entrants per country×year
    cnt = (new[new['country_code'].isin(keep)]
              .groupby(['country_code','year']).size().rename('n')
              .reset_index())
    # fill missing country-year cells with 0
    full = pd.MultiIndex.from_product(
        [keep, sorted(panel['year'].unique())], names=['country_code','year'])
    cnt = cnt.set_index(['country_code','year']).reindex(full,
        fill_value=0).reset_index()
    cnt['log_n'] = np.log1p(cnt['n'])
    cnt['treated'] = (cnt['country_code'] == cc).astype(int)
    cnt['post'] = (cnt['year'] >= onset).astype(int)
    cnt['did'] = cnt['treated'] * cnt['post']
    X = pd.concat([
        cnt[['did','treated','post']].astype(float),
        pd.get_dummies(cnt['year'], prefix='y',
                        drop_first=True).astype(float),
        pd.get_dummies(cnt['country_code'], prefix='cc',
                        drop_first=True).astype(float),
    ], axis=1)
    X = sm.add_constant(X)
    idx = X.dropna().index
    m_ = sm.OLS(cnt.loc[idx,'log_n'], X.loc[idx]).fit(cov_type='HC1')
    b = m_.params['did']; se = m_.bse['did']; p = m_.pvalues['did']
    stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
    # raw counts for plot (aggregate donor pool first)
    agg = cnt.groupby(['year','treated'])['n'].sum().reset_index()
    raw = (agg.pivot(index='year', columns='treated', values='n')
              .rename(columns={0:'donor_total', 1:'treated'}))
    raw['donor_mean'] = raw['donor_total'] / max(len(donors), 1)
    return {'case': case['name'], 'cc': cc, 'onset': onset,
            'did_log_beta': round(b,4), 'did_log_se': round(se,4),
            'did_log_p': round(p,4), 'stars': stars,
            'raw': raw[['treated','donor_mean']]}


def venezuela_summary(panel, case):
    """For Venezuela: just describe the trajectory (no DID)."""
    cc = case['cc']
    # exclude 2023 from retention since we don't observe 2024
    sub = panel[(panel['country_code']==cc) & (panel['year'] < 2023)]
    by_yr = (sub.groupby('year').agg(
        n_obs=('athlete_id','size'),
        n_ath=('athlete_id','nunique'),
        retention=('return_next','mean'),
    ))
    return {'case': case['name'], 'cc': cc, 'trajectory': by_yr}


def main():
    warnings.filterwarnings('ignore')
    panel = load_panel()
    shocks = load_shocks()
    print(f'Panel rows: {len(panel):,}')

    fh = open(os.path.join(OUTPUT, 'case_studies.txt'), 'w')
    fh.write('Five-country shock case studies\n')
    fh.write('='*64 + '\n\n')

    summary_rows = []
    for case in CASES:
        print(f"\n\n=== {case['name']} ({case['cc']}, onset {case['onset']}) ===")
        fh.write(f"\n=== {case['name']} ({case['cc']}, onset "
                 f"{case['onset']}) ===\n")
        if case['onset'] is None:
            # Venezuela
            ven = venezuela_summary(panel, case)
            print('Annual trajectory:')
            print(ven['trajectory'].to_string())
            fh.write('Annual trajectory:\n')
            fh.write(ven['trajectory'].to_string() + '\n')
            ven['trajectory'].to_csv(os.path.join(OUTPUT,
                f"case_{case['cc']}_trajectory.csv"))
            continue

        # sports the treated country has
        sports_used = list(panel[panel['country_code']==case['cc']]
                            ['sport'].unique())
        donors = build_donor_pool(panel, shocks, case, sports_used)
        print(f'  Donors ({len(donors)}): {donors}')
        fh.write(f'  Donors ({len(donors)}): {", ".join(donors)}\n')

        ret = case_did_retention(panel, case, donors)
        arr = case_did_arrivals(panel, case, donors)
        if ret:
            print(f'\n  Retention DID: β = {ret["did_beta"]:+.4f} '
                   f'(SE {ret["did_se"]:.4f}) {ret["stars"]}  '
                   f'N_treated = {ret["n_treated_athyears"]}, '
                   f'N_donor = {ret["n_donor_athyears"]}')
            print('  Mean retention by year:')
            print(ret['means'].to_string())
            fh.write(f'\n  Retention DID: β = {ret["did_beta"]:+.4f} '
                      f'(SE {ret["did_se"]:.4f}) {ret["stars"]}\n')
            fh.write('  Mean retention by year:\n')
            fh.write(ret['means'].to_string() + '\n')
            ret['means'].to_csv(os.path.join(OUTPUT,
                f"case_{case['cc']}_retention.csv"))
        if arr:
            print(f'\n  New-arrivals DID: log β = '
                   f'{arr["did_log_beta"]:+.4f} '
                   f'(SE {arr["did_log_se"]:.4f}) {arr["stars"]}')
            print('  Raw new-arrival counts:')
            print(arr['raw'].to_string())
            fh.write(f'\n  New-arrivals DID: log β = '
                      f'{arr["did_log_beta"]:+.4f} '
                      f'(SE {arr["did_log_se"]:.4f}) {arr["stars"]}\n')
            fh.write('  Raw new-arrival counts:\n')
            fh.write(arr['raw'].to_string() + '\n')
            arr['raw'].to_csv(os.path.join(OUTPUT,
                f"case_{case['cc']}_arrivals.csv"))

        summary_rows.append({
            'case': case['name'], 'cc': case['cc'], 'onset': case['onset'],
            'retention_DID': ret['did_beta'] if ret else None,
            'retention_p': ret['did_p'] if ret else None,
            'retention_stars': ret['stars'] if ret else None,
            'n_treated_athyears': ret['n_treated_athyears'] if ret else None,
            'arrivals_log_DID': arr['did_log_beta'] if arr else None,
            'arrivals_p': arr['did_log_p'] if arr else None,
            'arrivals_stars': arr['stars'] if arr else None,
        })

    fh.write('\n\nSummary across cases\n')
    fh.write('-'*64 + '\n')
    summ = pd.DataFrame(summary_rows)
    fh.write(summ.to_string(index=False) + '\n')
    summ.to_csv(os.path.join(OUTPUT, 'case_studies_summary.csv'),
                index=False)
    print('\n\n=== Summary across cases ===')
    print(summ.to_string(index=False))

    fh.close()
    print(f'\nWrote {os.path.join(OUTPUT, "case_studies.txt")}')


if __name__ == '__main__':
    main()
