"""
Athlete-year retention panel and home-country shock analysis.

Question: do political / economic shocks in athletes' home countries change
whether they remain enrolled at their U.S. NCAA school the following year?
And do those shocks affect their (domestic) teammates' retention?

Data: 8 sport rosters_panel files (2019-2023).
Outcome: 1{athlete is on the same school's roster in year t+1} for each
  athlete-year, restricted to athletes with class_year in {Fr, So, Jr,
  R-Fr, R-So, R-Jr}.  Seniors / 5th-year are dropped because graduation
  is the normal exit and would dominate the variation.

Two regressions:
  (1) Own-shock retention — international athletes (country != USA),
      within-athlete FE + year FE + sport FE.
  (2) Peer-spillover retention — domestic athletes (country == USA),
      treatment = share of (own school × sport × year) team from a
      shocked country (gdp / currency / political / conflict / disaster).
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


CLASS_RETURN_OK = {'FR', 'SO', 'JR', 'FY', 'R-FR', 'R-SO', 'R-JR',
                   'FRESHMAN', 'SOPHOMORE', 'JUNIOR'}
SR_LIKE        = {'SR', 'R-SR', '5TH', 'GR', 'SENIOR'}


def norm_class(c):
    if pd.isna(c): return 'UNK'
    s = str(c).upper().strip().replace('.', '')
    if s in CLASS_RETURN_OK: return 'UNDER'
    if s in SR_LIKE: return 'SENIOR'
    return 'UNK'


def load_athlete_panel():
    files = sorted(glob.glob(os.path.join(RAW_DATA, '*_rosters_panel.csv')))
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if 'sport' not in df.columns:
            # track panel has no sport column; tag it
            df['sport'] = 'Track'
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    # canonical athlete id = (school, gender, sport, name)
    # NB: names are imperfect IDs; collapse case + extra whitespace
    panel['name_k']    = (panel['name'].fillna('').str.upper()
                           .str.replace(r'\s+', ' ', regex=True).str.strip())
    panel['school_k']  = panel['school_name'].fillna('').str.upper().str.strip()
    panel['gender_k']  = panel['gender'].fillna('').str.upper().str.strip()
    panel['sport_k']   = panel['sport'].fillna('').str.upper().str.strip()
    panel['athlete_id'] = (panel['school_k'] + '|' + panel['gender_k'] + '|' +
                            panel['sport_k'] + '|' + panel['name_k'])
    panel['class_norm'] = panel['class_year'].apply(norm_class)
    panel['year'] = pd.to_numeric(panel['year'], errors='coerce').astype('Int64')
    panel = panel[panel['year'].notna()].copy()
    panel['year'] = panel['year'].astype(int)
    return panel


def build_retention(panel):
    """Add return_next_yr = 1{athlete_id appears on same school in year+1}."""
    seen = panel.groupby(['athlete_id'])['year'].apply(set).to_dict()
    panel['return_next'] = panel.apply(
        lambda r: int((r['year'] + 1) in seen.get(r['athlete_id'], set())),
        axis=1)
    return panel


def load_shocks():
    """Country×year shock indicators (collapse sport dim out)."""
    p = pd.read_csv(os.path.join(RAW_DATA, 'country_year_sport_panel.csv'))
    cols = ['country_code', 'year', 'gdp_shock', 'currency_crisis',
            'unemployment_spike', 'any_econ_shock', 'political_stability',
            'log_disaster_events', 'any_conflict', 'n_conflicts',
            'disaster_events']
    p = p[[c for c in cols if c in p.columns]].drop_duplicates(
        subset=['country_code', 'year'])
    # also build "political_instability" = negated z-score of WGI stability
    if 'political_stability' in p.columns:
        m = p['political_stability'].mean(); s = p['political_stability'].std()
        p['polstab_z']     = (p['political_stability'] - m) / s
        p['polstab_drop']  = (p['polstab_z'] < -1).astype(int)
    # Severe-disaster: top decile of log(events) across country-years
    if 'log_disaster_events' in p.columns:
        thr = p['log_disaster_events'].quantile(0.9)
        p['severe_disaster'] = (p['log_disaster_events'] > thr).astype(int)
    else:
        p['severe_disaster'] = 0
    # Tightened composite — drop the disaster-saturated component, keep
    # security/macro shocks only.
    p['any_shock'] = ((p.get('gdp_shock', 0) == 1) |
                      (p.get('currency_crisis', 0) == 1) |
                      (p.get('any_conflict', 0) == 1) |
                      (p.get('polstab_drop', 0) == 1) |
                      (p['severe_disaster'] == 1)).astype(int)
    return p


def main():
    warnings.filterwarnings('ignore')
    print('Loading rosters panel ...')
    panel = load_athlete_panel()
    print(f'  rows: {len(panel):,}')

    print('Computing return_next_yr ...')
    panel = build_retention(panel)
    # restrict to underclassmen years where return is non-trivial
    work = panel[panel['class_norm'] == 'UNDER'].copy()
    # drop final year 2023 (we can't observe 2024)
    work = work[work['year'] < 2023].copy()
    work['is_intl'] = ((work['country_code'].notna()) &
                       (work['country_code'] != 'USA')).astype(int)
    print(f'  observation panel (under-classmen, year < 2023): {len(work):,}')
    print(f'  international: {work["is_intl"].sum():,}  '
          f'domestic: {(1 - work["is_intl"]).sum():,}')
    print(f'  mean retention (all): {work["return_next"].mean():.3f}')
    print(f'  mean retention (intl): {work.loc[work.is_intl==1,"return_next"].mean():.3f}')
    print(f'  mean retention (dom):  {work.loc[work.is_intl==0,"return_next"].mean():.3f}')

    shocks = load_shocks()
    work = work.merge(shocks, on=['country_code', 'year'], how='left')

    fh = open(os.path.join(OUTPUT, 'retention_shocks.txt'), 'w')

    fh.write('Retention vs home-country shocks\n')
    fh.write('=' * 64 + '\n')
    fh.write(f'Athlete-year panel 2019-2022 (non-senior).\n')
    fh.write(f'N = {len(work):,}  athletes = {work["athlete_id"].nunique():,}\n')
    fh.write(f'8 sports stacked.\n\n')

    # ---------- (1) Own-shock effect on international athletes ------------
    intl = work[work['is_intl'] == 1].copy()
    print(f'\nInternational athlete-years with shock data: {len(intl):,}')
    intl = intl[intl['any_shock'].notna()]
    fh.write(f'(1) Own-shock retention — international athletes\n')
    fh.write(f'    N = {len(intl):,}\n\n')

    shock_vars = ['gdp_shock', 'currency_crisis', 'any_conflict',
                  'polstab_drop', 'severe_disaster', 'any_shock']

    def run_own(sub, sv, with_athlete_fe=False):
        sub = sub.copy()
        if with_athlete_fe:
            # within-athlete demean for shock + outcome; require ≥2 years
            counts = sub['athlete_id'].value_counts()
            keep = counts[counts >= 2].index
            sub = sub[sub['athlete_id'].isin(keep)].copy()
            for col in ['return_next', sv]:
                sub[col + '_dm'] = (sub[col].astype(float) -
                                    sub.groupby('athlete_id')[col]
                                       .transform('mean'))
            y = sub['return_next_dm']
            X = pd.concat([
                sub[[sv + '_dm']].rename(columns={sv + '_dm': sv}),
                pd.get_dummies(sub['year'], prefix='y',
                                drop_first=True).astype(float),
            ], axis=1)
        else:
            y = sub['return_next'].astype(float)
            X = pd.concat([
                sub[[sv]].astype(float),
                pd.get_dummies(sub['year'], prefix='y', drop_first=True
                                ).astype(float),
                pd.get_dummies(sub['sport'], prefix='sport',
                                drop_first=True).astype(float),
            ], axis=1)
        X = sm.add_constant(X)
        idx = X.dropna().index
        m = sm.OLS(y.loc[idx], X.loc[idx]).fit(
            cov_type='cluster',
            cov_kwds={'groups': sub.loc[idx, 'country_code']})
        b = m.params[sv]; se = m.bse[sv]; p = m.pvalues[sv]
        stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        return {'beta': round(b, 4), 'se': round(se, 4), 'p': round(p, 4),
                'stars': stars, 'N': len(idx)}

    # Baseline (year+sport FE)
    base_rows = []
    for sv in shock_vars:
        if sv not in intl.columns: continue
        sub = intl[intl[sv].notna()].copy()
        if len(sub) < 100: continue
        r = run_own(sub, sv, with_athlete_fe=False)
        r['shock'] = sv; r['spec'] = 'year+sport FE'
        base_rows.append(r)
    # Athlete FE (within-athlete variation)
    for sv in shock_vars:
        if sv not in intl.columns: continue
        sub = intl[intl[sv].notna()].copy()
        if len(sub) < 100: continue
        r = run_own(sub, sv, with_athlete_fe=True)
        r['shock'] = sv; r['spec'] = 'athlete+year FE'
        base_rows.append(r)
    own_df = pd.DataFrame(base_rows)[
        ['spec', 'shock', 'beta', 'se', 'p', 'stars', 'N']]
    print('\n--- Own-shock effect on intl athletes (Pr(return)) ---')
    print(own_df.to_string(index=False))
    fh.write('--- Own-shock effect on intl athletes (Pr(return)) ---\n')
    fh.write(own_df.to_string(index=False) + '\n\n')
    own_df.to_csv(os.path.join(OUTPUT, 'retention_own_shock.csv'), index=False)

    # Sport splits with the two clean indicators
    print('\n--- Own-shock effect by sport (conflict & polstab_drop) ---')
    fh.write('\n--- Own-shock effect by sport ---\n')
    sport_rows = []
    for sport in sorted(intl['sport'].unique()):
        for sv in ['any_conflict', 'polstab_drop', 'any_shock']:
            sub = intl[(intl['sport'] == sport) & intl[sv].notna()].copy()
            if len(sub) < 200: continue
            try:
                X = pd.concat([
                    sub[[sv]].astype(float),
                    pd.get_dummies(sub['year'], prefix='y',
                                    drop_first=True).astype(float),
                ], axis=1)
                X = sm.add_constant(X)
                idx = X.dropna().index
                m = sm.OLS(sub.loc[idx, 'return_next'].astype(float),
                           X.loc[idx]).fit(cov_type='cluster',
                                            cov_kwds={'groups':
                                              sub.loc[idx, 'country_code']})
                b = m.params[sv]; se = m.bse[sv]; p = m.pvalues[sv]
                stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
                sport_rows.append({'sport': sport, 'shock': sv,
                                    'beta': round(b,4), 'se': round(se,4),
                                    'p': round(p,4), 'stars': stars,
                                    'N': len(idx)})
            except Exception:
                continue
    sport_df = pd.DataFrame(sport_rows)
    print(sport_df.to_string(index=False))
    fh.write(sport_df.to_string(index=False) + '\n')
    sport_df.to_csv(os.path.join(OUTPUT, 'retention_own_shock_by_sport.csv'),
                     index=False)

    # ---------- (2) Peer spillover — domestic athletes -------------------
    # Build team-year exposure: among international teammates in same
    # (school, sport, gender, year), share whose home country had any_shock=1.
    team_grp = work.groupby(['school_k', 'gender_k', 'sport_k', 'year'])
    team_intl_n   = team_grp['is_intl'].sum().rename('team_n_intl')
    team_size     = team_grp.size().rename('team_n_total')
    intl_team = work[work['is_intl'] == 1].copy()
    # share shocked among intl teammates
    for v in ['gdp_shock', 'currency_crisis', 'any_conflict', 'polstab_drop',
              'any_shock']:
        if v not in intl_team.columns: continue
        intl_team[v + '_z'] = intl_team[v].fillna(0)
    agg = (intl_team.groupby(['school_k', 'gender_k', 'sport_k', 'year'])
                    [[c for c in intl_team.columns if c.endswith('_z')]]
                    .mean())
    team_chars = pd.concat([team_intl_n, team_size, agg], axis=1).reset_index()
    team_chars['intl_share'] = team_chars['team_n_intl'] / team_chars['team_n_total']

    dom = work[work['is_intl'] == 0].copy()
    dom = dom.merge(team_chars,
                    on=['school_k', 'gender_k', 'sport_k', 'year'], how='left')
    # treatment intensity for domestic athlete = intl_share * shock_share_z
    dom['expo_any']      = dom['intl_share'].fillna(0) * dom['any_shock_z'].fillna(0)
    dom['expo_currency'] = dom['intl_share'].fillna(0) * dom['currency_crisis_z'].fillna(0)
    dom['expo_conflict'] = dom['intl_share'].fillna(0) * dom['any_conflict_z'].fillna(0)
    dom['expo_polstab']  = dom['intl_share'].fillna(0) * dom['polstab_drop_z'].fillna(0)
    dom['expo_gdp']      = dom['intl_share'].fillna(0) * dom['gdp_shock_z'].fillna(0)

    fh.write(f'(2) Peer-spillover retention — domestic athletes\n')
    fh.write(f'    N = {len(dom):,}\n')
    fh.write(f'    Treatment: intl_share × team-average shock indicator\n\n')

    peer_rows = []
    for tv in ['expo_any', 'expo_currency', 'expo_conflict',
               'expo_polstab', 'expo_gdp']:
        sub = dom[dom[tv].notna()].copy()
        if len(sub) < 1000: continue
        X = pd.concat([
            sub[[tv, 'intl_share']].astype(float),
            pd.get_dummies(sub['year'], prefix='y', drop_first=True).astype(float),
            pd.get_dummies(sub['sport_k'], prefix='sport',
                            drop_first=True).astype(float),
        ], axis=1)
        X = sm.add_constant(X)
        idx = X.dropna().index
        m = sm.OLS(sub.loc[idx, 'return_next'].astype(float),
                   X.loc[idx]).fit(cov_type='cluster',
                                    cov_kwds={'groups': sub.loc[idx,'school_k']})
        b = m.params[tv]; se = m.bse[tv]; p = m.pvalues[tv]
        stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        peer_rows.append({'exposure': tv, 'beta': round(b, 4),
                          'se': round(se, 4), 'p': round(p, 4),
                          'stars': stars, 'N': len(idx)})
    peer_df = pd.DataFrame(peer_rows)
    print('\n--- Peer-spillover effect on domestic athletes (Pr(return)) ---')
    print(peer_df.to_string(index=False))
    fh.write('--- Peer-spillover effect on domestic athletes (Pr(return)) ---\n')
    fh.write(peer_df.to_string(index=False) + '\n')
    peer_df.to_csv(os.path.join(OUTPUT, 'retention_peer_spillover.csv'),
                    index=False)

    # ---------- Sanity: mean retention by shock category ------------------
    mn = intl.groupby('any_shock')['return_next'].agg(['mean', 'size'])
    print('\n--- Mean retention for intl, by any_shock ---')
    print(mn.to_string())
    fh.write('\n--- Mean retention for intl, by any_shock ---\n')
    fh.write(mn.to_string() + '\n')

    fh.close()
    print(f'\nWrote {os.path.join(OUTPUT, "retention_shocks.txt")}')


if __name__ == '__main__':
    main()
