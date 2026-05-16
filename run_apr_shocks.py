"""
Link NCAA APR (team × sport × year) to home-country shock exposure
built from the multi-year rosters panel.

Outcome:  apr_rate_{team,sport,t} (4-year rolling D-I academic progress rate)
Treatment: ShockExposure_{team,sport,t} =
              share_intl_{team,sport,t} × avg-shock-share among intl members

The hypothesis from §3.8 (within-athlete retention) was that conflict
and political-instability at home reduce both the affected athlete's
retention and their teammates'. If that retention effect aggregates to
team-level academic outcomes, then teams more exposed to shocks should
post lower APR.

Caveat: APR is a 4-year rolling average; contemporaneous shock × team
exposure is an over-aggressive treatment timing (most of the APR window
predates the shock). We also report a 3-year rolling exposure to align
windows.
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


# Map rosters_panel (sport, gender) → APR sport_code
ROSTER_TO_APR = {
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
    # Track: roster file is "Mixed" gender (combined men+women). We replicate
    # it twice, once for MTR and once for WTR, since APR has both.
    ('Track', 'Mixed'):      ('MTR', 'WTR'),
}


def load_rosters():
    files = sorted(glob.glob(os.path.join(RAW_DATA, '*_rosters_panel.csv')))
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if 'sport' not in df.columns:
            df['sport'] = 'Track'
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
            'any_conflict', 'political_stability', 'log_disaster_events',
            'disaster_events']
    p = p[[c for c in cols if c in p.columns]].drop_duplicates(
        subset=['country_code', 'year'])
    if 'political_stability' in p.columns:
        m = p['political_stability'].mean(); s = p['political_stability'].std()
        p['polstab_drop'] = ((p['political_stability'] - m) / s < -1).astype(int)
    if 'log_disaster_events' in p.columns:
        thr = p['log_disaster_events'].quantile(0.9)
        p['severe_disaster'] = (p['log_disaster_events'] > thr).astype(int)
    return p


def team_exposure(roster, shocks, apr_code_map):
    """Collapse roster × shocks to (UNITID, sport_code, year) exposure."""
    r = roster.merge(shocks, on=['country_code', 'year'], how='left')
    # Resolve sport_code: pairs (sport, gender) → list of codes
    def map_codes(row):
        key = (row['sport'], row['gender'])
        v = apr_code_map.get(key)
        if v is None:
            return []
        if isinstance(v, tuple):
            return list(v)
        return [v]
    r['sport_codes'] = r.apply(map_codes, axis=1)
    r = r[r['sport_codes'].str.len() > 0].copy()
    r = r.explode('sport_codes').rename(columns={'sport_codes': 'sport_code'})

    # We don't have ipeds_unitid in roster panels. Use school_name → unitid
    # via crosswalk.
    cw = pd.read_csv(os.path.join(RAW_DATA,
                     'ncaa_ipeds_crosswalk_verified.csv'))
    cw = cw[['ncaa_name', 'ipeds_unitid']].copy()
    cw['key'] = cw['ncaa_name'].str.upper().str.strip()
    r['key'] = r['school_name'].str.upper().str.strip()
    r = r.merge(cw[['key', 'ipeds_unitid']], on='key', how='left')
    print(f'  Roster rows w/ UNITID: {r["ipeds_unitid"].notna().sum():,} '
          f'/ {len(r):,}')
    r = r[r['ipeds_unitid'].notna()].copy()
    r['ipeds_unitid'] = r['ipeds_unitid'].astype(int)

    # Per (team, sport_code, year): intl_share and average shock indicators
    grp = r.groupby(['ipeds_unitid', 'sport_code', 'year'])
    out = grp.agg(
        n_total=('is_intl', 'size'),
        n_intl=('is_intl', 'sum'),
        share_intl=('is_intl', 'mean'),
    ).reset_index()
    # Share of intl who are shocked × intl_share
    intl_only = r[r['is_intl'] == 1]
    intl_grp = intl_only.groupby(['ipeds_unitid', 'sport_code', 'year'])
    for sv in ['any_conflict', 'polstab_drop', 'gdp_shock',
               'currency_crisis', 'severe_disaster']:
        if sv not in intl_only.columns: continue
        agg = intl_grp[sv].mean().rename(f'intl_share_{sv}').reset_index()
        out = out.merge(agg, on=['ipeds_unitid', 'sport_code', 'year'],
                        how='left')
        out[f'expo_{sv}'] = (out['share_intl'].fillna(0) *
                              out[f'intl_share_{sv}'].fillna(0))
    return out


def main():
    warnings.filterwarnings('ignore')

    print('Loading rosters panel...')
    roster = load_rosters()
    print(f'  rows: {len(roster):,}')

    print('Loading shocks...')
    shocks = load_shocks()
    print(f'  shock rows: {len(shocks):,}')

    print('Building team-year exposure...')
    expo = team_exposure(roster, shocks, ROSTER_TO_APR)
    print(f'  team-year rows: {len(expo):,}')

    print('Loading APR...')
    apr = pd.read_csv(os.path.join(RAW_DATA, 'ncaa_apr.csv'),
                       low_memory=False)
    apr = apr[['ipeds_unitid', 'sport_code', 'year', 'apr_rate',
                'n_athletes', 'penalties', 'school_name', 'sport']].copy()
    apr['ipeds_unitid'] = pd.to_numeric(apr['ipeds_unitid'],
                                          errors='coerce').astype('Int64')
    apr = apr[apr['ipeds_unitid'].notna()].copy()
    apr['ipeds_unitid'] = apr['ipeds_unitid'].astype(int)

    df = apr.merge(expo, on=['ipeds_unitid', 'sport_code', 'year'],
                   how='inner')
    print(f'  merged rows: {len(df):,}')
    print(f'  unique team-sport: '
          f'{df.groupby(["ipeds_unitid","sport_code"]).ngroups:,}')

    # 3-year rolling exposure (within team-sport)
    df = df.sort_values(['ipeds_unitid', 'sport_code', 'year']).reset_index(
        drop=True)
    expo_cols = [c for c in df.columns if c.startswith('expo_')] + [
        'share_intl']
    for c in expo_cols:
        df[c + '_3y'] = (df.groupby(['ipeds_unitid', 'sport_code'])[c]
                          .transform(lambda x: x.rolling(3, min_periods=1)
                                                  .mean()))

    fh = open(os.path.join(OUTPUT, 'apr_shocks.txt'), 'w')
    fh.write('NCAA APR vs home-country shock exposure\n')
    fh.write('=' * 64 + '\n')
    fh.write(f'Team-sport-year obs: {len(df):,}\n')
    fh.write(f'Unique team-sport panels: '
             f'{df.groupby(["ipeds_unitid","sport_code"]).ngroups:,}\n\n')

    # --- Regression: APR ~ exposure, team-sport FE, year FE -------------
    def run(varname, with_team_fe=True):
        sub = df[df[varname].notna() & df['apr_rate'].notna()].copy()
        sub['unit_sport_id'] = (sub['ipeds_unitid'].astype(str) + '_' +
                                  sub['sport_code'])
        # within-team-sport demean
        if with_team_fe:
            for c in [varname, 'apr_rate']:
                sub[c + '_dm'] = (sub[c] - sub.groupby('unit_sport_id')[c]
                                                .transform('mean'))
            y = sub['apr_rate_dm']
            X = pd.concat([
                sub[[varname + '_dm']].rename(
                    columns={varname + '_dm': varname}),
                pd.get_dummies(sub['year'], prefix='y',
                                drop_first=True).astype(float),
            ], axis=1)
            spec = 'team-sport + year FE'
        else:
            y = sub['apr_rate']
            X = pd.concat([
                sub[[varname]].astype(float),
                pd.get_dummies(sub['year'], prefix='y',
                                drop_first=True).astype(float),
                pd.get_dummies(sub['sport_code'], prefix='sp',
                                drop_first=True).astype(float),
            ], axis=1)
            spec = 'year + sport FE'
        X = sm.add_constant(X)
        idx = X.dropna().index
        m = sm.OLS(y.loc[idx], X.loc[idx]).fit(
            cov_type='cluster',
            cov_kwds={'groups': sub.loc[idx, 'unit_sport_id']})
        b = m.params[varname]; se = m.bse[varname]; p = m.pvalues[varname]
        stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        return {'spec': spec, 'var': varname, 'beta': round(b, 5),
                'se': round(se, 5), 'p': round(p, 4),
                'stars': stars, 'N': len(idx)}

    rows = []
    for v in ['expo_any_conflict', 'expo_polstab_drop',
              'expo_gdp_shock', 'expo_currency_crisis',
              'expo_severe_disaster',
              'expo_any_conflict_3y', 'expo_polstab_drop_3y',
              'share_intl', 'share_intl_3y']:
        if v not in df.columns: continue
        rows.append(run(v, with_team_fe=False))
        rows.append(run(v, with_team_fe=True))
    res = pd.DataFrame(rows)
    print('\n--- APR on shock exposure (β scales APR 0-1 rate) ---')
    print(res.to_string(index=False))
    fh.write('--- APR on shock exposure (β scales APR 0-1 rate) ---\n')
    fh.write(res.to_string(index=False) + '\n')
    res.to_csv(os.path.join(OUTPUT, 'apr_shocks_table.csv'), index=False)

    # --- Sport split for the key indicator -------------------------------
    print('\n--- By sport (expo_any_conflict, team-sport + year FE) ---')
    fh.write('\n--- By sport (expo_any_conflict, team-sport + year FE) ---\n')
    sport_rows = []
    for sc in sorted(df['sport_code'].unique()):
        sub = df[(df['sport_code'] == sc) &
                  df['expo_any_conflict'].notna()].copy()
        if len(sub) < 200: continue
        sub['unit_id'] = sub['ipeds_unitid'].astype(str)
        for c in ['expo_any_conflict', 'apr_rate']:
            sub[c + '_dm'] = (sub[c] - sub.groupby('unit_id')[c]
                                          .transform('mean'))
        y = sub['apr_rate_dm']
        X = pd.concat([
            sub[['expo_any_conflict_dm']].rename(
                columns={'expo_any_conflict_dm': 'expo_any_conflict'}),
            pd.get_dummies(sub['year'], prefix='y',
                            drop_first=True).astype(float),
        ], axis=1)
        X = sm.add_constant(X)
        idx = X.dropna().index
        try:
            m = sm.OLS(y.loc[idx], X.loc[idx]).fit(
                cov_type='cluster',
                cov_kwds={'groups': sub.loc[idx, 'unit_id']})
            b = m.params['expo_any_conflict']
            se = m.bse['expo_any_conflict']
            p  = m.pvalues['expo_any_conflict']
            stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
            sport_rows.append({'sport_code': sc, 'beta': round(b, 5),
                                'se': round(se, 5), 'p': round(p, 4),
                                'stars': stars, 'N': len(idx)})
        except Exception:
            continue
    sdf = pd.DataFrame(sport_rows)
    print(sdf.to_string(index=False))
    fh.write(sdf.to_string(index=False) + '\n')
    sdf.to_csv(os.path.join(OUTPUT, 'apr_shocks_by_sport.csv'),
                index=False)

    fh.close()
    print(f'\nWrote {os.path.join(OUTPUT, "apr_shocks.txt")}')


if __name__ == '__main__':
    main()
