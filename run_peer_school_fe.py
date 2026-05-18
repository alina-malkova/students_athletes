"""
Peer-spillover with school FE + TFRRS performance event study.

(1) The peer-spillover spec in run_retention_shocks.py regresses
domestic athletes' Pr(return) on (intl_share × avg_shock_among_intl)
with year + sport FE only. Reviewer concern: if intl_share is
endogenous to school type (R1, well-funded) and well-funded schools
recruit from particular regions where shocks cluster (LAC, MENA),
the coefficient could pick up school-type × shock-region confounds.
Add school FE via within-school demean.

(2) The TFRRS performance regression has athlete×event FE, which
mechanically absorbs country FE (country is constant within
athlete). It already identifies off within-country variation. Run
an event study on TFRRS performance to verify the timing matches
the shock onset (no large pre-trend, response at k=0 or k=+1).
"""
from __future__ import annotations

import os, glob, warnings, re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')


def norm_class(c):
    if pd.isna(c): return 'UNK'
    s = str(c).upper().strip().replace('.', '')
    if s in {'FR','SO','JR','FY','R-FR','R-SO','R-JR',
             'FRESHMAN','SOPHOMORE','JUNIOR'}: return 'UNDER'
    return 'OTHER'


# ============== PART 1: Peer-spillover with school FE ===================

def part1_peer_school_fe():
    files = sorted(glob.glob(os.path.join(RAW_DATA, '*_rosters_panel.csv')))
    fr = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if 'sport' not in df.columns: df['sport'] = 'Track'
        fr.append(df)
    panel = pd.concat(fr, ignore_index=True)
    for col, k in [('name','name_k'), ('school_name','school_k'),
                    ('gender','gender_k'), ('sport','sport_k')]:
        panel[k] = (panel[col].fillna('').str.upper()
                     .str.replace(r'\s+', ' ', regex=True).str.strip())
    panel['athlete_id'] = (panel['school_k'] + '|' + panel['gender_k'] +
                            '|' + panel['sport_k'] + '|' + panel['name_k'])
    panel['class_norm'] = panel['class_year'].apply(norm_class)
    panel['year'] = pd.to_numeric(panel['year'], errors='coerce').astype('Int64')
    panel = panel[panel['year'].notna()].copy()
    panel['year'] = panel['year'].astype(int)
    panel['is_intl'] = ((panel['country_code'].notna()) &
                       (panel['country_code'] != 'USA')).astype(int)

    seen = panel.groupby(['athlete_id'])['year'].apply(set).to_dict()
    panel['return_next'] = panel.apply(
        lambda r: int((r['year']+1) in seen.get(r['athlete_id'], set())),
        axis=1)

    work = panel[(panel['class_norm']=='UNDER') &
                  (panel['year'] < 2023)].copy()

    shocks = pd.read_csv(os.path.join(RAW_DATA,
                          'country_year_sport_panel.csv'))
    shocks = shocks[['country_code','year','any_conflict',
                      'political_stability','gdp_shock',
                      'currency_crisis']].drop_duplicates(
                      subset=['country_code','year'])
    m = shocks['political_stability'].mean()
    s = shocks['political_stability'].std()
    shocks['polstab_drop'] = ((shocks['political_stability']-m)/s < -1
                              ).astype(int)
    work = work.merge(shocks, on=['country_code','year'], how='left')

    # Build team exposure: avg shock indicator among intl teammates × intl share
    grp = work.groupby(['school_k','gender_k','sport_k','year'])
    team_total = grp.size().rename('team_n_total')
    intl_grp = work[work['is_intl']==1].groupby(
        ['school_k','gender_k','sport_k','year'])
    team_intl_n = intl_grp.size().rename('team_n_intl')
    pieces = [team_total, team_intl_n]
    for v in ['any_conflict','polstab_drop','gdp_shock','currency_crisis']:
        agg = intl_grp[v].mean().rename(f'intl_mean_{v}')
        pieces.append(agg)
    team_chars = pd.concat(pieces, axis=1).reset_index()
    team_chars['intl_share'] = (team_chars['team_n_intl'].fillna(0) /
                                  team_chars['team_n_total'])
    for v in ['any_conflict','polstab_drop','gdp_shock','currency_crisis']:
        team_chars[f'expo_{v}'] = (team_chars['intl_share'].fillna(0) *
                                     team_chars[f'intl_mean_{v}'].fillna(0))

    dom = work[work['is_intl']==0].merge(
        team_chars, on=['school_k','gender_k','sport_k','year'], how='left')

    def run(tv, with_school_fe=False):
        sub = dom[dom[tv].notna()].copy()
        y = sub['return_next'].astype(float)
        if with_school_fe:
            for c in [tv, 'intl_share', 'return_next']:
                sub[c+'_dm'] = (sub[c].astype(float) -
                                  sub.groupby('school_k')[c]
                                     .transform('mean'))
            y = sub['return_next_dm']
            X = pd.concat([
                sub[[tv+'_dm','intl_share_dm']].rename(
                    columns={tv+'_dm': tv, 'intl_share_dm':'intl_share'}),
                pd.get_dummies(sub['year'], prefix='y',
                                drop_first=True).astype(float),
                pd.get_dummies(sub['sport_k'], prefix='sp',
                                drop_first=True).astype(float),
            ], axis=1)
            spec = 'school + year + sport FE (within)'
        else:
            X = pd.concat([
                sub[[tv,'intl_share']].astype(float),
                pd.get_dummies(sub['year'], prefix='y',
                                drop_first=True).astype(float),
                pd.get_dummies(sub['sport_k'], prefix='sp',
                                drop_first=True).astype(float),
            ], axis=1)
            spec = 'year + sport FE'
        X = sm.add_constant(X)
        idx = X.dropna().index
        m = sm.OLS(y.loc[idx], X.loc[idx]).fit(cov_type='cluster',
            cov_kwds={'groups': sub.loc[idx,'school_k']})
        b = m.params[tv]; se = m.bse[tv]; p = m.pvalues[tv]
        st = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        return {'spec': spec, 'expo': tv, 'beta': round(b,4),
                'se': round(se,4), 'p': round(p,4),
                'stars': st, 'N': len(idx)}

    rows = []
    for tv in ['expo_any_conflict','expo_polstab_drop',
                'expo_gdp_shock','expo_currency_crisis']:
        rows.append(run(tv, with_school_fe=False))
        rows.append(run(tv, with_school_fe=True))
    return pd.DataFrame(rows)


# ============== PART 2: TFRRS event study =============================

TIMING = {'60 Meters','100 Meters','200 Meters','400 Meters',
          '800 Meters','1500 Meters','1,500 Meters','Mile',
          '3000 Meters','3,000 Meters','5000 Meters','5,000 Meters',
          '10,000 Meters','60 Hurdles','100 Hurdles','110 Hurdles',
          '400 Hurdles','3000 Meter Steeplechase'}
DISTANCE = {'Long Jump','Triple Jump','High Jump','Pole Vault',
            'Shot Put','Discus','Hammer','Javelin','Weight Throw'}


def parse_mark(mark, event):
    if pd.isna(mark): return np.nan
    s = re.sub(r'\([^)]*\)', '', str(mark)).strip().replace(',', '')
    if event in DISTANCE:
        m = re.match(r'^([\d.]+)\s*m?$', s)
        return float(m.group(1)) if m else np.nan
    if event in TIMING:
        m = re.match(r'^(\d+):([\d.]+)$', s)
        if m: return int(m.group(1))*60 + float(m.group(2))
        m = re.match(r'^([\d.]+)$', s)
        return float(m.group(1)) if m else np.nan
    return np.nan


def part2_tfrrs_event_study():
    df = pd.read_csv(os.path.join(RAW_DATA, 'tfrrs_performance.csv'),
                     low_memory=False)
    df = df[df['event'].notna() & df['event'].isin(TIMING|DISTANCE)].copy()
    df['year'] = df['year'].astype(int)
    df['mark_num'] = df.apply(lambda r: parse_mark(r['mark'], r['event']),
                              axis=1)
    df = df[df['mark_num'].notna() & (df['mark_num']>0)].copy()
    df['mark_z'] = (df.groupby(['event','year'])['mark_num']
                       .transform(lambda x: (x-x.mean())/x.std()))
    df.loc[df['event'].isin(TIMING), 'mark_z'] *= -1.0
    df = df[df['mark_z'].notna()].copy()
    df = (df.sort_values('mark_z', ascending=False)
              .groupby(['athlete','event','year']).head(1)
              .reset_index(drop=True))

    # Build shock-onset event time per country
    shocks = pd.read_csv(os.path.join(RAW_DATA,
                          'country_year_sport_panel.csv'))
    shocks = shocks[['country_code','year','any_conflict',
                      'political_stability']].drop_duplicates(
                      subset=['country_code','year'])
    m = shocks['political_stability'].mean()
    s = shocks['political_stability'].std()
    shocks['polstab_drop'] = ((shocks['political_stability']-m)/s < -1
                              ).astype(int)

    def make_event_time(ind):
        df_s = shocks[['country_code','year',ind]].sort_values(
            ['country_code','year'])
        out_rows = []
        for c, g in df_s.groupby('country_code'):
            g = g.sort_values('year')
            prior = 0; onset = None
            for _, r in g.iterrows():
                if r[ind]==1 and prior==0:
                    onset = int(r['year']); break
                if pd.notna(r[ind]): prior = max(prior, int(r[ind]))
            for _, r in g.iterrows():
                k = int(r['year']) - onset if onset else np.nan
                out_rows.append({'country_code': c, 'year': int(r['year']),
                                  f'k_{ind}': k})
        return pd.DataFrame(out_rows)

    results = {}
    for ind in ['any_conflict','polstab_drop']:
        et = make_event_time(ind)
        d = df.merge(et, on=['country_code','year'], how='left')
        d = d[d[f'k_{ind}'].notna() & d[f'k_{ind}'].between(-3,2)].copy()
        if d.empty: continue
        d['ath_event'] = d['athlete'] + '|' + d['event']
        # within-athlete-event demean
        for k in [-3,-2,0,1,2]:
            d[f'k{k}'] = (d[f'k_{ind}']==k).astype(float)
        dum_cols = [f'k{k}' for k in [-3,-2,0,1,2]]
        for c in dum_cols + ['mark_z']:
            d[c+'_dm'] = (d[c].astype(float) -
                            d.groupby('ath_event')[c].transform('mean'))
        y = d['mark_z_dm']
        X = pd.concat([
            d[[c+'_dm' for c in dum_cols]].rename(
                columns={c+'_dm':c for c in dum_cols}),
            pd.get_dummies(d['year'], prefix='y',
                            drop_first=True).astype(float),
        ], axis=1)
        X = sm.add_constant(X)
        idx = X.dropna().index
        if len(idx) < 50:
            results[ind] = None; continue
        m_ = sm.OLS(y.loc[idx], X.loc[idx]).fit(cov_type='cluster',
            cov_kwds={'groups': d.loc[idx,'country_code']})
        rows = []
        for k in [-3,-2,-1,0,1,2]:
            n_k = int((d[f'k_{ind}']==k).sum())
            if k==-1:
                rows.append({'k':k,'beta':0.0,'se':0.0,'p':None,'N':n_k})
            else:
                c = f'k{k}'
                if c not in m_.params: continue
                rows.append({'k':k,'beta':m_.params[c],
                              'se':m_.bse[c],'p':m_.pvalues[c],'N':n_k})
        results[ind] = pd.DataFrame(rows)
    return results


def main():
    warnings.filterwarnings('ignore')
    fh = open(os.path.join(OUTPUT, 'peer_school_fe.txt'), 'w')

    print('=== Part 1: Peer-spillover with school FE ===')
    res1 = part1_peer_school_fe()
    print(res1.to_string(index=False))
    fh.write('Part 1: Peer-spillover with school FE\n')
    fh.write(res1.to_string(index=False) + '\n\n')
    res1.to_csv(os.path.join(OUTPUT, 'peer_school_fe.csv'), index=False)

    print('\n=== Part 2: TFRRS event study ===')
    res2 = part2_tfrrs_event_study()
    fh.write('Part 2: TFRRS event study (within athlete×event + year FE)\n')
    for ind, tbl in res2.items():
        if tbl is None:
            print(f'  {ind}: insufficient events')
            continue
        print(f'\n--- TFRRS {ind} ---')
        print(tbl.to_string(index=False))
        fh.write(f'\nTFRRS {ind}\n' + tbl.to_string(index=False) + '\n')
        tbl.to_csv(os.path.join(OUTPUT,
            f'tfrrs_event_study_{ind}.csv'), index=False)

    fh.close()
    print(f'\nWrote {os.path.join(OUTPUT, "peer_school_fe.txt")}')


if __name__ == '__main__':
    main()
