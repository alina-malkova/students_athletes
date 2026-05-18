"""
Country-FE specification and pre-trends event study for §3.8.

The reviewer concern: the baseline retention regression has year + sport
FE but no country FE; identification leans on cross-country variation in
who happens to experience a shock. This script:

  (1) Adds country FE to the baseline, isolating within-country
      variation. Effective sample is countries that switch shock status
      between 2019 and 2023 (e.g., Ukraine, Russia, Myanmar, Afghanistan
      in 2021-22; Venezuela's continued instability).
  (2) Defines a "shock onset" event (year t where indicator goes 0→1
      with no prior shock in 2019-(t-1)) and computes an event-study
      coefficient at k = -2, -1, 0, +1 relative to onset. A negative
      coefficient at k = -1 would flag a pre-trend.

Both run on the 8-sport intl-athlete-year panel restricted to
under-classmen (same baseline as run_retention_shocks.py).
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


def norm_class(c):
    if pd.isna(c): return 'UNK'
    s = str(c).upper().strip().replace('.', '')
    if s in {'FR','SO','JR','FY','R-FR','R-SO','R-JR',
             'FRESHMAN','SOPHOMORE','JUNIOR'}: return 'UNDER'
    if s in {'SR','R-SR','5TH','GR','SENIOR'}: return 'SENIOR'
    return 'UNK'


def load_panel():
    files = sorted(glob.glob(os.path.join(RAW_DATA, '*_rosters_panel.csv')))
    fr = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if 'sport' not in df.columns: df['sport'] = 'Track'
        fr.append(df)
    panel = pd.concat(fr, ignore_index=True)
    panel['name_k']   = (panel['name'].fillna('').str.upper()
                          .str.replace(r'\s+', ' ', regex=True).str.strip())
    panel['school_k'] = panel['school_name'].fillna('').str.upper().str.strip()
    panel['gender_k'] = panel['gender'].fillna('').str.upper().str.strip()
    panel['sport_k']  = panel['sport'].fillna('').str.upper().str.strip()
    panel['athlete_id'] = (panel['school_k'] + '|' + panel['gender_k'] + '|' +
                            panel['sport_k'] + '|' + panel['name_k'])
    panel['class_norm'] = panel['class_year'].apply(norm_class)
    panel['year'] = pd.to_numeric(panel['year'], errors='coerce').astype('Int64')
    panel = panel[panel['year'].notna()].copy()
    panel['year'] = panel['year'].astype(int)
    panel['is_intl'] = ((panel['country_code'].notna()) &
                       (panel['country_code'] != 'USA')).astype(int)
    return panel


def add_return(panel):
    seen = panel.groupby(['athlete_id'])['year'].apply(set).to_dict()
    panel['return_next'] = panel.apply(
        lambda r: int((r['year']+1) in seen.get(r['athlete_id'], set())),
        axis=1)
    return panel


def load_shocks():
    p = pd.read_csv(os.path.join(RAW_DATA, 'country_year_sport_panel.csv'))
    cols = ['country_code', 'year', 'gdp_shock', 'currency_crisis',
            'any_conflict', 'political_stability']
    p = p[[c for c in cols if c in p.columns]].drop_duplicates(
        subset=['country_code', 'year'])
    m = p['political_stability'].mean(); s = p['political_stability'].std()
    p['polstab_drop'] = ((p['political_stability'] - m) / s < -1).astype(int)
    return p.sort_values(['country_code', 'year'])


def make_event_time(shocks, ind):
    """For each country with an onset of `ind` (0→1 with no prior 1),
    label every year by event-time k = year - onset_year. Returns a
    DataFrame indexed by (country_code, year) with column 'k_<ind>'."""
    df = shocks[['country_code', 'year', ind]].copy()
    df = df.sort_values(['country_code', 'year'])
    # First year where ind=1 (and onset means no shock in prior observed years)
    onsets = []
    for c, g in df.groupby('country_code'):
        g = g.sort_values('year')
        # find first year with ind=1 such that all prior years (within obs) had ind=0
        prior = 0
        onset_year = None
        for _, row in g.iterrows():
            if row[ind] == 1 and prior == 0:
                onset_year = int(row['year'])
                break
            prior = max(prior, int(row[ind]) if pd.notna(row[ind]) else 0)
        if onset_year is not None:
            onsets.append({'country_code': c, 'onset_year': onset_year})
    on = pd.DataFrame(onsets)
    out = df.merge(on, on='country_code', how='left')
    out[f'k_{ind}'] = out['year'] - out['onset_year']
    return out[['country_code', 'year', f'k_{ind}']]


def run_basic(intl, sv, country_fe=False):
    sub = intl[intl[sv].notna()].copy()
    pieces = [
        sub[[sv]].astype(float),
        pd.get_dummies(sub['year'], prefix='y', drop_first=True
                        ).astype(float),
        pd.get_dummies(sub['sport'], prefix='sport',
                        drop_first=True).astype(float),
    ]
    if country_fe:
        pieces.append(pd.get_dummies(sub['country_code'], prefix='cc',
                                       drop_first=True).astype(float))
    X = pd.concat(pieces, axis=1)
    X = sm.add_constant(X)
    idx = X.dropna().index
    m = sm.OLS(sub.loc[idx, 'return_next'].astype(float),
               X.loc[idx]).fit(cov_type='cluster',
                                cov_kwds={'groups':
                                  sub.loc[idx, 'country_code']})
    b = m.params[sv]; se = m.bse[sv]; p = m.pvalues[sv]
    stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
    return {'spec': 'country FE' if country_fe else 'no country FE',
            'shock': sv, 'beta': round(b, 4),
            'se': round(se, 4), 'p': round(p, 4),
            'stars': stars, 'N': len(idx)}


def event_study(intl, ev_col, label):
    """Estimate Pr(return) on event-time dummies k = -2, -1, 0, +1
    relative to shock onset, with country + year + sport FE."""
    df = intl[intl[ev_col].notna()].copy()
    # restrict to event-time window we can identify
    df = df[df[ev_col].between(-3, 2)].copy()
    if df.empty: return None
    # base = k = -1 (year before onset) — only one we drop
    for k in [-3, -2, 0, 1, 2]:
        df[f'k{k}'] = (df[ev_col] == k).astype(float)
    dum_cols = [f'k{k}' for k in [-3, -2, 0, 1, 2]]
    pieces = [
        df[dum_cols].astype(float),
        pd.get_dummies(df['year'], prefix='y',
                        drop_first=True).astype(float),
        pd.get_dummies(df['sport'], prefix='sport',
                        drop_first=True).astype(float),
        pd.get_dummies(df['country_code'], prefix='cc',
                        drop_first=True).astype(float),
    ]
    X = pd.concat(pieces, axis=1)
    X = sm.add_constant(X)
    idx = X.dropna().index
    m = sm.OLS(df.loc[idx, 'return_next'].astype(float),
                X.loc[idx]).fit(cov_type='cluster',
                                 cov_kwds={'groups':
                                   df.loc[idx, 'country_code']})
    rows = []
    for k in [-3, -2, -1, 0, 1, 2]:
        if k == -1:
            rows.append({'k': k, 'beta': 0.0, 'se': 0.0, 'p': None,
                          'N': int((df[ev_col]==-1).sum())})
        else:
            c = f'k{k}'
            if c not in m.params: continue
            rows.append({'k': k, 'beta': m.params[c], 'se': m.bse[c],
                          'p': m.pvalues[c],
                          'N': int((df[ev_col]==k).sum())})
    return pd.DataFrame(rows), label


def main():
    warnings.filterwarnings('ignore')
    panel = load_panel()
    panel = add_return(panel)
    work = panel[panel['class_norm']=='UNDER'].copy()
    work = work[work['year'] < 2023].copy()
    shocks = load_shocks()
    work = work.merge(shocks, on=['country_code','year'], how='left')
    intl = work[work['is_intl']==1].copy()
    print(f'intl under-classmen N = {len(intl):,}')

    fh = open(os.path.join(OUTPUT, 'retention_country_fe.txt'), 'w')

    # ---------- (1) Country FE comparison -------------------------------
    print('\n=== (1) Country-FE comparison ===')
    fh.write('(1) Country-FE comparison\n')
    rows = []
    for sv in ['any_conflict', 'polstab_drop', 'gdp_shock',
               'currency_crisis']:
        if sv not in intl.columns: continue
        rows.append(run_basic(intl, sv, country_fe=False))
        rows.append(run_basic(intl, sv, country_fe=True))
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    fh.write(res.to_string(index=False) + '\n\n')
    res.to_csv(os.path.join(OUTPUT, 'retention_country_fe.csv'),
                index=False)

    # ---------- (2) Event study ----------------------------------------
    print('\n=== (2) Pre-trends event study ===')
    fh.write('(2) Pre-trends event study\n')
    for ind, label in [('any_conflict', 'Armed conflict onset'),
                       ('polstab_drop', 'Political-stability drop onset')]:
        et = make_event_time(shocks, ind)
        intl2 = intl.merge(et, on=['country_code', 'year'], how='left')
        out = event_study(intl2, f'k_{ind}', label)
        if out is None:
            print(f'  {label}: insufficient events')
            continue
        tbl, lbl = out
        print(f'\n--- {lbl} ---')
        print(tbl.to_string(index=False))
        fh.write(f'\n{lbl}\n' + tbl.to_string(index=False) + '\n')
        tbl.to_csv(os.path.join(OUTPUT,
            f'retention_event_study_{ind}.csv'), index=False)

    # plot the polstab event study
    for ind in ['any_conflict', 'polstab_drop']:
        path = os.path.join(OUTPUT, f'retention_event_study_{ind}.csv')
        if not os.path.exists(path): continue
        d = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        d['ci_lo'] = d['beta'] - 1.96 * d['se']
        d['ci_hi'] = d['beta'] + 1.96 * d['se']
        ax.errorbar(d['k'], d['beta'],
                    yerr=[d['beta']-d['ci_lo'], d['ci_hi']-d['beta']],
                    fmt='o-', color='steelblue', capsize=4)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(-0.5, color='red', linestyle='--', linewidth=0.8,
                   label='shock onset')
        ax.set_xlabel('Event time (years from shock onset)')
        ax.set_ylabel(r'$\beta$ on Pr(return), base $k{=}-1$')
        ax.set_title(f'Pre-trends event study: {ind.replace("_"," ")}')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(OUTPUT, f'fig_retention_event_{ind}.png')
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'  wrote {out}')

    fh.close()
    print(f'\nWrote {os.path.join(OUTPUT, "retention_country_fe.txt")}')


if __name__ == '__main__':
    main()
