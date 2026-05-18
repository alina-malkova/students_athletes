"""
Within-athlete performance × home-country shock regression.

Cleans the heterogeneous mark field in raw_data/tfrrs_performance.csv:
  - timing events (running, hurdles, relays): "X.XX" or "X:XX.XX" → seconds
  - distance events (jumps, throws): "XX.XXm" → meters
  - rows with missing event or unparseable mark are dropped

For each event, marks are z-scored within event × year (so a positive
z always means "better"; timing events get sign-flipped first).

Identification:  Δ z-score for the same athlete across years against
contemporaneous + lagged shocks at the athlete's home country.
"""
from __future__ import annotations

import os
import re
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')


# Clean events we trust to be unambiguous individual marks
TIMING = {'60 Meters', '100 Meters', '200 Meters', '300 Meters',
          '400 Meters', '500 Meters', '600 Meters', '800 Meters',
          '1,000 Meters', '1500 Meters', '1,500 Meters', 'Mile',
          '3000 Meters', '3,000 Meters', '5000 Meters', '5,000 Meters',
          '10,000 Meters', '60 Hurdles', '100 Hurdles', '110 Hurdles',
          '400 Hurdles', '3000 Meter Steeplechase'}
DISTANCE = {'Long Jump', 'Triple Jump', 'High Jump', 'Pole Vault',
            'Shot Put', 'Discus', 'Hammer', 'Javelin', 'Weight Throw'}


def parse_mark(mark, event):
    """Return float numeric mark; for timing returns seconds, for
    distance returns meters."""
    if pd.isna(mark): return np.nan
    s = str(mark).strip()
    # strip wind reading and parenthetical
    s = re.sub(r'\([^)]*\)', '', s).strip()
    s = s.replace(',', '')
    # distance events: trailing 'm'
    if event in DISTANCE:
        m = re.match(r'^([\d.]+)\s*m?$', s)
        if m:
            return float(m.group(1))
        return np.nan
    if event in TIMING:
        # mm:ss.ss
        m = re.match(r'^(\d+):([\d.]+)$', s)
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        m = re.match(r'^([\d.]+)$', s)
        if m:
            return float(m.group(1))
        return np.nan
    return np.nan


def load_shocks():
    p = pd.read_csv(os.path.join(RAW_DATA, 'country_year_sport_panel.csv'))
    cols = ['country_code', 'year', 'gdp_shock', 'currency_crisis',
            'any_conflict', 'political_stability']
    p = p[[c for c in cols if c in p.columns]].drop_duplicates(
        subset=['country_code', 'year'])
    m = p['political_stability'].mean(); s = p['political_stability'].std()
    p['polstab_drop'] = ((p['political_stability'] - m) / s < -1).astype(int)
    p = p.sort_values(['country_code', 'year'])
    for sv in ['any_conflict', 'polstab_drop']:
        p[sv + '_lag1'] = p.groupby('country_code')[sv].shift(1)
    return p


def main():
    warnings.filterwarnings('ignore')
    df = pd.read_csv(os.path.join(RAW_DATA, 'tfrrs_performance.csv'),
                     low_memory=False)
    df = df[df['event'].notna()].copy()
    df = df[df['event'].isin(TIMING | DISTANCE)].copy()
    df['year'] = df['year'].astype(int)
    df['mark_num'] = df.apply(lambda r: parse_mark(r['mark'], r['event']),
                              axis=1)
    df = df[df['mark_num'].notna() & (df['mark_num'] > 0)].copy()
    print(f'Parseable individual marks: {len(df):,}')
    print(f'  Unique athletes: {df["athlete"].nunique():,}')
    print(f'  Events: {df["event"].nunique()}  Years: '
          f'{df["year"].min()}-{df["year"].max()}')

    # Within (event × year), z-score; flip sign for timing so + = better
    df['mark_z'] = (df.groupby(['event', 'year'])['mark_num']
                       .transform(lambda x: (x - x.mean()) / x.std()))
    df.loc[df['event'].isin(TIMING), 'mark_z'] *= -1.0

    # Drop events with insufficient within-event-year variance
    df = df[df['mark_z'].notna()].copy()
    # Keep one mark per athlete-event-year (best should already be there)
    df = (df.sort_values('mark_z', ascending=False)
              .groupby(['athlete', 'event', 'year'])
              .head(1)
              .reset_index(drop=True))
    print(f'After dedup + drop NaN-z: {len(df):,}')

    # Merge shocks
    shocks = load_shocks()
    df = df.merge(shocks, on=['country_code', 'year'], how='left')

    # Restrict to athletes with marks in >=2 distinct years
    yrs = df.groupby('athlete')['year'].nunique()
    keep = yrs[yrs >= 2].index
    df = df[df['athlete'].isin(keep)].copy()
    print(f'After ≥2-year filter: {len(df):,}  athletes: '
          f'{df["athlete"].nunique():,}')

    fh = open(os.path.join(OUTPUT, 'tfrrs_perf_shocks.txt'), 'w')
    fh.write('TFRRS performance × home-country shocks\n')
    fh.write('=' * 64 + '\n')
    fh.write(f'N = {len(df):,}  athletes = {df["athlete"].nunique():,}\n')
    fh.write(f'Events = {df["event"].nunique()}, years '
              f'{df["year"].min()}-{df["year"].max()}\n\n')

    def run(sv, with_athlete_fe=True):
        sub = df[df[sv].notna()].copy()
        sub['ath_event'] = sub['athlete'] + '|' + sub['event']
        if with_athlete_fe:
            for c in [sv, 'mark_z']:
                sub[c + '_dm'] = (sub[c].astype(float) -
                                  sub.groupby('ath_event')[c]
                                     .transform('mean'))
            y = sub['mark_z_dm']
            X = pd.concat([
                sub[[sv + '_dm']].rename(columns={sv + '_dm': sv}),
                pd.get_dummies(sub['year'], prefix='y',
                                drop_first=True).astype(float),
            ], axis=1)
            spec = 'athlete×event + year FE'
        else:
            y = sub['mark_z']
            X = pd.concat([
                sub[[sv]].astype(float),
                pd.get_dummies(sub['year'], prefix='y',
                                drop_first=True).astype(float),
                pd.get_dummies(sub['event'], prefix='ev',
                                drop_first=True).astype(float),
            ], axis=1)
            spec = 'event + year FE'
        X = sm.add_constant(X)
        idx = X.dropna().index
        m = sm.OLS(y.loc[idx], X.loc[idx]).fit(
            cov_type='cluster',
            cov_kwds={'groups': sub.loc[idx, 'country_code']})
        b = m.params[sv]; se = m.bse[sv]; p = m.pvalues[sv]
        stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        return {'spec': spec, 'var': sv, 'beta': round(b, 4),
                'se': round(se, 4), 'p': round(p, 4),
                'stars': stars, 'N': len(idx)}

    rows = []
    for sv in ['any_conflict', 'polstab_drop', 'gdp_shock',
               'currency_crisis', 'any_conflict_lag1',
               'polstab_drop_lag1']:
        if sv not in df.columns: continue
        rows.append(run(sv, with_athlete_fe=False))
        rows.append(run(sv, with_athlete_fe=True))
    res = pd.DataFrame(rows)
    print('\n--- Effect on within-event z-scored mark (+ = better) ---')
    print(res.to_string(index=False))
    fh.write('--- Effect on within-event z-scored mark (+ = better) ---\n')
    fh.write(res.to_string(index=False) + '\n')
    res.to_csv(os.path.join(OUTPUT, 'tfrrs_perf_shocks_table.csv'),
                index=False)

    fh.close()
    print(f'\nWrote {os.path.join(OUTPUT, "tfrrs_perf_shocks.txt")}')


if __name__ == '__main__':
    main()
