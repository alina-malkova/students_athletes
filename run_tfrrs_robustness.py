"""
TFRRS performance robustness diagnostics.

Step 1 of the B+C plan: figure out whether the within-athlete TFRRS
result (-0.39 SD conflict / -0.62 SD polstab in run_tfrrs_perf_shocks.py)
is driven by a small set of country-year cells.

  (a) Country composition of the 244-athlete TFRRS panel.
  (b) Leave-one-country-out: drop each country in turn, re-estimate.
      If any single country swings the coefficient by > 1 SE, the
      result is fragile.
  (c) Cluster bootstrap on country (1,000 reps): resample country
      clusters with replacement; compare to country-clustered SE.
      Country-clustered SE is over-precise if effective clusters
      are << nominal clusters.
"""
from __future__ import annotations

import os, re, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')


TIMING = {'60 Meters','100 Meters','200 Meters','400 Meters',
          '800 Meters','1500 Meters','1,500 Meters','Mile',
          '3000 Meters','3,000 Meters','5000 Meters','5,000 Meters',
          '10,000 Meters','60 Hurdles','100 Hurdles','110 Hurdles',
          '400 Hurdles','3000 Meter Steeplechase'}
DISTANCE = {'Long Jump','Triple Jump','High Jump','Pole Vault',
            'Shot Put','Discus','Hammer','Javelin','Weight Throw'}


def parse_mark(m, ev):
    if pd.isna(m): return np.nan
    s = re.sub(r'\([^)]*\)', '', str(m)).strip().replace(',', '')
    if ev in DISTANCE:
        x = re.match(r'^([\d.]+)\s*m?$', s)
        return float(x.group(1)) if x else np.nan
    if ev in TIMING:
        x = re.match(r'^(\d+):([\d.]+)$', s)
        if x: return int(x.group(1))*60 + float(x.group(2))
        x = re.match(r'^([\d.]+)$', s)
        return float(x.group(1)) if x else np.nan
    return np.nan


def load_data():
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
    yrs = df.groupby('athlete')['year'].nunique()
    df = df[df['athlete'].isin(yrs[yrs>=2].index)].copy()

    shocks = pd.read_csv(os.path.join(RAW_DATA,
                          'country_year_sport_panel.csv'))
    cols = ['country_code','year','gdp_shock','currency_crisis',
            'any_conflict','political_stability']
    shocks = shocks[[c for c in cols if c in shocks.columns]
                    ].drop_duplicates(['country_code','year'])
    m = shocks['political_stability'].mean()
    s = shocks['political_stability'].std()
    shocks['polstab_drop'] = ((shocks['political_stability']-m)/s < -1
                              ).astype(int)
    df = df.merge(shocks, on=['country_code','year'], how='left')
    return df


def within_athlete_event_beta(df, sv):
    """Return point estimate and country-clustered SE for the
    within-athlete×event spec used in run_tfrrs_perf_shocks.py."""
    sub = df[df[sv].notna()].copy()
    sub['ae'] = sub['athlete'] + '|' + sub['event']
    for c in ['mark_z', sv]:
        sub[c+'_dm'] = (sub[c].astype(float) -
                          sub.groupby('ae')[c].transform('mean'))
    y = sub['mark_z_dm']
    X = pd.concat([
        sub[[sv+'_dm']].rename(columns={sv+'_dm': sv}),
        pd.get_dummies(sub['year'], prefix='y',
                        drop_first=True).astype(float),
    ], axis=1)
    X = sm.add_constant(X)
    idx = X.dropna().index
    m_ = sm.OLS(y.loc[idx], X.loc[idx]).fit(cov_type='cluster',
        cov_kwds={'groups': sub.loc[idx,'country_code']})
    return m_.params[sv], m_.bse[sv], len(idx)


def main():
    warnings.filterwarnings('ignore')
    df = load_data()
    fh = open(os.path.join(OUTPUT, 'tfrrs_robustness.txt'), 'w')

    # (a) Country composition --------------------------------------------
    comp = (df.groupby('country_code')
              .agg(n_obs=('mark_z','size'),
                    n_ath=('athlete','nunique'),
                    n_shock_obs_conflict=('any_conflict',
                        lambda x: int(x.fillna(0).sum())),
                    n_shock_obs_polstab=('polstab_drop',
                        lambda x: int(x.fillna(0).sum())))
              .sort_values('n_obs', ascending=False))
    print(f'TFRRS panel: {len(df):,} obs, '
          f'{df["athlete"].nunique():,} athletes, '
          f'{df["country_code"].nunique():,} countries.\n')
    print('Top 15 countries by obs:')
    print(comp.head(15).to_string())
    fh.write(f'TFRRS panel: {len(df):,} obs, '
              f'{df["athlete"].nunique():,} athletes, '
              f'{df["country_code"].nunique():,} countries.\n\n')
    fh.write('Country composition (top 15):\n')
    fh.write(comp.head(15).to_string() + '\n\n')

    # (b) Leave-one-country-out ------------------------------------------
    print('\n--- Leave-one-country-out (top 10 by impact) ---')
    fh.write('Leave-one-country-out (top 10 by impact)\n')
    for sv in ['any_conflict', 'polstab_drop']:
        beta0, se0, n0 = within_athlete_event_beta(df, sv)
        rows = []
        for cc in df['country_code'].dropna().unique():
            sub = df[df['country_code'] != cc]
            try:
                b, se, n = within_athlete_event_beta(sub, sv)
                rows.append({'drop': cc, 'beta': round(b, 4),
                              'se': round(se, 4),
                              'delta_from_full': round(b - beta0, 4),
                              'N': n})
            except Exception:
                continue
        r = pd.DataFrame(rows).sort_values('delta_from_full',
                                            key=abs, ascending=False)
        print(f'\n{sv}: full β = {beta0:.4f} (SE {se0:.4f}, N {n0:,})')
        print(r.head(10).to_string(index=False))
        fh.write(f'\n{sv}: full β = {beta0:.4f} (SE {se0:.4f}, N {n0:,})\n')
        fh.write(r.head(10).to_string(index=False) + '\n')
        r.to_csv(os.path.join(OUTPUT, f'tfrrs_loco_{sv}.csv'),
                  index=False)

    # (c) Cluster bootstrap ----------------------------------------------
    rng = np.random.default_rng(42)
    print('\n--- Cluster bootstrap on country (1,000 reps) ---')
    fh.write('\nCluster bootstrap on country (1,000 reps)\n')
    for sv in ['any_conflict', 'polstab_drop']:
        beta0, se0, _ = within_athlete_event_beta(df, sv)
        cc_list = list(df['country_code'].dropna().unique())
        n_cc = len(cc_list)
        boots = []
        for b in range(1000):
            samp = rng.choice(cc_list, size=n_cc, replace=True)
            # form bootstrap dataset by stacking each sampled country
            # (with new tag to keep their athlete-event groups distinct)
            parts = []
            for i, c in enumerate(samp):
                d = df[df['country_code']==c].copy()
                d['athlete'] = d['athlete'] + f'__bs{i}'
                parts.append(d)
            bdf = pd.concat(parts, ignore_index=True)
            try:
                bb, _, _ = within_athlete_event_beta(bdf, sv)
                boots.append(bb)
            except Exception:
                continue
        boots = np.array(boots)
        boot_se = boots.std(ddof=1)
        boot_lo, boot_hi = np.percentile(boots, [2.5, 97.5])
        line = (f'{sv}: full β = {beta0:+.4f}  '
                f'cluster-SE (analytic) = {se0:.4f}  '
                f'bootstrap-SE = {boot_se:.4f}  '
                f'95% CI = [{boot_lo:+.4f}, {boot_hi:+.4f}]  '
                f'(reps = {len(boots)})')
        print(line)
        fh.write(line + '\n')

    fh.close()
    print(f'\nWrote {os.path.join(OUTPUT, "tfrrs_robustness.txt")}')


if __name__ == '__main__':
    main()
