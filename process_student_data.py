"""
Process the student-delivered roster file:

  1. Match each school_name (full-form, e.g. "University of Akron")
     to an IPEDS UNITID via the verified NCAA-IPEDS crosswalk.
  2. Apply the patched country-mapping function to all hometowns,
     producing country_code + country_match_confidence.
  3. Write raw_data/student_rosters_processed.csv.
  4. Write raw_data/school_name_crosswalk.csv that maps:
        student_full_name -> short_name (TFRRS style) -> UNITID
     so the student data can join cleanly to my D-I track / soccer /
     historical panels.

Runs in parallel with the historical scrape; uses no Selenium.
"""
from __future__ import annotations

import os
import re
import sys
import string
import warnings
import numpy as np
import pandas as pd

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')

sys.path.insert(0, HERE)
from run_country_mapping import build_wdi_dict, make_mapper  # uses patched mapping
from run_merge_analysis import aggressive_norm, ROSTER_TO_CROSSWALK_ALIASES


def build_school_to_unitid(school_names, crosswalk_d1):
    """For each unique school name, find UNITID via aggressive_norm + alias map."""
    from thefuzz import process, fuzz
    cw = crosswalk_d1.assign(_norm=crosswalk_d1['ncaa_name'].apply(aggressive_norm))
    cw_norm_list = cw['_norm'].tolist()
    out = {}
    for s in school_names:
        nt = aggressive_norm(s)
        target = ROSTER_TO_CROSSWALK_ALIASES.get(nt, nt)
        # Exact normalized match
        hit = cw[cw['_norm'] == target]
        if len(hit) >= 1:
            row = hit.iloc[0]
            out[s] = (int(row['ipeds_unitid']) if pd.notna(row['ipeds_unitid']) else None,
                      'exact', 100)
            continue
        # Fuzzy on normalized strings
        result = process.extractOne(target, cw_norm_list, scorer=fuzz.token_set_ratio)
        if result and result[1] >= 90:
            row = cw.iloc[cw_norm_list.index(result[0])]
            out[s] = (int(row['ipeds_unitid']) if pd.notna(row['ipeds_unitid']) else None,
                      'fuzzy', int(result[1]))
        else:
            out[s] = (None, 'no_match', int(result[1]) if result else 0)
    return out


def main():
    warnings.filterwarnings('ignore')
    src = os.path.join(RAW_DATA, 'student_rosters_raw.csv')
    df = pd.read_csv(src, low_memory=False)
    print(f'Loaded {len(df):,} rows from {src}')

    # ---- School name matching --------------------------------------
    crosswalk = pd.read_csv(os.path.join(RAW_DATA, 'ncaa_ipeds_crosswalk_verified.csv'))
    # All divisions: D-I (1), D-II (2), D-III (3)
    print(f'\nCrosswalk: {len(crosswalk)} NCAA entries')
    schools_unique = sorted(df['school_name'].dropna().unique())
    print(f'Unique school names in student file: {len(schools_unique)}')

    s2u = build_school_to_unitid(schools_unique, crosswalk)
    matched = sum(1 for v in s2u.values() if v[0] is not None)
    print(f'  Matched: {matched}/{len(s2u)} ({100*matched/len(s2u):.1f}%)')
    no_match = [s for s, v in s2u.items() if v[0] is None]
    print(f'  No match (first 15): {no_match[:15]}')

    df['ipeds_unitid'] = df['school_name'].map(lambda s: s2u.get(s, (None,))[0])
    df['unitid_match_type'] = df['school_name'].map(lambda s: s2u.get(s, (None,'missing'))[1])

    # Save crosswalk
    cw_rows = []
    for s, (uid, mtype, score) in s2u.items():
        cw_rows.append({'school_name_full': s, 'ipeds_unitid': uid,
                        'match_type': mtype, 'match_score': score})
    cw_df = pd.DataFrame(cw_rows)
    cw_out = os.path.join(RAW_DATA, 'school_name_crosswalk.csv')
    cw_df.to_csv(cw_out, index=False)
    print(f'\nWrote {cw_out}')

    # ---- Country mapping ------------------------------------------
    print(f'\nMapping {df["hometown"].notna().sum():,} hometowns to country codes...')
    wdi = build_wdi_dict()
    mapper = make_mapper(wdi)
    # Apply
    results = df['hometown'].apply(mapper)
    df['country_code'] = [r[0] for r in results]
    df['country_match_confidence'] = [r[1] for r in results]

    n_mapped = df['country_code'].notna().sum()
    n_intl = ((df['country_code'].notna()) & (df['country_code'] != 'USA')).sum()
    print(f'  Mapped: {n_mapped:,}/{len(df):,} ({100*n_mapped/len(df):.1f}%)')
    print(f'  International: {n_intl:,}')

    # Confidence breakdown
    print(f'\n  Confidence breakdown:')
    print(df['country_match_confidence'].value_counts().to_string())

    # By sport: international share
    print(f'\n  International share by sport (top 15):')
    by_sport = (df.groupby('sport')
                  .agg(n=('school_name', 'count'),
                       n_intl=('country_code',
                               lambda c: ((c.notna()) & (c != 'USA')).sum()))
                  .assign(intl_pct=lambda x: 100 * x['n_intl'] / x['n'])
                  .sort_values('n_intl', ascending=False))
    print(by_sport.head(15).to_string())

    # By division
    print(f'\n  International share by division:')
    by_div = (df.groupby('division')
                .agg(n=('school_name', 'count'),
                     n_intl=('country_code',
                             lambda c: ((c.notna()) & (c != 'USA')).sum()))
                .assign(intl_pct=lambda x: 100 * x['n_intl'] / x['n']))
    print(by_div.to_string())

    # Top countries
    print(f'\n  Top 15 source countries:')
    intl = df[(df['country_code'].notna()) & (df['country_code'] != 'USA')]
    print(intl['country_code'].value_counts().head(15).to_string())

    # Save
    out = os.path.join(RAW_DATA, 'student_rosters_processed.csv')
    df.to_csv(out, index=False)
    print(f'\nWrote {out}: {df.shape}')


if __name__ == '__main__':
    main()
