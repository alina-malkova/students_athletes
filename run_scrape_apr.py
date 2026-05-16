"""
Scrape NCAA Division I Academic Progress Rate (APR) data.

The NCAA APR search tool at https://web3.ncaa.org/aprsearch/aprsearch
exposes a JSON endpoint at the same URL via POST. With no filters it
returns the full national dataset: ~121K (sport × school × year) rows
covering academic years 2004 through 2025.

Pulls the full dataset in one request, joins to the NCAA-IPEDS
crosswalk on `ncaa_name` ↔ `nameOfficial`, and writes
`raw_data/ncaa_apr.csv` with normalized columns.

Note: The 'apr' field is the multi-year rolling rate scaled by 1000
(867 = 0.867). The 'academicYear' field is the terminal year of the
rolling 4-year window (e.g. 2024 = average over 2020-21 through
2023-24).
"""
from __future__ import annotations

import os
import re
import json
import time
import warnings
import requests
import pandas as pd
from bs4 import BeautifulSoup

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')

APR_URL = 'https://web3.ncaa.org/aprsearch/aprsearch'


def fetch_csrf(session):
    r = session.get(APR_URL, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')
    meta = soup.find('meta', {'name': '_csrf'})
    if not meta:
        raise RuntimeError('CSRF meta tag not found')
    return meta['content']


def fetch_all():
    s = requests.Session()
    s.headers['User-Agent'] = ('Mozilla/5.0 (academic research scraper; '
                                'student-athletes panel)')
    csrf = fetch_csrf(s)
    print(f'  CSRF token acquired: {csrf[:24]}...')

    headers = {
        'X-CSRF-TOKEN': csrf,
        'X-Requested-With': 'XMLHttpRequest',
        'Accept': 'application/json',
    }
    # `displayAll` posts with empty body
    r = s.post(APR_URL, data={'_csrf': csrf}, headers=headers, timeout=120)
    r.raise_for_status()
    data = r.json()
    print(f'  retrieved {len(data):,} records')
    return data


def normalize(records):
    df = pd.DataFrame(records)
    # apr is integer scaled by 1000; cast to int then divide
    df['apr_int'] = pd.to_numeric(df['apr'], errors='coerce')
    df['apr_rate'] = df['apr_int'] / 1000.0
    df = df.rename(columns={
        'nameOfficial':   'school_name',
        'academicYear':   'year',
        'description':    'sport',
        'sportCode':      'sport_code',
        'orgId':          'ncaa_org_id',
        'numOfAthletes':  'n_athletes',
        'penaltyType':    'penalty_type_code',
        'postSeasonType': 'post_season_code',
    })
    keep = ['school_name', 'state', 'sport', 'sport_code', 'year',
            'apr_rate', 'apr_int', 'n_athletes',
            'penalties', 'post_season', 'penalty_type_code',
            'post_season_code', 'ncaa_org_id', 'publicFilename']
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    return df


def attach_unitid(df):
    """Join APR records to IPEDS UNITID via the existing crosswalk."""
    cw = pd.read_csv(os.path.join(RAW_DATA,
                     'ncaa_ipeds_crosswalk_verified.csv'))
    cw = cw[['ncaa_name', 'ipeds_unitid']].copy()
    cw['key'] = cw['ncaa_name'].str.upper().str.strip()
    df = df.copy()
    df['key'] = df['school_name'].str.upper().str.strip()
    merged = df.merge(cw[['key', 'ipeds_unitid']], on='key', how='left')
    merged = merged.drop(columns='key')
    n = merged['ipeds_unitid'].notna().sum()
    print(f'  IPEDS UNITID matched: {n:,} / {len(merged):,} '
          f'({100*n/len(merged):.1f}%)')
    miss = merged[merged['ipeds_unitid'].isna()][
        'school_name'].value_counts().head(15)
    if len(miss):
        print('  Top unmatched school names:')
        for k, v in miss.items():
            print(f'    {v:5d}  {k}')
    return merged


def main():
    warnings.filterwarnings('ignore')
    print('Fetching NCAA APR full dataset...')
    records = fetch_all()
    df = normalize(records)
    print(f'\nDataset shape: {df.shape}')
    print(f'Years covered: {sorted(df["year"].unique())}')
    print(f'Sports: {df["sport"].nunique()} '
          f'({df["sport"].value_counts().head(5).to_dict()})')
    print(f'Schools: {df["school_name"].nunique()}')

    df = attach_unitid(df)

    out = os.path.join(RAW_DATA, 'ncaa_apr.csv')
    df.to_csv(out, index=False)
    print(f'\nWrote {out} ({os.path.getsize(out)/1e6:.1f} MB)')

    # Quick descriptive: median APR by year
    summ = df.groupby('year')['apr_rate'].agg(['median', 'mean', 'count'])
    print('\nMedian/Mean APR by terminal year:')
    print(summ.tail(8).round(3).to_string())


if __name__ == '__main__':
    main()
