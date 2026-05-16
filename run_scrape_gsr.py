"""
Scrape NCAA Division I Graduation Success Rate (GSR) and Federal
Graduation Rate (FGR) data.

Endpoint is structurally identical to APR: POST to
https://web3.ncaa.org/aprsearch/gsrsearch with CSRF header. Returns
JSON with one record per (school × sport × cohortYear).

Fields:
  cohortYear   : freshman-entering year (6-year graduation = cohortYear + 6)
  gsr          : Graduation Success Rate (transfer-adjusted, 0-100)
  fgr          : Federal Graduation Rate (no transfer adjustment)
  sportCode    : 'ALL' for overall, else MBA/MBB/etc.
"""
from __future__ import annotations

import os
import warnings
import requests
import pandas as pd
from bs4 import BeautifulSoup

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')

GSR_URL = 'https://web3.ncaa.org/aprsearch/gsrsearch'


def fetch():
    s = requests.Session()
    s.headers['User-Agent'] = ('Mozilla/5.0 (academic research scraper)')
    r = s.get(GSR_URL, timeout=30)
    r.raise_for_status()
    csrf = BeautifulSoup(r.text, 'html.parser').find(
        'meta', {'name': '_csrf'})['content']
    headers = {'X-CSRF-TOKEN': csrf,
               'X-Requested-With': 'XMLHttpRequest',
               'Accept': 'application/json'}
    r = s.post(GSR_URL, data={'_csrf': csrf}, headers=headers, timeout=120)
    r.raise_for_status()
    return r.json()


def main():
    warnings.filterwarnings('ignore')
    print('Fetching NCAA GSR dataset ...')
    records = fetch()
    df = pd.DataFrame(records)
    df = df.rename(columns={
        'orgName':       'school_name',
        'orgId':         'ncaa_org_id',
        'sportDesc':     'sport',
        'cohortYear':    'cohort_year',
    })
    df['gsr'] = pd.to_numeric(df['gsr'], errors='coerce')
    df['fgr'] = pd.to_numeric(df['fgr'], errors='coerce')

    # crosswalk to IPEDS unitid via school name
    cw = pd.read_csv(os.path.join(RAW_DATA,
                     'ncaa_ipeds_crosswalk_verified.csv'))
    cw['key'] = cw['ncaa_name'].str.upper().str.strip()
    df['key'] = df['school_name'].str.upper().str.strip()
    df = df.merge(cw[['key', 'ipeds_unitid']], on='key',
                  how='left').drop(columns='key')

    keep = ['school_name', 'ncaa_org_id', 'ipeds_unitid', 'state',
            'conferenceName', 'sport', 'sportCode', 'cohort_year',
            'gsr', 'fgr', 'gsrReportLink', 'fgrReportLink']
    df = df[[c for c in keep if c in df.columns]]
    df = df.rename(columns={'sportCode': 'sport_code'})

    print(f'  records: {len(df):,}')
    print(f'  cohort years: {sorted(df["cohort_year"].unique())}')
    print(f'  unitid matched: {df["ipeds_unitid"].notna().sum():,}')

    out = os.path.join(RAW_DATA, 'ncaa_gsr.csv')
    df.to_csv(out, index=False)
    print(f'  wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)')

    summ = df.groupby('cohort_year')[['gsr', 'fgr']].mean()
    print('\nMean GSR/FGR by cohort year:')
    print(summ.tail(8).round(1).to_string())


if __name__ == '__main__':
    main()
