"""
Fetch additional country-year shock data:
  - CPI inflation (WDI: FP.CPI.TOTL.ZG)
  - UCDP/PRIO battle-related deaths (UCDP REST API)
  - UNHCR refugees + asylum seekers by origin (UNHCR Refugee Statistics)

Writes:
  raw_data/wdi_inflation.csv          country_code, year, inflation_pct
  raw_data/ucdp_battle_deaths.csv     country_code, year, battle_deaths
  raw_data/unhcr_refugees.csv         country_code, year, refugees_origin

Each file is a country-year long panel for 2010-2023 where available.
"""
from __future__ import annotations

import os
import time
import pandas as pd
import requests

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
HEADERS  = {'User-Agent': 'StudentAthletesResearch/1.0 (alina@malkova.net)'}


# ---------- (1) WDI inflation ----------
def fetch_inflation():
    import wbgapi as wb
    print('Fetching WDI FP.CPI.TOTL.ZG (consumer price inflation, annual %)...')
    df = wb.data.DataFrame('FP.CPI.TOTL.ZG', time=range(2010, 2024), labels=False)
    df = df.reset_index().rename(columns={'economy': 'country_code'})
    long = df.melt(id_vars='country_code', var_name='year', value_name='inflation_pct')
    long['year'] = long['year'].str.replace('YR', '').astype(int)
    long = long.dropna(subset=['inflation_pct'])
    out = os.path.join(RAW_DATA, 'wdi_inflation.csv')
    long.to_csv(out, index=False)
    print(f'  wrote {out}: {len(long)} country-year rows, '
          f'{long["country_code"].nunique()} countries')


# ---------- (1b) WDI terms of trade ----------
def fetch_terms_of_trade():
    import wbgapi as wb
    print('\nFetching WDI TT.PRI.MRCH.XD.WD (terms of trade index, 2015=100)...')
    df = wb.data.DataFrame('TT.PRI.MRCH.XD.WD', time=range(2010, 2024), labels=False)
    df = df.reset_index().rename(columns={'economy': 'country_code'})
    long = df.melt(id_vars='country_code', var_name='year',
                   value_name='terms_of_trade_index')
    long['year'] = long['year'].str.replace('YR', '').astype(int)
    long = long.dropna(subset=['terms_of_trade_index'])
    out = os.path.join(RAW_DATA, 'wdi_terms_of_trade.csv')
    long.to_csv(out, index=False)
    print(f'  wrote {out}: {len(long)} country-year rows, '
          f'{long["country_code"].nunique()} countries')


# ---------- (2) UCDP battle deaths ----------
def fetch_battle_deaths():
    """
    UCDP API: https://ucdpapi.pcr.uu.se/api/battledeaths/24.1
    Each row is a conflict-year with bd_best (best estimate).
    We aggregate to country-year by primary location (gwno_loc / loc_dyad).
    """
    print('\nFetching UCDP battle deaths (Battle-Related Deaths v24.1)...')
    base = 'https://ucdpapi.pcr.uu.se/api/battledeaths/24.1'
    rows = []
    page = 1
    while True:
        try:
            r = requests.get(base, params={'pagesize': 1000, 'page': page},
                             headers=HEADERS, timeout=30)
            j = r.json()
        except Exception as e:
            print(f'  err on page {page}: {e}')
            break
        results = j.get('Result', [])
        if not results:
            break
        rows.extend(results)
        total_pages = j.get('TotalPages', 1)
        print(f'  page {page}/{total_pages}: +{len(results)} rows '
              f'(total so far: {len(rows)})')
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.3)

    if not rows:
        print('  no data fetched -- API may have changed; falling back to '
              'UCDP REST endpoint')
        return

    df = pd.DataFrame(rows)
    print(f'  raw: {len(df)} rows, columns: {sorted(df.columns)[:15]}...')
    # Inspect: typical fields = year, conflict_id, location, gwnoloc, bd_best
    # Aggregate by location-year using best estimate
    if 'location' in df.columns and 'year' in df.columns and 'bd_best' in df.columns:
        # location may be "Afghanistan" or "Afghanistan, Pakistan" -- split on comma
        df['country_name'] = df['location'].astype(str).str.split(',').str[0].str.strip()
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['bd_best'] = pd.to_numeric(df['bd_best'], errors='coerce').fillna(0)
        agg = (df.dropna(subset=['country_name', 'year'])
                  .groupby(['country_name', 'year'], as_index=False)
                  ['bd_best'].sum()
                  .rename(columns={'bd_best': 'battle_deaths'}))
        # Translate country_name -> country_code via UCDP conflict file (already
        # has both columns)
        cf = pd.read_csv(os.path.join(RAW_DATA, 'ucdp_conflict.csv'))
        name_to_code = dict(cf.drop_duplicates('country_name')[['country_name', 'iso3']].values)
        agg['country_code'] = agg['country_name'].map(name_to_code)
        out = os.path.join(RAW_DATA, 'ucdp_battle_deaths.csv')
        agg[['country_code', 'country_name', 'year', 'battle_deaths']].to_csv(out, index=False)
        print(f'  wrote {out}: {len(agg)} country-year rows, '
              f'{agg["country_code"].notna().sum()} with iso3')
    else:
        print(f'  unexpected schema -- columns: {df.columns.tolist()}')


# ---------- (3) UNHCR refugees ----------
def fetch_refugees():
    """
    UNHCR Refugee Statistics API. The API returns bilateral
    (origin x destination) rows; we aggregate to origin-year totals.
    """
    print('\nFetching UNHCR refugees + asylum seekers (bilateral, then aggregating)...')
    base = 'https://api.unhcr.org/population/v1/population/'
    all_rows = []
    for year in range(2010, 2024):
        page = 1
        while True:
            try:
                r = requests.get(base, params={
                    'yearFrom': year, 'yearTo': year, 'limit': 5000,
                    'page': page,
                    'coo_all': 'true', 'coa_all': 'false',
                    'columns[]': ['refugees', 'asylum_seekers'],
                }, headers=HEADERS, timeout=60)
                j = r.json()
                items = j.get('items', [])
                if not items:
                    break
                all_rows.extend(items)
                # Stop when we've drained the year (returned fewer than limit)
                if len(items) < 5000:
                    break
                page += 1
                time.sleep(0.3)
            except Exception as e:
                print(f'  err {year} page {page}: {e}')
                break
        print(f'  {year}: total accumulated {len(all_rows)}')
        time.sleep(0.5)

    if not all_rows:
        print('  no data -- aborting')
        return

    df = pd.DataFrame(all_rows)
    df['year']     = pd.to_numeric(df['year'], errors='coerce')
    df['refugees']       = pd.to_numeric(df['refugees'], errors='coerce').fillna(0)
    df['asylum_seekers'] = pd.to_numeric(df['asylum_seekers'], errors='coerce').fillna(0)
    df = df[df['coo_iso'] != '-']  # drop the aggregated rows where origin is "various"

    agg = (df.groupby(['coo_iso', 'year'], as_index=False)
              .agg(refugees=('refugees', 'sum'),
                   asylum_seekers=('asylum_seekers', 'sum')))
    agg = agg.rename(columns={'coo_iso': 'country_code'})
    agg['refugees_plus_asylum'] = agg['refugees'] + agg['asylum_seekers']

    out = os.path.join(RAW_DATA, 'unhcr_refugees.csv')
    agg.to_csv(out, index=False)
    print(f'  wrote {out}: {len(agg)} rows, '
          f'{agg["country_code"].nunique()} origin countries, '
          f'years {agg["year"].min()}-{agg["year"].max()}')


def main():
    fetch_inflation()
    fetch_terms_of_trade()
    # fetch_battle_deaths()  -- UCDP API now requires registered token; skip.
    # The existing ucdp_conflict.csv has the binary indicators we need.
    fetch_refugees()


if __name__ == '__main__':
    main()
