"""
Probe historical-roster availability for 10 known-working schools.

For each school we test:
  (1) Native archive URL patterns appended to the current roster URL:
        /season/{year}-{year+1}
        /season/{year}
        /{year}-{year+1}
        /?season={year}
        /seasons/{year}-{year+1}
  (2) Wayback Machine CDX index for snapshots between 2020 and 2024.

Reports which schools have what kind of historical access.
"""
from __future__ import annotations

import os
import re
import time
import requests
import pandas as pd
from urllib.parse import urlparse

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (research; alina@malkova.net)',
    'Accept': 'text/html,application/xhtml+xml',
}

CDX = 'https://web.archive.org/cdx/search/cdx'


PILOT_SCHOOLS = [
    # Mix of platform types and sender sizes
    'Mount St. Mary\'s',  # Sidearm static
    'Rider',              # Sidearm static
    'Stanford',           # WMT
    'Cornell',            # Sidearm card layout
    'Akron',              # Sidearm-cards (Zips)
    'Air Force',          # Sidearm-cards
    'Alabama',            # rolltide.com (rediscovered)
    'Notre Dame',         # rediscovered
    'Arkansas',           # rediscovered
    'Cal Poly',           # rediscovered
]


def native_archive_check(roster_url: str) -> list[tuple[str, int, int]]:
    """
    Try several URL templates with year 2022. Return list of
    (url, status_code, hometown_count) for the ones that returned 200.
    """
    base = roster_url.rstrip('/')
    candidates = [
        f'{base}/season/2022-23',
        f'{base}/season/2022',
        f'{base}/2022-23',
        f'{base}/2022',
        f'{base}?season=2022',
        f'{base}?season=2022-23',
        f'{base}/seasons/2022-23',
    ]
    results = []
    for url in candidates:
        try:
            r = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
            if r.status_code == 200:
                n_ht = r.text.lower().count('hometown')
                # Only count this as a real archive if the page differs from current
                results.append((url, r.status_code, n_ht))
        except Exception:
            continue
    return results


def wayback_snapshots(roster_url: str, year_from=2020, year_to=2024) -> list[str]:
    """Query Wayback CDX index for snapshots. Returns list of snapshot URLs."""
    params = {
        'url': roster_url,
        'from': f'{year_from}0101',
        'to':   f'{year_to}1231',
        'output': 'json',
        'limit': 50,
        'filter': 'statuscode:200',
    }
    try:
        r = requests.get(CDX, params=params, headers=HEADERS, timeout=20)
        rows = r.json()
        if not rows:
            return []
        # First row is header
        urls = []
        for row in rows[1:]:
            ts = row[1]
            orig = row[2]
            wb_url = f'https://web.archive.org/web/{ts}/{orig}'
            urls.append(wb_url)
        return urls
    except Exception as e:
        return []


def main():
    platforms = pd.read_csv(os.path.join(RAW_DATA, 'school_track_platforms.csv'))
    print(f'Platforms file: {len(platforms)} schools')

    results = []
    for school in PILOT_SCHOOLS:
        row = platforms[platforms['school_name'] == school]
        if row.empty:
            print(f'\n--- {school}: not in platforms file ---')
            continue
        url = row.iloc[0]['roster_url']
        status = row.iloc[0]['probe_status']
        print(f'\n--- {school}  ({status}) ---')
        print(f'  current: {url}')

        # (1) Native archives
        native = native_archive_check(url)
        if native:
            for u, s, n_ht in native:
                print(f'  NATIVE  {s}  hometown_n={n_ht:>3}  {u}')
        else:
            print('  NATIVE  none')

        # (2) Wayback snapshots
        snaps = wayback_snapshots(url)
        per_year = {}
        for s in snaps:
            yr = s.split('/web/')[1][:4]
            per_year.setdefault(yr, []).append(s)
        if snaps:
            print(f'  WAYBACK total snapshots: {len(snaps)}')
            for yr in sorted(per_year):
                print(f'    {yr}: {len(per_year[yr])} snapshots, '
                      f'sample: {per_year[yr][0]}')
        else:
            print('  WAYBACK  none')

        results.append({
            'school': school,
            'url': url,
            'native_n': len(native),
            'wayback_n': len(snaps),
            'wayback_years': sorted(per_year.keys()),
        })
        time.sleep(0.5)  # politeness on CDX

    summary = pd.DataFrame(results)
    print('\n=== Summary ===')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
