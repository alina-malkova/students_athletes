"""
TFRRS individual-performance scraper.

For each athlete in raw_data/tfrrs_intl_to_scrape.csv (international
athletes matched between the TFRRS roster pull and the multi-year
school-website roster panel), fetch the athlete profile page and
parse the season-best marks by event and year.

Output: raw_data/tfrrs_performance.csv with columns
   profile_url, school, athlete, event, season, year, mark, country_code
"""
from __future__ import annotations

import os
import re
import time
import warnings
import pandas as pd
import requests
from bs4 import BeautifulSoup

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')


def parse_profile(html):
    """Return list of dicts with event/season/year/mark/meet."""
    soup = BeautifulSoup(html, 'html.parser')
    rows = []
    # Each event has a div containing event name and a table inside
    # with year/mark/meet columns. The structure: each `<div>` holding
    # event-level history has rows like: ['year', 'mark', 'meet (date)']
    # We iterate over h-named tables that show "year | mark | meet".
    for tbl in soup.find_all('table'):
        cls = ' '.join(tbl.get('class') or [])
        if 'table-hover' not in cls:
            continue
        # Find associated event name in the closest preceding h-tag
        # (TFRRS uses a heading right above each table)
        prev = tbl.find_previous(['h2', 'h3', 'h4', 'a'])
        # Better: look for the parent panel-heading text or look upward
        ehead = None
        for p in tbl.find_all_previous(['h2', 'h3', 'h4', 'span'],
                                        limit=8):
            txt = p.get_text(strip=True)
            if not txt: continue
            if re.search(r'(meters|miles|kilometers|hurdles|long jump|'
                          r'high jump|shot|discus|hammer|javelin|relay|'
                          r'mile|yards|pole vault|triple jump|heptathlon|'
                          r'decathlon)', txt, re.I):
                ehead = txt; break
        # season indicator from table class
        season = 'unknown'
        for s, lbl in [('xc_bests', 'XC'), ('indoor', 'Indoor'),
                        ('outdoor', 'Outdoor')]:
            if s in cls: season = lbl
        # parse rows
        for tr in tbl.find_all('tr'):
            tds = [td.get_text(' ', strip=True) for td in tr.find_all('td')]
            if len(tds) < 2: continue
            # Heuristic: first col is year (4 digits) → year-table format
            yr = re.match(r'^(\d{4})', tds[0]) if tds else None
            if yr:
                year = int(yr.group(1)); mark = tds[1]
                meet = tds[2] if len(tds) > 2 else ''
                rows.append({'event': ehead, 'season': season,
                             'year': year, 'mark': mark, 'meet': meet})
            elif len(tds) >= 3 and re.match(r'^\d+\.', tds[0]):
                # alt format: ['7.15', 'meet name', 'date']
                m = re.search(r'(\d{4})', tds[-1] or '')
                year = int(m.group(1)) if m else None
                rows.append({'event': ehead, 'season': season,
                             'year': year, 'mark': tds[0],
                             'meet': tds[1] if len(tds) > 1 else ''})
    return rows


def main():
    warnings.filterwarnings('ignore')
    to_scrape = pd.read_csv(os.path.join(RAW_DATA,
                            'tfrrs_intl_to_scrape.csv'))
    print(f'Athletes to scrape: {len(to_scrape):,}')

    s = requests.Session()
    s.headers['User-Agent'] = ('Mozilla/5.0 (academic research scraper; '
                                'student-athletes panel)')

    out_rows = []
    fail = 0
    for i, row in to_scrape.iterrows():
        url = row['profile_url']
        try:
            r = s.get(url, timeout=15)
            r.raise_for_status()
            recs = parse_profile(r.text)
            for rec in recs:
                rec.update({
                    'profile_url': url,
                    'school': row['school_norm'],
                    'athlete': row['name_norm'],
                    'country_code': row['country_code'],
                })
                out_rows.append(rec)
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f'  fail {url}: {e}')
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(to_scrape)}  rows so far: {len(out_rows):,}  '
                  f'fails: {fail}')
        time.sleep(0.5)

    df = pd.DataFrame(out_rows)
    out = os.path.join(RAW_DATA, 'tfrrs_performance.csv')
    df.to_csv(out, index=False)
    print(f'\nWrote {out}  ({len(df):,} rows; {fail} failures)')
    if len(df):
        print(f'  athletes with results: '
              f'{df["athlete"].nunique():,}')
        print(f'  year range: {df["year"].dropna().min()}-'
              f'{df["year"].dropna().max()}')
        print(f'  unique events: {df["event"].nunique()}')


if __name__ == '__main__':
    main()
