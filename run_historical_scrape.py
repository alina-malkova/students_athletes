"""
Historical roster scraper: build a multi-year panel by hitting
native archive URLs at each athletic department site.

For each school in school_track_platforms.csv we:
  1. Try a list of archive URL templates with sample year 2022.
     Pick the template that returns the most "City, ST" / hometown
     hits in the HTML body.
  2. Apply that template for years 2019, 2020, 2021, 2022, 2023.
  3. Parse with the cascade of Sidearm-cards / Sidearm-table /
     Presto. (WMT/Selenium-only schools are deferred to a separate
     pass.)

Output: raw_data/track_rosters_panel.csv with all athlete-year rows.
"""
from __future__ import annotations

import os
import re
import sys
import time
import warnings
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_track_scrape import (
    scrape_sidearm_cards as _scrape_cards_requests,  # not used directly
    scrape_sidearm_roster,
    scrape_prestosports_roster,
    infer_gender_from_url,
)
from run_track_scrape import scrape_sidearm_cards

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
LOG_PATH = os.path.join(RAW_DATA, 'historical_scrape.log')

REQ_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'en-US,en;q=0.9',
}

DELAY = 1.0  # seconds between fetches
YEARS_HISTORICAL = [2019, 2020, 2021, 2022, 2023]
DISCOVERY_YEAR = 2022  # year used to find the best URL template

# Templates that worked in the pilot, ordered by how often they yielded data
TEMPLATES = [
    '{base}?season={year}',
    '{base}?season={year}-{year2}',
    '{base}/{year}-{year2}',
    '{base}/season/{year}',
    '{base}/season/{year}-{year2}',
    '{base}/{year}',
    '{base}/seasons/{year}-{year2}',
]


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as fh:
        fh.write(msg + '\n')


def fmt_template(template: str, base: str, year: int) -> str:
    yy = year + 1
    yy2 = str(yy)[-2:]
    return template.format(base=base.rstrip('/'),
                            year=year,
                            year2=yy2)


def count_hometowns_in_html(html: str) -> int:
    """City, ST-style pattern count (more reliable than raw 'hometown' frequency)."""
    return len(re.findall(
        r'>[^<>]*[A-Z][a-zA-Z\.\- ]{2,30},\s*(?:[A-Z]{2}\.?|[A-Z][a-z]+\.?)[^<>]*<',
        html))


def _extract_names(html: str) -> set:
    """Pull all candidate athlete-name strings from an HTML response for
    overlap detection. We use a broad heuristic that catches Sidearm
    cards, classic Sidearm tables, and Presto layouts."""
    soup = BeautifulSoup(html, 'html.parser')
    names = set()
    # Sidearm cards: first text node in each .s-person-card
    for card in soup.select('.s-person-card'):
        parts = [p.strip() for p in card.get_text('|', strip=True).split('|')
                 if p.strip()]
        if parts:
            names.add(parts[0])
    # Classic Sidearm list cards
    for card in soup.select('li.sidearm-roster-player, .sidearm-roster-player'):
        nm = card.select_one('.sidearm-roster-player-name a, .sidearm-roster-player-name')
        if nm:
            names.add(nm.get_text(' ', strip=True))
    # Table layout
    for table in soup.find_all('table'):
        if 'roster' not in str(table).lower():
            continue
        for tr in table.find_all('tr')[1:]:
            cols = tr.find_all('td')
            if len(cols) < 4:
                continue
            for c in cols:
                txt = c.get_text(' ', strip=True)
                # crude name heuristic: two or more capitalised words
                if re.match(r'^[A-Z][a-zA-Z\-\'\.]+\s[A-Z][a-zA-Z\-\'\.\s]+$', txt):
                    names.add(txt)
                    break
    return names


def discover_template(roster_url: str) -> Optional[str]:
    """
    Pick the URL template that yields a real archive: rich hometown
    data AND low athlete-name overlap with the current roster (which
    indicates the server actually honours the year parameter).
    """
    # Current roster as reference for overlap detection
    try:
        cur = requests.get(roster_url, headers=REQ_HEADERS, timeout=15,
                            allow_redirects=True)
        current_names = _extract_names(cur.text) if cur.status_code == 200 else set()
    except Exception:
        current_names = set()

    best = (0, None, 0.0)  # (hometown_count, template, overlap)
    for tmpl in TEMPLATES:
        u = fmt_template(tmpl, roster_url, DISCOVERY_YEAR)
        try:
            r = requests.get(u, headers=REQ_HEADERS, timeout=15,
                              allow_redirects=True)
            if r.status_code != 200:
                continue
            n = count_hometowns_in_html(r.text)
            if n < 5:
                continue
            archive_names = _extract_names(r.text)
            if archive_names and current_names:
                overlap = len(archive_names & current_names) / max(1, len(archive_names))
            else:
                overlap = 0.0
            # Reject if archive is basically the current roster (server
            # ignored the year). Threshold 0.70: ~30% turnover is realistic
            # for a 2-year gap.
            if overlap > 0.70:
                continue
            # Prefer high hometown count, tiebreak by lower overlap
            score = n - 200 * overlap
            if score > best[0]:
                best = (score, tmpl, overlap)
        except Exception:
            continue
    if best[1] is None:
        return None
    return best[1]


def cascade_parse_html(html: str, school: str, gender: str) -> pd.DataFrame:
    """Try Sidearm-cards, Sidearm-table, Presto on the same HTML."""
    # Build a "url-like" string for the existing scrapers, but they actually
    # only need the html. We patch by fetching ourselves and parsing locally.
    soup_text = html
    # Sidearm-cards: parse .s-person-card via the original logic on a
    # temporary file or string. Simpler: re-implement here using
    # BeautifulSoup, mirroring scrape_sidearm_cards.
    rows = []
    soup = BeautifulSoup(html, 'html.parser')

    LABEL_TO_FIELD = {
        'hometown': 'hometown',
        'last school': 'high_school_club',
        'high school': 'high_school_club',
        'previous school': 'high_school_club',
        'club': 'high_school_club',
        'academic year': 'class_year',
        'class': 'class_year',
        'year': 'class_year',
        'position': 'position',
        'event': 'position',
        'events': 'position',
        'height': 'height',
        'no.': 'jersey_number',
        'number': 'jersey_number',
    }

    # 1. Modern Sidearm cards
    cards = soup.select('.s-person-card')
    for card in cards:
        parts = [p.strip() for p in card.get_text('|', strip=True).split('|') if p.strip()]
        parts = [p for p in parts
                 if not p.lower().startswith(('full bio', 'expand for more', 'for '))]
        if not parts:
            continue
        a = {
            'school_name': school, 'gender': gender,
            'jersey_number': '', 'name': parts[0],
            'position': '', 'class_year': '', 'height': '',
            'hometown': '', 'high_school_club': '',
        }
        i = 1
        while i < len(parts) - 1:
            label = parts[i].lower().rstrip(':')
            if label in LABEL_TO_FIELD:
                a[LABEL_TO_FIELD[label]] = parts[i + 1]
                i += 2
            else:
                i += 1
        if a['name']:
            rows.append(a)

    if rows:
        return pd.DataFrame(rows)

    # 2. Sidearm classic table (header-mapped)
    cls_re = r'\b(Fr\.?|So\.?|Jr\.?|Sr\.?|Gr\.?|Freshman|Sophomore|Junior|Senior|Graduate|R-Fr\.?|R-So\.?|R-Jr\.?|R-Sr\.?)\b'
    for table in soup.find_all('table'):
        if 'roster' not in str(table).lower():
            continue
        trs = table.find_all('tr')
        if not trs:
            continue
        headers = [th.get_text(strip=True).lower() for th in trs[0].find_all(['th', 'td'])]
        for tr in trs[1:]:
            cols = tr.find_all('td')
            if len(cols) < 4:
                continue
            a = {
                'school_name': school, 'gender': gender,
                'jersey_number': '', 'name': '', 'position': '',
                'class_year': '', 'height': '', 'hometown': '',
                'high_school_club': '',
            }
            for i, col in enumerate(cols):
                txt = col.get_text(strip=True)
                if i < len(headers):
                    h = headers[i]
                    if 'no' in h or '#' in h or 'num' in h:
                        a['jersey_number'] = txt
                    elif 'name' in h or 'player' in h:
                        a['name'] = txt
                    elif 'pos' in h:
                        a['position'] = txt
                    elif any(k in h for k in ['year', 'class', 'cl.', 'yr.']):
                        a['class_year'] = txt
                    elif 'height' in h or 'ht' in h:
                        a['height'] = txt
                    elif 'hometown' in h or 'city' in h:
                        if '/' in txt:
                            l, r = txt.split('/', 1)
                            a['hometown'] = l.strip()
                            a['high_school_club'] = r.strip()
                        else:
                            a['hometown'] = txt
                    elif any(k in h for k in ['school', 'club', 'previous']):
                        a['high_school_club'] = txt
                if not a['class_year'] and re.search(cls_re, txt, re.I):
                    a['class_year'] = txt
            if a['name']:
                rows.append(a)
    if rows:
        return pd.DataFrame(rows)

    # 3. Presto / classic Sidearm li.sidearm-roster-player
    for card in soup.select('li.sidearm-roster-player, .sidearm-roster-player'):
        a = {
            'school_name': school, 'gender': gender,
            'jersey_number': '', 'name': '', 'position': '',
            'class_year': '', 'height': '', 'hometown': '',
            'high_school_club': '',
        }
        nm = card.select_one('.sidearm-roster-player-name a, .sidearm-roster-player-name')
        if nm:
            a['name'] = nm.get_text(' ', strip=True)
        pos = card.select_one('.sidearm-roster-player-position-long-short, .sidearm-roster-player-position')
        if pos:
            a['position'] = pos.get_text(strip=True)
        yr = card.select_one('.sidearm-roster-player-academic-year')
        if yr:
            a['class_year'] = yr.get_text(strip=True)
        ht = card.select_one('.sidearm-roster-player-hometown')
        if ht:
            a['hometown'] = ht.get_text(strip=True)
        hs = card.select_one('.sidearm-roster-player-highschool, .sidearm-roster-player-previous-school')
        if hs:
            a['high_school_club'] = hs.get_text(strip=True)
        if a['name']:
            rows.append(a)
    return pd.DataFrame(rows)


def main():
    warnings.filterwarnings('ignore')
    platforms = pd.read_csv(os.path.join(RAW_DATA, 'school_track_platforms.csv'))
    # Restrict to schools where requests-only scraping worked (probe_status
    # OK_static or OK_rediscovered). JS-only schools need Selenium and we
    # leave them for a later pass.
    platforms = platforms[platforms['probe_status'].isin(['OK_static',
                                                            'OK_rediscovered'])]
    platforms = platforms[platforms['roster_url'].notna()
                          & platforms['roster_url'].str.startswith('http', na=False)]
    print(f'Schools queued: {len(platforms)}')

    # Resume support: skip schools we already covered for all 5 years
    out_path = os.path.join(RAW_DATA, 'track_rosters_panel.csv')
    done_pairs = set()
    if os.path.exists(out_path):
        prior = pd.read_csv(out_path)
        for _, r in prior[['school_name', 'year']].drop_duplicates().iterrows():
            done_pairs.add((r['school_name'], int(r['year'])))
        print(f'Resuming: {len(done_pairs)} (school, year) pairs already in panel')

    failures = []
    buffer = []

    for idx, row in platforms.reset_index(drop=True).iterrows():
        school = row['school_name']
        url = row['roster_url']
        gender = infer_gender_from_url(url)
        all_done = all((school, y) in done_pairs for y in YEARS_HISTORICAL)
        if all_done:
            continue

        log(f'\n[{idx+1}/{len(platforms)}] {school}')
        log(f'  current URL: {url}')

        # Step 1: discover best template
        tmpl = discover_template(url)
        if tmpl is None:
            log(f'  no working archive template found')
            failures.append({'school_name': school, 'reason': 'no_template'})
            continue
        log(f'  template: {tmpl}')

        # Step 2: fetch each year
        for year in YEARS_HISTORICAL:
            if (school, year) in done_pairs:
                continue
            archive_url = fmt_template(tmpl, url, year)
            try:
                r = requests.get(archive_url, headers=REQ_HEADERS, timeout=20)
                if r.status_code != 200:
                    log(f'  {year}: {r.status_code} {archive_url}')
                    continue
                df_year = cascade_parse_html(r.text, school, gender)
                if df_year.empty:
                    log(f'  {year}: 0 athletes (parser found nothing)')
                    continue
                df_year['year'] = year
                df_year['archive_url'] = archive_url
                buffer.append(df_year)
                log(f'  {year}: {len(df_year)} athletes')
            except Exception as e:
                log(f'  {year}: exception {e}')
            time.sleep(DELAY)

        # Checkpoint every 25 schools
        if buffer and (idx + 1) % 25 == 0:
            df_buf = pd.concat(buffer, ignore_index=True)
            if os.path.exists(out_path):
                old = pd.read_csv(out_path)
                df_buf = pd.concat([old, df_buf], ignore_index=True)
            df_buf = df_buf.drop_duplicates(
                subset=['school_name', 'year', 'name'], keep='last')
            df_buf.to_csv(out_path, index=False)
            log(f'  [CHECKPOINT @ {idx+1}]  total panel rows: {len(df_buf)}')
            buffer = []

    # Final flush
    if buffer:
        df_buf = pd.concat(buffer, ignore_index=True)
        if os.path.exists(out_path):
            old = pd.read_csv(out_path)
            df_buf = pd.concat([old, df_buf], ignore_index=True)
        df_buf = df_buf.drop_duplicates(
            subset=['school_name', 'year', 'name'], keep='last')
        df_buf.to_csv(out_path, index=False)
        log(f'\nFinal: {len(df_buf)} panel rows written to {out_path}')

    if failures:
        pd.DataFrame(failures).to_csv(
            os.path.join(RAW_DATA, 'historical_failures.csv'), index=False)
        log(f'  {len(failures)} failed schools')


if __name__ == '__main__':
    main()
