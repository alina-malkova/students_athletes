"""
Selenium-driven scraper for track rosters that need JavaScript rendering.

Reads raw_data/track_selenium_deferred.csv (the schools the requests-based
runner couldn't handle), opens each in headless Chrome, and feeds the
fully-rendered HTML through the same Sidearm/Presto card parsers used in
run_track_scrape.py. Output appends to raw_data/track_rosters.csv.

Uses Chrome for Testing from ~/chrome-for-testing — no /Applications install
needed.

Usage:
    python run_track_selenium.py [--limit N] [--delay 2.0]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import pandas as pd
from io import StringIO

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Reuse parsers from the requests-based runner.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_track_scrape import (
    scrape_sidearm_cards as _parse_sidearm_cards_html,  # noqa: F401  unused name
    append_to_roster_csv,
    infer_gender_from_url,
)

# We need the pure parsing logic, not the requests-fetching wrapper. Re-implement
# the parser to operate on a BeautifulSoup soup directly.
import re
from bs4 import BeautifulSoup

HERE          = os.path.dirname(os.path.abspath(__file__))
RAW_DATA      = os.path.join(HERE, 'raw_data')
DEFERRED_FILE = os.path.join(RAW_DATA, 'track_selenium_deferred.csv')
SUCCESS_FILE  = os.path.join(RAW_DATA, 'track_rosters.csv')
FAILURE_FILE  = os.path.join(RAW_DATA, 'track_roster_failures.csv')
LOG_FILE      = os.path.join(RAW_DATA, 'track_selenium.log')

# Chrome for Testing paths (downloaded by the user, no admin needed).
HOME       = os.path.expanduser('~')
CHROME_BIN = os.path.join(HOME, 'chrome-for-testing/chrome-mac-x64',
                          'Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing')
CHROMEDRIVER = os.path.join(HOME, 'chrome-for-testing/chromedriver-mac-x64/chromedriver')


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as fh:
        fh.write(msg + '\n')


def setup_driver() -> webdriver.Chrome:
    opts = Options()
    opts.binary_location = CHROME_BIN
    opts.add_argument('--headless=new')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    opts.add_argument('--disable-gpu')
    opts.add_argument('--window-size=1920,1080')
    opts.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X) Chrome/148.0.7778.97 Safari/537.36')
    return webdriver.Chrome(service=Service(CHROMEDRIVER), options=opts)


# ---- Pure parsers operating on a BeautifulSoup object ----------------------

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


def parse_sidearm_cards(soup: BeautifulSoup, school_name: str, gender: str) -> pd.DataFrame:
    rows = []
    for card in soup.select('.s-person-card'):
        parts = [p.strip() for p in card.get_text('|', strip=True).split('|') if p.strip()]
        parts = [p for p in parts
                 if not p.lower().startswith(('full bio', 'expand for more', 'for '))]
        if not parts:
            continue
        a = {
            'school_name': school_name, 'gender': gender,
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
    return pd.DataFrame(rows)


def parse_sidearm_table(soup: BeautifulSoup, school_name: str, gender: str) -> pd.DataFrame:
    rows = []
    cls_re = (
        r'\b(Fr\.?|So\.?|Jr\.?|Sr\.?|Gr\.?|Freshman|Sophomore|Junior|Senior|'
        r'Graduate|R-Fr\.?|R-So\.?|R-Jr\.?|R-Sr\.?)\b'
    )
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
                'school_name': school_name, 'gender': gender,
                'jersey_number': '', 'name': '', 'position': '',
                'class_year': '', 'height': '', 'hometown': '',
                'high_school_club': '',
            }
            for i, col in enumerate(cols):
                txt = col.get_text(strip=True)
                if i < len(headers):
                    h = headers[i]
                    if 'no' in h or '#' in h or 'num' in h: a['jersey_number'] = txt
                    elif 'name' in h or 'player' in h:    a['name'] = txt
                    elif 'pos' in h:                       a['position'] = txt
                    elif any(k in h for k in ['year', 'class', 'cl.', 'yr.']): a['class_year'] = txt
                    elif 'height' in h or 'ht' in h:       a['height'] = txt
                    elif 'hometown' in h or 'city' in h:
                        if '/' in txt:
                            l, r = txt.split('/', 1)
                            a['hometown'] = l.strip(); a['high_school_club'] = r.strip()
                        else:
                            a['hometown'] = txt
                    elif any(k in h for k in ['school', 'club', 'previous']):
                        a['high_school_club'] = txt
                if not a['class_year'] and re.search(cls_re, txt, re.I):
                    a['class_year'] = txt
            if a['name']:
                rows.append(a)
    return pd.DataFrame(rows)


def parse_wmt_cards(soup: BeautifulSoup, school_name: str, gender: str) -> pd.DataFrame:
    """WMT Digital uses .roster-card-item or .roster__player with various inner classes."""
    rows = []
    selectors = ['.roster-card-item', '.roster__player', '.roster-card',
                 '.player-card', 'li.roster-item']
    cards = []
    for sel in selectors:
        # Skip selectors that match very few elements -- likely false matches
        # in side panels rather than the main roster grid.
        cards = soup.select(sel)
        if len(cards) >= 5:
            break
        cards = []
    if not cards:
        cards = soup.select('[data-cy="roster-card"]')
    for card in cards:
        a = {
            'school_name': school_name, 'gender': gender,
            'jersey_number': '', 'name': '', 'position': '',
            'class_year': '', 'height': '', 'hometown': '',
            'high_school_club': '',
        }
        # First try BEM-style sub-element selectors (ASU/Sun Devils, etc.).
        for fld, sels in [
            ('name',             ['.roster-card__title', '.roster-card__title-link', '.roster-card__heading']),
            ('hometown',         ['.roster-card__hometown']),
            ('position',         ['.roster-card__position']),
            ('class_year',       ['.roster-card__year', '.roster-card__class']),
            ('high_school_club', ['.roster-card__high-school', '.roster-card__previous-school']),
            ('height',           ['.roster-card__height']),
            ('jersey_number',    ['.roster-card__number', '.roster-card__jersey']),
        ]:
            for sel in sels:
                el = card.select_one(sel)
                if el:
                    a[fld] = el.get_text(' ', strip=True)
                    break
        # Name fallback: img alt or first heading
        if not a['name']:
            img = card.find('img')
            if img and img.get('alt'):
                a['name'] = img['alt'].strip()
        if not a['name']:
            h = card.find(['h2', 'h3', 'h4'])
            if h:
                a['name'] = h.get_text(' ', strip=True)
        # Stanford-style profile-field walk (label/value pairs in dedicated divs).
        if not a['hometown']:
            for f in card.select('.roster-player-card-profile-field, .roster-players-cards-item__profile-field'):
                lbl_el = f.select_one('.roster-player-card-profile-field__label, [class*="label"]')
                val_el = f.select_one('.roster-player-card-profile-field__value, [class*="value"]')
                if lbl_el and val_el:
                    lbl = lbl_el.get_text(strip=True).lower().rstrip(':')
                    if lbl in LABEL_TO_FIELD and not a[LABEL_TO_FIELD[lbl]]:
                        a[LABEL_TO_FIELD[lbl]] = val_el.get_text(' ', strip=True)
        # Last-resort text walk for label-value pairs
        if not a['hometown']:
            text_parts = [p.strip() for p in card.get_text('|', strip=True).split('|') if p.strip()]
            i = 0
            while i < len(text_parts) - 1:
                lbl = text_parts[i].lower().rstrip(':')
                if lbl in LABEL_TO_FIELD and not a[LABEL_TO_FIELD[lbl]]:
                    a[LABEL_TO_FIELD[lbl]] = text_parts[i + 1]
                    i += 2
                else:
                    i += 1
        if a['name']:
            rows.append(a)
    return pd.DataFrame(rows)


def parse_presto_cards(soup: BeautifulSoup, school_name: str, gender: str) -> pd.DataFrame:
    """Classic Presto/Sidearm: <li class='sidearm-roster-player'> with named sub-classes."""
    rows = []
    for card in soup.select('li.sidearm-roster-player, .sidearm-roster-player'):
        a = {
            'school_name': school_name, 'gender': gender,
            'jersey_number': '', 'name': '', 'position': '',
            'class_year': '', 'height': '', 'hometown': '',
            'high_school_club': '',
        }
        num = card.select_one('.sidearm-roster-player-jersey-number')
        if num: a['jersey_number'] = num.get_text(strip=True)
        nm = card.select_one('.sidearm-roster-player-name a, .sidearm-roster-player-name')
        if nm: a['name'] = nm.get_text(' ', strip=True)
        pos = card.select_one('.sidearm-roster-player-position-long-short, .sidearm-roster-player-position')
        if pos: a['position'] = pos.get_text(strip=True)
        yr = card.select_one('.sidearm-roster-player-academic-year')
        if yr: a['class_year'] = yr.get_text(strip=True)
        ht = card.select_one('.sidearm-roster-player-hometown')
        if ht: a['hometown'] = ht.get_text(strip=True)
        hs = card.select_one('.sidearm-roster-player-highschool, .sidearm-roster-player-previous-school')
        if hs: a['high_school_club'] = hs.get_text(strip=True)
        if a['name']:
            rows.append(a)
    return pd.DataFrame(rows)


def cascade_parse(soup: BeautifulSoup, school: str, gender: str) -> tuple[pd.DataFrame, str]:
    for name, fn in [('Sidearm-cards', parse_sidearm_cards),
                     ('Presto',        parse_presto_cards),
                     ('WMT-cards',     parse_wmt_cards),
                     ('Sidearm-table', parse_sidearm_table)]:
        df = fn(soup, school, gender)
        if df is not None and not df.empty:
            return df, name
    return pd.DataFrame(), ''


# ---- Main ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--delay', type=float, default=2.0)
    p.add_argument('--wait',  type=float, default=6.0,
                   help='seconds to wait after page load for JS to render')
    args = p.parse_args()

    deferred = pd.read_csv(DEFERRED_FILE)
    deferred = deferred[deferred['roster_url'].notna()
                        & deferred['roster_url'].str.startswith('http', na=False)].reset_index(drop=True)
    if args.limit:
        deferred = deferred.head(args.limit)
    log(f"Selenium queue: {len(deferred)} schools")

    already = set()
    if os.path.exists(SUCCESS_FILE):
        already = set(pd.read_csv(SUCCESS_FILE)['school_name'].dropna().unique())
    deferred = deferred[~deferred['school_name'].isin(already)].reset_index(drop=True)
    log(f"After dedup vs existing track_rosters.csv: {len(deferred)}")

    failures = []
    batch = []
    CHECKPOINT = 10

    driver = setup_driver()
    log("Chrome for Testing driver started")

    try:
        for idx, row in deferred.iterrows():
            school = row['school_name']
            url    = row['roster_url']
            gender = infer_gender_from_url(url)
            log(f"\n[{idx + 1}/{len(deferred)}] {school} -> {url}")
            try:
                driver.get(url)
                # WebDriverWait can exit when card *skeletons* appear but
                # before the async-fetched content lands. A flat sleep is
                # more reliable for these JS-heavy sites.
                time.sleep(args.wait)
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                df, scraper = cascade_parse(soup, school, gender)
                if df.empty:
                    log("   no data from any parser")
                    failures.append({'school_name': school, 'roster_url': url,
                                     'probe_status': row.get('probe_status', ''),
                                     'reason': 'selenium_no_data'})
                else:
                    n_ht = df['hometown'].astype(str).str.strip().replace('nan', '').ne('').sum()
                    log(f"   {scraper}: {len(df)} athletes, {n_ht} with hometown")
                    batch.append(df)
            except Exception as e:
                log(f"   exception: {e}")
                failures.append({'school_name': school, 'roster_url': url,
                                 'probe_status': row.get('probe_status', ''),
                                 'reason': f'selenium_exception: {type(e).__name__}'})

            if (idx + 1) % CHECKPOINT == 0 and batch:
                append_to_roster_csv(pd.concat(batch, ignore_index=True), SUCCESS_FILE)
                batch = []
                log(f"   [CHECKPOINT @ {idx + 1}]")

            time.sleep(args.delay)

        if batch:
            append_to_roster_csv(pd.concat(batch, ignore_index=True), SUCCESS_FILE)
    finally:
        driver.quit()

    # Append failures (preserving prior failures)
    if failures:
        new_fail = pd.DataFrame(failures)
        if os.path.exists(FAILURE_FILE):
            existing = pd.read_csv(FAILURE_FILE)
            combined = pd.concat([existing, new_fail], ignore_index=True)
            combined = combined.drop_duplicates(subset=['school_name'], keep='last')
        else:
            combined = new_fail
        combined.to_csv(FAILURE_FILE, index=False)
        log(f"\nSaved {len(combined)} total failures to {FAILURE_FILE}")

    if os.path.exists(SUCCESS_FILE):
        final = pd.read_csv(SUCCESS_FILE)
        n_ht = final['hometown'].astype(str).str.strip().replace('nan', '').ne('').sum()
        log("\n=== Selenium pass complete ===")
        log(f"  Schools covered: {final['school_name'].nunique()}")
        log(f"  Total athletes:  {len(final)}")
        log(f"  With hometown:   {n_ht} ({100 * n_ht / max(len(final), 1):.1f}%)")


if __name__ == '__main__':
    main()
