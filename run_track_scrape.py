"""
Local runner for the track roster scraper.

Mirrors Section 12 of student_athletes_pipeline.ipynb but adapted for local
execution (no Colab, no Drive mount). Skips schools whose probe_status
requires Selenium — those must be run from Colab where Chrome is set up.

Usage:
    python run_track_scrape.py
"""
from __future__ import annotations

import os
import re
import sys
import time
import argparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')

PLATFORMS_FILE   = os.path.join(RAW_DATA, 'school_track_platforms.csv')
SUCCESS_FILE     = os.path.join(RAW_DATA, 'track_rosters.csv')
FAILURE_FILE     = os.path.join(RAW_DATA, 'track_roster_failures.csv')
SELENIUM_DEFERRED = os.path.join(RAW_DATA, 'track_selenium_deferred.csv')
LOG_FILE         = os.path.join(RAW_DATA, 'track_scraper.log')

REQ_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
    'Accept-Language': 'en-US,en;q=0.9',
}

DELAY = 1.5  # seconds between schools


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as fh:
        fh.write(msg + '\n')


# ----------------------------------------------------------------------
# Scraper implementations (extracted from notebook cells 26 + 28).
# WMT/Selenium scraper (cell 27) is intentionally omitted — local Chrome
# isn't available; those schools land in track_selenium_deferred.csv.
# ----------------------------------------------------------------------

def scrape_sidearm_roster(url: str, school_name: str, gender: str) -> pd.DataFrame:
    roster_data = []
    try:
        response = requests.get(url, headers=REQ_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        cls_pattern = (
            r'\b(Fr\.?|So\.?|Jr\.?|Sr\.?|Gr\.?|Freshman|Sophomore|'
            r'Junior|Senior|Graduate|R-Fr\.?|R-So\.?|R-Jr\.?|R-Sr\.?|'
            r'First Year|Second Year|Third Year|Fourth Year)\b'
        )

        for table in soup.find_all('table'):
            if 'roster' not in str(table).lower():
                continue
            rows = table.find_all('tr')
            if not rows:
                continue
            headers_text = [th.get_text(strip=True).lower()
                            for th in rows[0].find_all(['th', 'td'])]

            for row in rows[1:]:
                cols = row.find_all('td')
                if len(cols) < 4:
                    continue
                a = {
                    'school_name': school_name, 'gender': gender,
                    'jersey_number': '', 'name': '', 'position': '',
                    'class_year': '', 'height': '', 'hometown': '',
                    'high_school_club': '',
                }
                for i, col in enumerate(cols):
                    text = col.get_text(strip=True)
                    if i < len(headers_text):
                        h = headers_text[i]
                        if 'no' in h or '#' in h or 'num' in h:
                            a['jersey_number'] = text
                        elif 'name' in h or 'player' in h:
                            a['name'] = text
                        elif 'pos' in h:
                            a['position'] = text
                        elif any(k in h for k in ['year', 'class', 'cl.', 'yr.']):
                            a['class_year'] = text
                        elif 'height' in h or 'ht' in h:
                            a['height'] = text
                        elif 'hometown' in h or 'city' in h:
                            if '/' in text:
                                left, right = text.split('/', 1)
                                a['hometown'] = left.strip()
                                a['high_school_club'] = right.strip()
                            else:
                                a['hometown'] = text
                        elif any(k in h for k in ['school', 'club', 'previous']):
                            a['high_school_club'] = text
                    if not a['class_year'] and re.search(cls_pattern, text, re.I):
                        a['class_year'] = text
                if a['name']:
                    roster_data.append(a)
    except Exception as e:
        log(f"   Sidearm error: {e}")
    return pd.DataFrame(roster_data)


def scrape_sidearm_cards(url: str, school_name: str, gender: str) -> pd.DataFrame:
    """Modern Sidearm layout: .s-person-card with label/value pairs."""
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
    roster_data = []
    try:
        response = requests.get(url, headers=REQ_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        cards = soup.select('.s-person-card')
        for card in cards:
            parts = [p.strip() for p in card.get_text('|', strip=True).split('|') if p.strip()]
            # Drop trailing/leading boilerplate
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
            # Walk label/value pairs
            i = 1
            while i < len(parts) - 1:
                label = parts[i].lower().rstrip(':')
                if label in LABEL_TO_FIELD:
                    a[LABEL_TO_FIELD[label]] = parts[i + 1]
                    i += 2
                else:
                    i += 1
            if a['name']:
                roster_data.append(a)
    except Exception as e:
        log(f"   Sidearm-cards error: {e}")
    return pd.DataFrame(roster_data)


def scrape_prestosports_roster(url: str, school_name: str, gender: str) -> pd.DataFrame:
    roster_data = []
    try:
        response = requests.get(url, headers=REQ_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # PrestoSports uses a list of player items with <li class="sidearm-roster-player">
        cards = soup.select('li.sidearm-roster-player') or soup.select('.sidearm-roster-player')
        for card in cards:
            a = {
                'school_name': school_name, 'gender': gender,
                'jersey_number': '', 'name': '', 'position': '',
                'class_year': '', 'height': '', 'hometown': '',
                'high_school_club': '',
            }
            num = card.select_one('.sidearm-roster-player-jersey-number')
            if num:
                a['jersey_number'] = num.get_text(strip=True)
            name = card.select_one('.sidearm-roster-player-name a, .sidearm-roster-player-name')
            if name:
                a['name'] = name.get_text(' ', strip=True)
            pos = card.select_one('.sidearm-roster-player-position-long-short, .sidearm-roster-player-position')
            if pos:
                a['position'] = pos.get_text(strip=True)
            year = card.select_one('.sidearm-roster-player-academic-year')
            if year:
                a['class_year'] = year.get_text(strip=True)
            hgt = card.select_one('.sidearm-roster-player-height')
            if hgt:
                a['height'] = hgt.get_text(strip=True)
            ht = card.select_one('.sidearm-roster-player-hometown')
            if ht:
                a['hometown'] = ht.get_text(strip=True)
            hs = card.select_one('.sidearm-roster-player-highschool, .sidearm-roster-player-previous-school')
            if hs:
                a['high_school_club'] = hs.get_text(strip=True)
            if a['name']:
                roster_data.append(a)
    except Exception as e:
        log(f"   Presto error: {e}")
    return pd.DataFrame(roster_data)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def infer_gender_from_url(url: str) -> str:
    if not url:
        return 'Mixed'
    u = url.lower()
    if 'womens' in u or '/w-' in u or 'wxc' in u:
        return 'Women'
    if 'mens' in u or '/m-' in u or 'mxc' in u:
        return 'Men'
    return 'Mixed'


def append_to_roster_csv(new_df: pd.DataFrame, filepath: str,
                          key_columns=('school_name', 'name', 'gender')) -> None:
    column_order = ['school_name', 'gender', 'jersey_number', 'name',
                    'position', 'class_year', 'height', 'hometown',
                    'high_school_club']
    new_df = new_df[[c for c in column_order if c in new_df.columns]]
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=list(key_columns), keep='last')
    else:
        combined = new_df
    combined.to_csv(filepath, index=False, encoding='utf-8')


def scrape_with_cascade(url: str, school: str, gender: str,
                         probe_status: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Try Sidearm first (most static rosters), then Presto. Skip WMT (Selenium).
    Returns (df, scraper_name) or (None, None) if all fail.
    """
    for name, fn in [('Sidearm-cards', scrape_sidearm_cards),
                     ('Sidearm-table', scrape_sidearm_roster),
                     ('Presto',        scrape_prestosports_roster)]:
        try:
            df = fn(url, school, gender)
            if df is not None and not df.empty:
                return df, name
        except Exception as e:
            log(f"   {name} raised: {e}")
    return None, None


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=None,
                   help='only process first N schools (for testing)')
    p.add_argument('--delay', type=float, default=DELAY,
                   help=f'seconds between requests (default {DELAY})')
    args = p.parse_args()

    if not os.path.exists(PLATFORMS_FILE):
        sys.exit(f"Missing {PLATFORMS_FILE}; run discover_track_urls.py first.")

    platforms_all = pd.read_csv(PLATFORMS_FILE)
    log(f"Loaded {len(platforms_all)} schools from {PLATFORMS_FILE}")

    # Resume from prior partial run.
    already = set()
    if os.path.exists(SUCCESS_FILE):
        already = set(pd.read_csv(SUCCESS_FILE)['school_name'].dropna().unique())
        log(f"Resuming: skipping {len(already)} already-scraped schools")

    # Defer to Colab/Selenium: js_only AND any school missing a probed roster URL.
    sel_mask = (platforms_all['probe_status'] == 'js_only') | platforms_all['roster_url'].isna()
    needs_sel = platforms_all[sel_mask]
    needs_sel.to_csv(SELENIUM_DEFERRED, index=False)
    log(f"Deferred to Colab/Selenium: {len(needs_sel)} schools (saved to {SELENIUM_DEFERRED})")

    # Process the rest: have a roster URL and aren't pure js_only.
    to_process = platforms_all[~sel_mask].copy()
    to_process = to_process[to_process['roster_url'].str.startswith('http', na=False)]
    to_process = to_process[~to_process['school_name'].isin(already)].reset_index(drop=True)
    if args.limit:
        to_process = to_process.head(args.limit)
    log(f"To process this run: {len(to_process)}")

    failures = []
    batch = []
    CHECKPOINT = 25

    for idx, row in to_process.iterrows():
        school = row['school_name']
        url    = row['roster_url']
        status = row['probe_status']
        gender = infer_gender_from_url(url)

        log(f"\n[{idx + 1}/{len(to_process)}] {school} ({status}) -> {url}")
        df, scraper = scrape_with_cascade(url, school, gender, status)

        if df is not None and not df.empty:
            n_ht = df['hometown'].astype(str).str.strip().replace('nan', '').ne('').sum()
            log(f"   {scraper}: {len(df)} athletes, {n_ht} with hometown")
            batch.append(df)
        else:
            log("   no data from any scraper")
            failures.append({'school_name': school, 'roster_url': url,
                             'probe_status': status, 'reason': 'no_data'})

        if (idx + 1) % CHECKPOINT == 0 and batch:
            combined = pd.concat(batch, ignore_index=True)
            append_to_roster_csv(combined, SUCCESS_FILE)
            log(f"   [CHECKPOINT @ {idx + 1}]")
            batch = []
            if failures:
                pd.DataFrame(failures).to_csv(FAILURE_FILE, index=False)

        time.sleep(args.delay)

    if batch:
        append_to_roster_csv(pd.concat(batch, ignore_index=True), SUCCESS_FILE)

    if failures:
        pd.DataFrame(failures).to_csv(FAILURE_FILE, index=False)
        log(f"\nSaved {len(failures)} failures to {FAILURE_FILE}")

    if os.path.exists(SUCCESS_FILE):
        final = pd.read_csv(SUCCESS_FILE)
        n_ht = final['hometown'].astype(str).str.strip().replace('nan', '').ne('').sum()
        log("\n=== Track scrape complete ===")
        log(f"  Schools covered: {final['school_name'].nunique()}")
        log(f"  Total athletes:  {len(final)}")
        log(f"  With hometown:   {n_ht} ({100 * n_ht / max(len(final), 1):.1f}%)")


if __name__ == '__main__':
    main()
