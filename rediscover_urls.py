"""
Selenium-driven URL re-discovery for schools where the initial probe
failed or returned a stale URL.

Strategy: visit the base athletic site, scan the rendered DOM for
anchor tags whose href contains "roster" and whose text/href hints at
track-and-field or cross-country. Score candidates and pick the best.
Validate by loading the candidate and checking for athlete cards
(.s-person-card, .roster-card-item, etc.).

Updates raw_data/school_track_platforms.csv in-place: schools whose
URLs were corrected get probe_status='OK_rediscovered' with the new
roster_url. Schools that still can't be resolved keep their existing
status (typically js_only with NaN roster_url).

Usage:
    python rediscover_urls.py [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import re
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_track_selenium import setup_driver

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
PLATFORMS_FILE = os.path.join(RAW_DATA, 'school_track_platforms.csv')
LOG_FILE = os.path.join(RAW_DATA, 'url_rediscovery.log')

# Score by sport-specific keywords found in the href. The link must
# contain "roster" plus at least one track/XC keyword in the path.
TRACK_KEYWORDS = [
    # priority order: most specific first
    'track-and-field', 'mens-track-and-field', 'womens-track-and-field',
    'track-field', 'xctrack', 'mens-xctrack', 'womens-xctrack',
    'track', 'm-track', 'w-track', 'mtrack', 'wtrack',
    'mens-cross-country', 'womens-cross-country', 'cross-country',
    'crosscountry', 'mxc', 'wxc', 'm-xc', 'w-xc', 'tf', 'xc',
]


def score_href(full_lc: str) -> int | None:
    """Lower score = better match. Returns None if not a track/XC roster URL."""
    if 'roster' not in full_lc:
        return None
    for i, kw in enumerate(TRACK_KEYWORDS):
        # require word boundary on at least one side to avoid spurious matches
        if re.search(rf'/(?:[a-z\-]*-)?{re.escape(kw)}(?:-[a-z\-]*)?/', full_lc) \
           or re.search(rf'/{re.escape(kw)}/', full_lc):
            return i
    return None


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as fh:
        fh.write(msg + '\n')


def collect_candidate_links(driver, base_url: str) -> list[tuple[str, int]]:
    """
    Load base_url, return list of (full_href, priority_score) for anchors
    that look like track/XC roster pages. Lower score = higher priority.
    """
    driver.get(base_url)
    time.sleep(3.5)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    candidates = []
    seen = set()
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if not href:
            continue
        full = urljoin(base_url, href)
        host = urlparse(full).netloc
        # Skip clearly external services; athletic department links may live on
        # a sibling subdomain (sundevils.com -> thesundevils.com).
        if any(x in host for x in ('facebook.', 'instagram.', 'twitter.', 'x.com',
                                   'youtube.', 'tiktok.', 'wikipedia.', 'spotify.')):
            continue
        full_lc = full.split('#')[0].lower()
        score = score_href(full_lc)
        if score is None:
            continue
        if full_lc in seen:
            continue
        seen.add(full_lc)
        candidates.append((full, score))

    candidates.sort(key=lambda x: x[1])  # best score first
    return candidates


def has_roster_data(driver, url: str) -> tuple[bool, int]:
    """Load url, return (looks_like_real_roster, athlete_card_count)."""
    try:
        driver.get(url)
        # Match the scraper's wait so a URL that validates here will also
        # parse during scraping. Nuxt/Vue rosters need 6+s to settle.
        time.sleep(7.0)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        for sel in ['.s-person-card', '.roster-card-item', '.roster-card',
                    '.roster__player', '.sidearm-roster-player', '.player-card',
                    'li.sidearm-roster-player']:
            cards = soup.select(sel)
            if len(cards) >= 5:
                return True, len(cards)
        # Tables with hometown column count too
        for table in soup.find_all('table'):
            if 'hometown' in str(table).lower():
                rows = [r for r in table.find_all('tr') if r.find('td')]
                if len(rows) >= 5:
                    return True, len(rows)
        return False, 0
    except Exception as e:
        log(f"   probe error: {e}")
        return False, 0


def rediscover_for_school(driver, school: str, base_url: str) -> tuple[str | None, int]:
    """Returns (roster_url, athlete_count) or (None, 0)."""
    log(f"\n{school} -> {base_url}")
    try:
        candidates = collect_candidate_links(driver, base_url)
    except Exception as e:
        log(f"   base load error: {e}")
        return None, 0
    log(f"   candidates: {len(candidates)}")
    for url, score in candidates[:5]:
        log(f"   trying [score={score}] {url}")
        ok, n = has_roster_data(driver, url)
        if ok:
            log(f"   ✓ roster found ({n} cards)")
            return url, n
    return None, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()

    df = pd.read_csv(PLATFORMS_FILE)
    done = set(pd.read_csv(os.path.join(RAW_DATA, 'track_rosters.csv'))['school_name'].unique())
    todo = df[~df['school_name'].isin(done)].copy()
    todo = todo[todo['base_url'].notna()].reset_index(drop=True)
    if args.limit:
        todo = todo.head(args.limit)
    log(f"Schools needing URL re-discovery: {len(todo)}")

    driver = setup_driver()
    log("Chrome for Testing driver started")

    updates = []  # list of (school, new_url, n_cards)
    try:
        for idx, row in todo.iterrows():
            school = row['school_name']
            base = row['base_url']
            log(f"\n[{idx + 1}/{len(todo)}]")
            new_url, n = rediscover_for_school(driver, school, base)
            if new_url:
                updates.append((school, new_url, n))
            time.sleep(1.0)
    finally:
        driver.quit()

    # Patch platforms CSV
    if updates:
        for school, new_url, n in updates:
            mask = df['school_name'] == school
            df.loc[mask, 'roster_url']   = new_url
            df.loc[mask, 'probe_status'] = 'OK_rediscovered'
            df.loc[mask, 'real_hometowns'] = n
        df.to_csv(PLATFORMS_FILE, index=False)
        log(f"\nUpdated {len(updates)} rows in {PLATFORMS_FILE}")

    # Rebuild deferred file
    deferred = df[~df['school_name'].isin(done)]
    deferred.to_csv(os.path.join(RAW_DATA, 'track_selenium_deferred.csv'), index=False)

    log("\n=== Re-discovery summary ===")
    log(f"  schools attempted: {len(todo)}")
    log(f"  URLs rediscovered: {len(updates)}")
    log(f"  remaining unresolved: {len(todo) - len(updates)}")


if __name__ == '__main__':
    main()
