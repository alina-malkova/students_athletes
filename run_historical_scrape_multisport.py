"""
Historical roster scraper for sports other than track and field.

Strategy: take the school_track_platforms.csv list (which already has
working base URLs and verified archive templates), substitute the
sport segment, and run the same template discovery + per-year scrape
for each requested sport.

Per-sport URL conventions on Sidearm / WMT sites:
  Soccer:   /sports/mens-soccer/roster        /sports/womens-soccer/roster
  Tennis:   /sports/mens-tennis/roster        /sports/womens-tennis/roster
  Golf:     /sports/mens-golf/roster          /sports/womens-golf/roster
  Basketball: /sports/mens-basketball/roster  /sports/womens-basketball/roster
  Swimming: /sports/mens-swimming-and-diving/roster (etc.)

Output: raw_data/{sport}_rosters_panel.csv, one file per sport, with
columns school_name, sport, gender, year, name, hometown, ...

Usage:
    python run_historical_scrape_multisport.py soccer
    python run_historical_scrape_multisport.py tennis golf
"""
from __future__ import annotations

import os
import re
import sys
import time
import warnings
import argparse
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_historical_scrape import (REQ_HEADERS, DELAY, YEARS_HISTORICAL,
                                     TEMPLATES, fmt_template,
                                     count_hometowns_in_html,
                                     cascade_parse_html, _extract_names,
                                     log as base_log)

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')

# Sport configurations: name -> list of (gender, URL segment)
SPORT_SEGMENTS = {
    'soccer':     [('Men',   'mens-soccer'),     ('Women', 'womens-soccer')],
    'tennis':     [('Men',   'mens-tennis'),     ('Women', 'womens-tennis')],
    'golf':       [('Men',   'mens-golf'),       ('Women', 'womens-golf')],
    'basketball': [('Men',   'mens-basketball'), ('Women', 'womens-basketball')],
    'swimming':   [('Men',   'mens-swimming-and-diving'),
                   ('Women', 'womens-swimming-and-diving')],
    'volleyball': [('Women', 'womens-volleyball'),
                   ('Men',   'mens-volleyball')],
    'crosscountry': [('Men',   'mens-cross-country'),
                     ('Women', 'womens-cross-country')],
    'icehockey':  [('Men',   'mens-ice-hockey'), ('Women', 'womens-ice-hockey')],
    'baseball':   [('Men',   'baseball')],
    'softball':   [('Women', 'softball')],
    'lacrosse':   [('Men',   'mens-lacrosse'),   ('Women', 'womens-lacrosse')],
}


def substitute_sport_url(base_url: str, sport_segment: str) -> str | None:
    """
    Replace the track segment in a base roster URL with the new sport.

    Examples:
      /sports/track-and-field/roster -> /sports/mens-soccer/roster
      /sports/xctrack/roster         -> /sports/mens-soccer/roster
      /sport/m-track/roster          -> /sport/mens-soccer/roster
    """
    # Try several regex substitutions
    patterns = [
        # Sidearm-modern: /sports/track-and-field/roster
        (r'/sports/[a-z\-]+(?=/roster)', f'/sports/{sport_segment}'),
        (r'/sport/[a-z\-]+(?=/roster)',  f'/sport/{sport_segment}'),
    ]
    for pat, repl in patterns:
        new_url = re.sub(pat, repl, base_url, count=1)
        if new_url != base_url:
            return new_url
    # Fallback: replace the last segment before /roster
    # e.g. /track/roster -> /mens-soccer/roster
    m = re.search(r'/([a-z\-]+)/roster', base_url)
    if m:
        return base_url.replace(m.group(0), f'/{sport_segment}/roster')
    return None


def discover_template_for_sport(roster_url: str) -> str | None:
    """Same as run_historical_scrape.discover_template but inlined to use
    the substituted URL."""
    try:
        cur = requests.get(roster_url, headers=REQ_HEADERS, timeout=15,
                            allow_redirects=True)
        if cur.status_code != 200:
            return None
        current_names = _extract_names(cur.text)
    except Exception:
        return None
    if not current_names:
        return None

    best = (0, None)
    for tmpl in TEMPLATES:
        u = fmt_template(tmpl, roster_url, 2022)
        try:
            r = requests.get(u, headers=REQ_HEADERS, timeout=15, allow_redirects=True)
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
            if overlap > 0.70:
                continue
            score = n - 200 * overlap
            if score > best[0]:
                best = (score, tmpl)
        except Exception:
            continue
    return best[1]


def scrape_sport_panel(sport: str, schools: pd.DataFrame, out_path: str,
                        log_path: str):
    """For a given sport, iterate over schools, substitute URL, discover
    template, scrape 5 years."""
    segments = SPORT_SEGMENTS[sport]

    # Resume support
    done_pairs = set()
    if os.path.exists(out_path):
        prior = pd.read_csv(out_path)
        for _, r in prior[['school_name', 'gender', 'year']].drop_duplicates().iterrows():
            done_pairs.add((r['school_name'], r['gender'], int(r['year'])))
        print(f'Resuming: {len(done_pairs)} (school, gender, year) cells already done')

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as fh:
            fh.write(msg + '\n')

    buffer = []
    failures = []

    for idx, row in schools.reset_index(drop=True).iterrows():
        school = row['school_name']
        base_url = row['roster_url']
        if pd.isna(base_url):
            continue

        for gender, segment in segments:
            new_url = substitute_sport_url(base_url, segment)
            if not new_url:
                continue
            all_done = all((school, gender, y) in done_pairs for y in YEARS_HISTORICAL)
            if all_done:
                continue

            log(f'\n[{idx+1}/{len(schools)}] {school} ({gender})')
            log(f'  url: {new_url}')

            tmpl = discover_template_for_sport(new_url)
            if tmpl is None:
                log('  no working archive template')
                failures.append({'school_name': school, 'sport': sport,
                                  'gender': gender, 'reason': 'no_template'})
                continue
            log(f'  template: {tmpl}')

            for year in YEARS_HISTORICAL:
                if (school, gender, year) in done_pairs:
                    continue
                archive_url = fmt_template(tmpl, new_url, year)
                try:
                    r = requests.get(archive_url, headers=REQ_HEADERS, timeout=20)
                    if r.status_code != 200:
                        continue
                    df_yr = cascade_parse_html(r.text, school, gender)
                    if df_yr.empty:
                        continue
                    df_yr['sport'] = sport
                    df_yr['year'] = year
                    df_yr['archive_url'] = archive_url
                    buffer.append(df_yr)
                    log(f'  {year}: {len(df_yr)} athletes')
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
                subset=['school_name', 'sport', 'gender', 'year', 'name'],
                keep='last')
            df_buf.to_csv(out_path, index=False)
            log(f'  [CHECKPOINT @ {idx+1}]  total {sport} panel rows: {len(df_buf)}')
            buffer = []

    if buffer:
        df_buf = pd.concat(buffer, ignore_index=True)
        if os.path.exists(out_path):
            old = pd.read_csv(out_path)
            df_buf = pd.concat([old, df_buf], ignore_index=True)
        df_buf = df_buf.drop_duplicates(
            subset=['school_name', 'sport', 'gender', 'year', 'name'],
            keep='last')
        df_buf.to_csv(out_path, index=False)
        log(f'\nFinal {sport}: {len(df_buf)} panel rows in {out_path}')

    if failures:
        fpath = os.path.join(RAW_DATA, f'{sport}_failures.csv')
        pd.DataFrame(failures).to_csv(fpath, index=False)
        log(f'  {len(failures)} failed (school, gender) cells in {fpath}')


def scrape_sport_panel_from_full(sport: str, platforms_full: pd.DataFrame,
                                   out_path: str, log_path: str):
    """Variant that uses the student's school_website_platforms_full.csv,
    which has a column like `{sport}_roster` with the per-school URL.
    Iterates over the platforms_full rows (one per school×gender)."""
    col = SPORT_TO_COLUMN[sport]

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as fh:
            fh.write(msg + '\n')

    sub = platforms_full[platforms_full[col].notna()].copy()
    log(f'Sport {sport}: {len(sub)} (school, gender) rows have a URL in {col}')

    done_pairs = set()
    if os.path.exists(out_path):
        prior = pd.read_csv(out_path)
        for _, r in prior[['school_name', 'gender', 'year']].drop_duplicates().iterrows():
            done_pairs.add((r['school_name'], r['gender'], int(r['year'])))
        log(f'Resuming: {len(done_pairs)} (school, gender, year) cells already done')

    buffer = []
    failures = []

    for idx, row in sub.reset_index(drop=True).iterrows():
        school = row['school_name']
        gender = row['gender']
        url    = row[col]

        all_done = all((school, gender, y) in done_pairs for y in YEARS_HISTORICAL)
        if all_done:
            continue

        log(f'\n[{idx+1}/{len(sub)}] {school} ({gender})')
        log(f'  url: {url}')

        tmpl = discover_template_for_sport(url)
        if tmpl is None:
            log('  no working archive template')
            failures.append({'school_name': school, 'sport': sport,
                              'gender': gender, 'url': url,
                              'reason': 'no_template'})
            continue
        log(f'  template: {tmpl}')

        for year in YEARS_HISTORICAL:
            if (school, gender, year) in done_pairs:
                continue
            archive_url = fmt_template(tmpl, url, year)
            try:
                r = requests.get(archive_url, headers=REQ_HEADERS, timeout=20)
                if r.status_code != 200:
                    continue
                df_yr = cascade_parse_html(r.text, school, gender)
                if df_yr.empty:
                    continue
                df_yr['sport'] = sport
                df_yr['year']  = year
                df_yr['archive_url'] = archive_url
                buffer.append(df_yr)
                log(f'  {year}: {len(df_yr)} athletes')
            except Exception as e:
                log(f'  {year}: exception {e}')
            time.sleep(DELAY)

        # Checkpoint every 50 rows
        if buffer and (idx + 1) % 50 == 0:
            df_buf = pd.concat(buffer, ignore_index=True)
            if os.path.exists(out_path):
                old = pd.read_csv(out_path)
                df_buf = pd.concat([old, df_buf], ignore_index=True)
            df_buf = df_buf.drop_duplicates(
                subset=['school_name', 'sport', 'gender', 'year', 'name'],
                keep='last')
            df_buf.to_csv(out_path, index=False)
            log(f'  [CHECKPOINT @ {idx+1}]  total {sport} rows: {len(df_buf)}')
            buffer = []

    if buffer:
        df_buf = pd.concat(buffer, ignore_index=True)
        if os.path.exists(out_path):
            old = pd.read_csv(out_path)
            df_buf = pd.concat([old, df_buf], ignore_index=True)
        df_buf = df_buf.drop_duplicates(
            subset=['school_name', 'sport', 'gender', 'year', 'name'],
            keep='last')
        df_buf.to_csv(out_path, index=False)
        log(f'\nFinal {sport}: {len(df_buf)} panel rows -> {out_path}')

    if failures:
        fpath = os.path.join(RAW_DATA, f'{sport}_failures.csv')
        pd.DataFrame(failures).to_csv(fpath, index=False)
        log(f'  {len(failures)} failures -> {fpath}')


# Sport key -> column name in school_website_platforms_full.csv
SPORT_TO_COLUMN = {
    'soccer':       'soccer_roster',
    'tennis':       'tennis_roster',
    'golf':         'golf_roster',
    'basketball':   'basketball_roster',
    'swimming':     'swimming_diving_roster',
    'volleyball':   'volleyball_roster',
    'crosscountry': 'cross_country_roster',
    'icehockey':    'ice_hockey_roster',
    'baseball':     'baseball_roster',
    'softball':     'softball_roster',
    'lacrosse':     'lacrosse_roster',
    'football':     'football_roster',
    'wrestling':    'wrestling_roster',
    'rowing':       'rowing_roster',
    'fieldhockey':  'field_hockey_roster',
    'waterpolo':    'water_polo_roster',
    'gymnastics':   'gymnastics_roster',
    'fencing':      'fencing_roster',
    'bowling':      'bowling_roster',
    'beachvolleyball': 'beach_volleyball_roster',
    'trackfield':   'track_field_roster',
}


def main():
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('sports', nargs='+', help='sport keys')
    parser.add_argument('--limit', type=int, default=None,
                        help='for piloting, limit to first N rows')
    parser.add_argument('--divisions', nargs='+', default=None,
                        help='restrict to specific divisions (e.g. I I-FBS I-FCS)')
    args = parser.parse_args()

    # Prefer the student's full per-sport URL table if available.
    full_path = os.path.join(RAW_DATA, 'school_website_platforms_full.csv')
    if not os.path.exists(full_path):
        print(f'Missing {full_path}; run with the student-provided platforms file.')
        return
    platforms_full = pd.read_csv(full_path, low_memory=False)
    if args.divisions:
        platforms_full = platforms_full[platforms_full['division'].isin(args.divisions)]
    if args.limit:
        platforms_full = platforms_full.head(args.limit)
    print(f'Loaded school_website_platforms_full.csv: {len(platforms_full)} rows')

    for sport in args.sports:
        if sport not in SPORT_TO_COLUMN:
            print(f'Unknown sport: {sport}; choices: {sorted(SPORT_TO_COLUMN)}')
            continue
        print(f'\n{"=" * 60}\nSPORT: {sport}\n{"=" * 60}')
        out_path = os.path.join(RAW_DATA, f'{sport}_rosters_panel.csv')
        log_path = os.path.join(RAW_DATA, f'{sport}_scrape.log')
        if os.path.exists(log_path):
            os.remove(log_path)
        scrape_sport_panel_from_full(sport, platforms_full, out_path, log_path)


if __name__ == '__main__':
    main()
