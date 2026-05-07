"""
Discover athletic department track-roster URLs for all NCAA D-I schools.

Produces raw_data/school_track_platforms.csv, which the Track Roster Scraper
Orchestrator (Section 12 of student_athletes_pipeline.ipynb) consumes.

Pipeline:
  1. Scrape Wikipedia "List of NCAA Division I institutions" -> school +
     mascot + link to each program's Wikipedia article.
  2. For each program article, parse the infobox to extract the official
     athletic department URL (Website row).
  3. Probe a small set of known roster URL paths
     (/sports/track-and-field/roster, /sports/mens-track-and-field/roster, ...)
     and pick the first one that returns 200.
  4. Validate the candidate by counting "City, ST"-pattern occurrences in the
     rendered HTML; schools with low real_hometowns are flagged as
     js_table_stub (table renders client-side and needs Selenium).
  5. Match Wikipedia schools to TFRRS schools (tfrrs_all_rosters.csv) using
     a normalized-string + manual alias map.

Run from the repo root after a fresh data drop:
    python discover_track_urls.py

Cost: ~366 Wikipedia fetches + up to ~7 probes per school = ~3-5 min.
"""
from __future__ import annotations

import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')

HEADERS = {
    'User-Agent': 'StudentAthletesResearch/1.0 (alina@malkova.net)',
    'Accept': 'text/html',
}

ROSTER_PATHS = [
    '/sports/track-and-field/roster',
    '/sports/mens-track-and-field/roster',
    '/sports/m-track/roster',
    '/sports/track-field/roster',
    '/sports/m-xc/roster',
    '/sports/cross-country/roster',
    '/sports/mens-cross-country/roster',
]

CITY_STATE_RE = re.compile(
    r'>[^<>]*[A-Z][a-zA-Z\.\- ]{2,30},\s*(?:[A-Z]{2}\.?|[A-Z][a-z]+\.?)[^<>]*<'
)

# TFRRS school name -> normalized Wikipedia school name (after normalize()).
ALIASES = {
    'army west point': 'army army west point',
    'csu bakersfield': 'bakersfield',
    'citadel': 'the citadel',
    'col of charleston': 'charleston',
    'connecticut': 'uconn',
    'grambling': 'grambling state',
    'iu indianapolis': 'iu indy',
    'illinois chicago': 'uic',
    'jacksonville state': 'jax state',
    'loyola ill': 'loyola chicago',
    'loyola md': 'loyola',
    'mcneese state': 'mcneese',
    'miami fla': 'miami florida',
    'mid tenn state': 'middle tennessee',
    'miss state': 'mississippi state',
    'n c central': 'north carolina central',
    'nicholls state': 'nicholls',
    'queens n c': 'queens nc',
    'sc state': 'south carolina state',
    'se louisiana': 'southeastern louisiana',
    'se missouri': 'semo',
    'siu edwardsville': 'siue',
    'ualbany': 'albany',
    'umass amherst': 'umass massachusetts',
    'ut rio grande valley': 'utrgv',
    'xavier ohio': 'xavier',
    'app state': 'appalachian state',
    'east tenn state': 'east tennessee state',
    'mississippi valley': 'mississippi valley state',
    'csun': 'cal state northridge csun',
    'fdu': 'fairleigh dickinson',
    'state mary s cal': 'saint mary s',
    'texas aandm cc': 'texas aandm corpus christi',
    'state peter s': 'saint peter s',
    'n carolina aandt': 'north carolina aandt',
    'prairie view': 'prairie view aandm',
    'saint francis': 'saint francis pa',
    'state thomas minn': 'state thomas',
}


def normalize(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'\[[^\]]*\]', '', s)
    s = re.sub(r'\(([a-z\.]+)\)', r'\1', s)
    s = s.replace('ʻ', '').replace('–', '-').replace('—', '-')
    s = re.sub(r'\bst\.?\b', 'state', s)
    s = re.sub(r'\bu\.?\b', '', s)
    s = re.sub(r'\buniversity\b', '', s)
    s = re.sub(r'\bcollege\b', '', s)
    s = re.sub(r'&', 'and', s)
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def fetch_wikipedia_dI_list() -> pd.DataFrame:
    """Scrape the D-I list page for (school, mascot, article_url) triples."""
    r = requests.get(
        'https://en.wikipedia.org/wiki/List_of_NCAA_Division_I_institutions',
        headers=HEADERS, timeout=20,
    )
    soup = BeautifulSoup(r.text, 'html.parser')
    tables = soup.find_all('table', class_=re.compile(r'wikitable'))
    rows = []
    # First three wikitables = main + reclassifying schools
    for tbl in tables[:3]:
        for tr in tbl.find_all('tr')[1:]:
            cells = tr.find_all(['td', 'th'])
            if len(cells) < 3:
                continue
            common = cells[1].get_text(' ', strip=True)
            link = cells[2].find('a', href=True)
            if not (link and link['href'].startswith('/wiki/')):
                continue
            rows.append({
                'wiki_school': common,
                'mascot': link.get_text(strip=True),
                'article_url': 'https://en.wikipedia.org' + link['href'],
            })
    return pd.DataFrame(rows)


def extract_athletic_website(article_url: str) -> str | None:
    try:
        r = requests.get(article_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        infobox = soup.find('table', class_=re.compile(r'infobox'))
        if not infobox:
            return None
        for row in infobox.find_all('tr'):
            th = row.find('th')
            if th and 'website' in th.get_text(strip=True).lower():
                a = row.find('a', class_='external', href=True) or row.find('a', href=True)
                if a:
                    href = a['href']
                    return 'https:' + href if href.startswith('//') else href
        # fallback: first non-Wikipedia external link in infobox
        for a in infobox.find_all('a', class_='external', href=True):
            href = a['href']
            if 'wikipedia' not in href and 'ncaa' not in href.lower():
                return 'https:' + href if href.startswith('//') else href
    except Exception:
        pass
    return None


def probe_track(base_url: str | None) -> tuple[str | None, str, int]:
    """Return (roster_url, status, raw_hometown_count)."""
    if not base_url:
        return None, 'no_base', 0
    base = base_url.rstrip('/')
    seen_200 = None
    for p in ROSTER_PATHS:
        url = base + p
        try:
            r = requests.get(url, headers=HEADERS, timeout=8, allow_redirects=True)
            if r.status_code != 200:
                continue
            n = r.text.lower().count('hometown')
            if n >= 5:
                return url, 'OK_static', n
            if seen_200 is None and len(r.text) > 50_000:
                seen_200 = url
        except Exception:
            continue
    return seen_200, 'js_only', 0


def real_hometowns(html: str) -> int:
    """Count occurrences of City, ST-style strings between HTML tags."""
    return len(CITY_STATE_RE.findall(html))


def revalidate(roster_url: str | None) -> int:
    if not roster_url:
        return 0
    try:
        r = requests.get(roster_url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return 0
        return real_hometowns(r.text)
    except Exception:
        return -1


def match_tfrrs_to_wiki(tfrrs_schools: list[str], wiki_norms: list[str],
                       norm_to_wiki: dict) -> pd.DataFrame:
    rows = []
    for t in tfrrs_schools:
        nt = normalize(t)
        target = ALIASES.get(nt, nt)
        if target in norm_to_wiki:
            mtype = 'alias' if nt != target else 'exact'
            rows.append({'tfrrs_school': t, 'wiki_school': norm_to_wiki[target],
                         'match_type': mtype, 'score': 100})
            continue
        # fuzzy backup
        best, best_n = 0, None
        nt_tokens = set(target.split())
        for n in wiki_norms:
            w_tokens = set(n.split())
            if not w_tokens:
                continue
            jacc = len(nt_tokens & w_tokens) / len(nt_tokens | w_tokens)
            seq = SequenceMatcher(None, target, n).ratio()
            score = 0.6 * jacc + 0.4 * seq
            if score > best:
                best, best_n = score, n
        if best >= 0.65:
            rows.append({'tfrrs_school': t, 'wiki_school': norm_to_wiki[best_n],
                         'match_type': 'fuzzy', 'score': round(best * 100, 1)})
        else:
            rows.append({'tfrrs_school': t, 'wiki_school': None,
                         'match_type': 'no_match', 'score': round(best * 100, 1)})
    return pd.DataFrame(rows)


def main():
    print("Step 1: scraping Wikipedia D-I institutions list...")
    wiki = fetch_wikipedia_dI_list()
    print(f"  {len(wiki)} programs found")

    print("\nStep 2: extracting athletic website + probing track URL (parallel)...")

    def process(row):
        site = extract_athletic_website(row['article_url'])
        roster_url, status, n_raw = probe_track(site)
        return {
            'wiki_school': row['wiki_school'],
            'mascot': row['mascot'],
            'base_url': site,
            'roster_url': roster_url,
            'status': status,
            'hometown_count_raw': n_raw,
        }

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(process, r) for _, r in wiki.iterrows()]
        results = [f.result() for f in as_completed(futures)]

    wiki_results = pd.DataFrame(results).sort_values('wiki_school').reset_index(drop=True)
    print(wiki_results['status'].value_counts().to_string())

    print("\nStep 3: re-validating with stricter City/ST pattern...")
    wiki_results['real_hometowns'] = [revalidate(u) for u in wiki_results['roster_url']]
    mask = (wiki_results['status'] == 'OK_static') & (wiki_results['real_hometowns'] < 30)
    wiki_results.loc[mask, 'status'] = 'js_table_stub'

    print("\nStep 4: matching to TFRRS school list...")
    tfrrs_path = os.path.join(RAW_DATA, 'tfrrs_all_rosters.csv')
    tfrrs_schools = sorted(pd.read_csv(tfrrs_path)['school_name'].dropna().unique())

    wiki_results['norm'] = wiki_results['wiki_school'].apply(normalize)
    norm_to_wiki = dict(zip(wiki_results['norm'], wiki_results['wiki_school']))
    matches = match_tfrrs_to_wiki(tfrrs_schools, wiki_results['norm'].tolist(), norm_to_wiki)
    print(matches['match_type'].value_counts().to_string())

    final = matches.merge(wiki_results.drop(columns='norm'),
                          on='wiki_school', how='left')
    final = final.rename(columns={
        'tfrrs_school': 'school_name',
        'wiki_school': 'wiki_match',
        'score': 'match_score',
        'status': 'probe_status',
        'hometown_count_raw': 'hometown_count',
    })
    final = final[['school_name', 'wiki_match', 'match_type', 'match_score',
                   'mascot', 'base_url', 'roster_url', 'probe_status',
                   'hometown_count', 'real_hometowns']]

    out = os.path.join(RAW_DATA, 'school_track_platforms.csv')
    final.to_csv(out, index=False)
    print(f"\nWrote {len(final)} rows to {out}")
    print()
    print("Final probe_status:")
    print(final['probe_status'].value_counts().to_string())


if __name__ == '__main__':
    main()
