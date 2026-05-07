"""
Local runner for Section 14 of the pipeline (merge analysis dataset).

Joins roster data (soccer + track) -> NCAA-IPEDS crosswalk -> IPEDS HD ->
EADA athletics financials -> WDI macro indicators. Output:
raw_data/analysis_dataset.csv.

Track supersedes TFRRS when present (school-site rosters carry hometown,
TFRRS profile pages don't).

Usage:
    python run_merge_analysis.py
"""
from __future__ import annotations

import os
import re
import string
import pandas as pd

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')


def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())


# Aggressive normalization for school-name matching across roster/crosswalk.
# Strips university/college markers, expands "St." to "state", normalizes "&" -> "and".
def aggressive_norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'\[[^\]]*\]', '', s)
    s = re.sub(r'\(([a-z\.]+)\)', r'\1', s)
    s = s.replace('ʻ', '').replace('–', '-').replace('—', '-')
    s = re.sub(r'\bst\.?\b', 'state', s)
    s = re.sub(r'\bu\.?\b', '', s)
    s = re.sub(r'\buniversity\b', '', s)
    s = re.sub(r'\bcollege\b', '', s)
    s = re.sub(r'\bof\b', '', s)
    s = re.sub(r'\bthe\b', '', s)
    s = re.sub(r'&', 'and', s)
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# Manual aliases for roster school names that don't fuzzy-match cleanly.
ROSTER_TO_CROSSWALK_ALIASES = {
    'army west point': 'army',
    'csu bakersfield': 'california state bakersfield',
    'cal state bakersfield': 'california state bakersfield',
    'cal state fullerton': 'california state fullerton',
    'csun': 'california state northridge',
    'cal poly': 'california polytechnic state san luis obispo',
    'col of charleston': 'charleston',
    'connecticut': 'connecticut',
    'iu indianapolis': 'indiana indianapolis',
    'illinois chicago': 'illinois chicago',
    'jacksonville state': 'jacksonville state',
    'loyola ill': 'loyola chicago',
    'loyola md': 'loyola maryland',
    'mcneese state': 'mcneese',
    'miami fla': 'miami florida',
    'mid tenn state': 'middle tennessee state',
    'miss state': 'mississippi state',
    'n c central': 'north carolina central',
    'queens n c': 'queens charlotte',
    'sc state': 'south carolina state',
    'se louisiana': 'southeastern louisiana',
    'se missouri': 'southeast missouri state',
    'siu edwardsville': 'southern illinois edwardsville',
    'ut rio grande valley': 'texas rio grande valley',
    'xavier ohio': 'xavier',
    'app state': 'appalachian state',
    'east tenn state': 'east tennessee state',
    'fdu': 'fairleigh dickinson',
    'state mary state cal': 'saint marys california',
    'texas aandm cc': 'texas aandm corpus christi',
    'state peter s': 'saint peters',
    'umass amherst': 'massachusetts amherst',
    # Common acronyms / short names
    'byu': 'brigham young',
    'lsu': 'louisiana state',
    'smu': 'southern methodist',
    'tcu': 'texas christian',
    'unlv': 'nevada las vegas',
    'ucla': 'california los angeles',
    'usc': 'southern california',
    'ucf': 'central florida',
    'utep': 'texas el paso',
    'utsa': 'texas san antonio',
    'utrgv': 'texas rio grande valley',
    'fiu': 'florida international',
    'fau': 'florida atlantic',
    'ulm': 'louisiana monroe',
    'liu': 'long island',
    'nc state': 'north carolina state',
    'navy': 'naval academy',
    'ole miss': 'mississippi',
    'penn': 'pennsylvania',
    'pitt': 'pittsburgh',
    'njit': 'new jersey institute of technology',
    'uab': 'alabama birmingham',
    'umbc': 'maryland baltimore county',
    'uc davis': 'california davis',
    'uc irvine': 'california irvine',
    'uc riverside': 'california riverside',
    'uc san diego': 'california san diego',
    'uc santa barbara': 'california santa barbara',
    'unc asheville': 'north carolina asheville',
    'unc greensboro': 'north carolina greensboro',
    'uncw': 'north carolina wilmington',
    'usc upstate': 'south carolina upstate',
    'ut arlington': 'texas arlington',
    'ut martin': 'tennessee martin',
    'vmi': 'virginia military institute',
    'vcu': 'virginia commonwealth',
    'texas aandm': 'texas aandm station',
    'texas aandm cc': 'texas aandm corpus christi',
    'miami ohio': 'miami university ohio',
    'miami fla': 'miami florida',
    'southern miss': 'southern mississippi',
    'army west point': 'army',
    'state mary state cal': 'saint marys california',
    'col of charleston': 'charleston',
    'usc upstate': 'south carolina upstate',
    'wisc green bay': 'wisconsin green bay',
    'wisc milwaukee': 'wisconsin milwaukee',
    'umass lowell': 'massachusetts lowell',
    'ualbany': 'albany',
    'ohio state': 'ohio state',
    'state thomas minn': 'state thomas',
    'omaha': 'nebraska omaha',
    'grambling': 'grambling state',
    'mississippi valley': 'mississippi valley state',
    'n carolina aandt': 'north carolina aandt state',
    'prairie view': 'prairie view aandm',
    'saint francis': 'saint francis pa',
    'east texas aandm': 'east texas aandm',
    'tarleton state': 'tarleton state',
    'jax state': 'jacksonville state',
    'central connecticut': 'central connecticut state',
    'detroit mercy': 'detroit mercy',
    'green bay': 'wisconsin green bay',
    'milwaukee': 'wisconsin milwaukee',
    'kennesaw state': 'kennesaw state',
    'incarnate word': 'incarnate word',
    'cleveland state': 'cleveland state',
    'long beach state': 'long beach state',
    'miami ohio': 'miami university',
    'north dakota state': 'north dakota state',
    'south dakota state': 'south dakota state',
    'north dakota': 'north dakota',
    'south dakota': 'south dakota',
}


def build_school_to_unitid(roster_schools, crosswalk):
    """For each unique roster school name, find best matching UNITID in crosswalk."""
    from thefuzz import process, fuzz
    # Normalize from the original ncaa_name; cleaned_name has known artifacts
    # (e.g. "universitycorpus" with no space).
    cw_norm = crosswalk.assign(_norm=crosswalk['ncaa_name'].apply(aggressive_norm))
    cw_norm_list = cw_norm['_norm'].tolist()

    out = {}
    for s in roster_schools:
        nt = aggressive_norm(s)
        target = ROSTER_TO_CROSSWALK_ALIASES.get(nt, nt)
        # Exact normalized match first
        hits = cw_norm[cw_norm['_norm'] == target]
        if len(hits) == 1:
            out[s] = (int(hits.iloc[0]['ipeds_unitid']) if pd.notna(hits.iloc[0]['ipeds_unitid']) else None,
                      'exact', 100)
            continue
        # Fuzzy on normalized strings
        result = process.extractOne(target, cw_norm_list, scorer=fuzz.token_set_ratio)
        if result and result[1] >= 90:
            row = cw_norm.iloc[cw_norm_list.index(result[0])]
            out[s] = (int(row['ipeds_unitid']) if pd.notna(row['ipeds_unitid']) else None,
                      'fuzzy', int(result[1]))
        else:
            out[s] = (None, 'no_match', int(result[1]) if result else 0)
    return out


def normalize_roster(df: pd.DataFrame, default_sport: str, source_label: str) -> pd.DataFrame:
    if 'name' in df.columns and 'athlete_name' not in df.columns:
        df = df.rename(columns={'name': 'athlete_name'})
    if 'sport' not in df.columns:
        df['sport'] = default_sport
    df['source'] = source_label
    return df


def main():
    # 1. Load rosters: track supersedes TFRRS
    roster_dfs = []

    soccer_path = os.path.join(RAW_DATA, 'soccer_rosters.csv')
    if os.path.exists(soccer_path):
        soccer = normalize_roster(pd.read_csv(soccer_path), 'Soccer', 'soccer_rosters')
        roster_dfs.append(soccer)
        print(f"Soccer roster: {len(soccer)} athletes")

    track_path = os.path.join(RAW_DATA, 'track_rosters.csv')
    tfrrs_path = os.path.join(RAW_DATA, 'tfrrs_all_rosters.csv')
    if os.path.exists(track_path):
        track = normalize_roster(pd.read_csv(track_path), 'Track & Field', 'track_rosters')
        roster_dfs.append(track)
        print(f"Track roster: {len(track)} athletes (school-site scrape)")
        if os.path.exists(tfrrs_path):
            print('  Skipping tfrrs_all_rosters.csv -- superseded by track_rosters.csv')
    elif os.path.exists(tfrrs_path):
        tfrrs = normalize_roster(pd.read_csv(tfrrs_path), 'Track & Field', 'tfrrs')
        roster_dfs.append(tfrrs)
        print(f"TFRRS roster: {len(tfrrs)} athletes (no hometown column)")

    if not roster_dfs:
        raise SystemExit('No roster data found.')

    all_rosters = pd.concat(roster_dfs, ignore_index=True)
    print(f"\nCombined: {len(all_rosters)} athletes")
    print(f"  by sport: {all_rosters['sport'].value_counts().to_dict()}")
    print(f"  with hometown: {all_rosters['hometown'].notna().sum()}")
    print(f"  with country_code: {all_rosters['country_code'].notna().sum() if 'country_code' in all_rosters.columns else 0}")

    # 2. Crosswalk: school_name -> UNITID. The project is about D-I athletes,
    # so restrict the lookup to D-I crosswalk entries to avoid fuzzy-matching
    # "California" to a D-II school of the same name.
    cw_full = pd.read_csv(os.path.join(RAW_DATA, 'ncaa_ipeds_crosswalk_verified.csv'))
    cw = cw_full[cw_full['ncaa_division'] == 1].copy()
    print(f"\nCrosswalk: {len(cw_full)} total entries, {len(cw)} D-I entries (used for matching)")

    # Build school -> (UNITID, match_type, score) using fuzzy matching with
    # aggressive normalization + manual aliases.
    print("\nBuilding school -> UNITID lookup...")
    schools_unique = sorted(all_rosters['school_name'].dropna().unique())
    s2u = build_school_to_unitid(schools_unique, cw)
    matched = sum(1 for v in s2u.values() if v[0] is not None)
    print(f"  Matched {matched}/{len(s2u)} unique schools to UNITID")
    no_match = [s for s, v in s2u.items() if v[0] is None]
    if no_match:
        print(f"  Schools without UNITID match ({len(no_match)}): "
              + ', '.join(no_match[:20])
              + (' ...' if len(no_match) > 20 else ''))

    all_rosters['ipeds_unitid'] = all_rosters['school_name'].map(lambda s: s2u.get(s, (None,))[0])
    all_rosters['unitid_match_type'] = all_rosters['school_name'].map(lambda s: s2u.get(s, (None, 'missing', 0))[1])
    # Bring in NCAA division/conference from crosswalk via UNITID
    cw_dedup = cw.dropna(subset=['ipeds_unitid']).drop_duplicates(subset='ipeds_unitid')
    cw_lookup = cw_dedup.set_index('ipeds_unitid')[['ncaa_division', 'ncaa_conference']].to_dict('index')
    all_rosters['ncaa_division']   = all_rosters['ipeds_unitid'].map(lambda u: cw_lookup.get(u, {}).get('ncaa_division'))
    all_rosters['ncaa_conference'] = all_rosters['ipeds_unitid'].map(lambda u: cw_lookup.get(u, {}).get('ncaa_conference'))
    merged = all_rosters
    print(f"After crosswalk join: {len(merged)} rows, "
          f"{merged['ipeds_unitid'].notna().sum()} with UNITID "
          f"({100 * merged['ipeds_unitid'].notna().sum() / len(merged):.1f}%)")

    # 3. IPEDS HD
    hd = pd.read_csv(os.path.join(RAW_DATA, 'HD2023.csv'), encoding='latin-1')
    # HD2023.csv has BOM in header
    hd.columns = [c.lstrip('﻿') for c in hd.columns]
    if 'UNITID' not in hd.columns:
        hd = hd.rename(columns={hd.columns[0]: 'UNITID'})
    hd_cols = ['UNITID', 'INSTNM', 'CITY', 'STABBR', 'SECTOR', 'CONTROL',
               'HLOFFER', 'OBEREG', 'LOCALE', 'INSTSIZE']
    hd_subset = hd[[c for c in hd_cols if c in hd.columns]].copy()
    hd_subset['UNITID'] = pd.to_numeric(hd_subset['UNITID'], errors='coerce')
    merged['ipeds_unitid'] = pd.to_numeric(merged['ipeds_unitid'], errors='coerce')
    merged = pd.merge(merged, hd_subset, left_on='ipeds_unitid', right_on='UNITID',
                      how='left', suffixes=('', '_ipeds'))
    print(f"\nAfter IPEDS HD join: {len(merged)} rows, "
          f"{merged['INSTNM'].notna().sum()} with institution name")

    # 4. EADA -- load from xlsx
    eada_xlsx = os.path.join(RAW_DATA, 'EADA_All_Data_Combined_2022-2023_SAS_SPSS_EXCEL',
                              'EADA_2023.xlsx')
    if os.path.exists(eada_xlsx):
        print(f"\nLoading EADA from {os.path.basename(eada_xlsx)}...")
        eada = pd.read_excel(eada_xlsx)
        if 'UNITID' not in eada.columns:
            uc = next((c for c in eada.columns if c.lower() == 'unitid'), None)
            if uc:
                eada = eada.rename(columns={uc: 'UNITID'})
        eada['UNITID'] = pd.to_numeric(eada['UNITID'], errors='coerce')
        # Keep first 10 numeric/aggregate columns alongside UNITID; the user can
        # re-merge specific EADA fields later if needed.
        keep = ['UNITID'] + [c for c in eada.columns if c != 'UNITID'][:10]
        eada_subset = eada[keep].copy()
        merged = pd.merge(merged, eada_subset, left_on='ipeds_unitid', right_on='UNITID',
                          how='left', suffixes=('', '_eada'))
        print(f"After EADA join: {len(merged)} rows")
    else:
        print(f"\nEADA xlsx not found, skipping. ({eada_xlsx})")

    # 5. Macro: country_code, year=2023
    macro = pd.read_csv(os.path.join(RAW_DATA, 'macro_combined.csv'))
    if 'country_code' in merged.columns:
        macro_2023 = macro[macro['year'] == 2023].drop(columns=['year'], errors='ignore')
        merged = pd.merge(merged, macro_2023, on='country_code', how='left',
                          suffixes=('', '_macro'))
        n_gdp = merged['gdp_per_capita_ppp'].notna().sum() if 'gdp_per_capita_ppp' in merged.columns else 0
        print(f"\nAfter macro join: {len(merged)} rows, {n_gdp} with GDP data")

    # 6. Save
    out = os.path.join(RAW_DATA, 'analysis_dataset.csv')
    merged.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(f"  shape: {merged.shape}")

    # Quick summary
    print('\n=== Summary ===')
    if 'country_code' in merged.columns:
        intl = merged[(merged['country_code'].notna()) & (merged['country_code'] != 'USA')]
        print(f"  International athletes: {len(intl)}")
        if 'gdp_per_capita_ppp' in intl.columns:
            print(f"  Of which joined to GDP: {intl['gdp_per_capita_ppp'].notna().sum()}")
        print(f"  Top 10 international source countries:")
        print('  ' + intl['country_code'].value_counts().head(10).to_string().replace('\n', '\n  '))
    if 'sport' in merged.columns:
        print(f"  By sport: {merged['sport'].value_counts().to_dict()}")
    if 'ncaa_division' in merged.columns:
        print(f"  By NCAA division: {merged['ncaa_division'].value_counts(dropna=False).to_dict()}")


if __name__ == '__main__':
    main()
