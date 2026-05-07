"""
Local runner for Section 13 of the pipeline (country mapping).

Adds country_code + country_match_confidence columns to each roster CSV
that exists. Mirrors the notebook's map_hometown_to_country, including
the post-patch edge-case handling.

Usage:
    python run_country_mapping.py
"""
from __future__ import annotations

import os
import re
import pandas as pd
from thefuzz import fuzz, process

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')

US_STATE_ABBREVS = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
    'DC', 'PR', 'VI', 'GU', 'AS', 'MP',
}
US_STATE_NAMES = {
    'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
    'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
    'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
    'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
    'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
    'new hampshire', 'new jersey', 'new mexico', 'new york',
    'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon',
    'pennsylvania', 'rhode island', 'south carolina', 'south dakota',
    'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
    'west virginia', 'wisconsin', 'wyoming', 'district of columbia',
    'puerto rico',
    "hawai'i", 'd.c.', 'washington d.c.',
}
US_AP_ABBREVS = {
    'ala.', 'ariz.', 'ark.', 'calif.', 'cal.', 'colo.', 'conn.', 'del.', 'fla.',
    'ga.', 'ill.', 'illi.', 'ind.', 'kan.', 'kans.', 'ken.', 'ky.', 'la.', 'louis.',
    'md.', 'mass.', 'mich.', 'min.', 'minn.', 'miss.', 'mo.', 'mont.',
    'neb.', 'nev.', 'n.h.', 'n.j.', 'n.m.', 'n.y.', 'n.c.', 'n.d.',
    'okla.', 'ore.', 'oreg.', 'pa.', 'penn.', 'r.i.',
    's.c.', 's.d.', 's. dak.', 'sd.',
    'tenn.', 'tex.', 'texa.', 'vt.', 'va.', 'wash.', 'w.va.', 'w. va.', 'wva.',
    'wis.', 'wisc.', 'wyo.',
}
CANADIAN_PROVINCES = {
    'alberta', 'british columbia', 'manitoba', 'new brunswick',
    'newfoundland and labrador', 'newfoundland', 'nova scotia',
    'northwest territories', 'nunavut', 'ontario', 'prince edward island',
    'quebec', 'saskatchewan', 'yukon',
    'ab', 'bc', 'mb', 'nb', 'nl', 'ns', 'nt', 'nu', 'on', 'pe', 'qc', 'sk', 'yt',
    'b.c.', 'p.e.i.', 'n.b.', 'n.s.', 'n.l.', 'n.w.t.',
    'ont.', 'que.', 'sask.', 'alta.', 'man.',
}
COUNTRY_ALIASES = {
    'united states': 'USA', 'usa': 'USA', 'us': 'USA',
    'u.s.': 'USA', 'u.s.a.': 'USA', 'america': 'USA',
    'usvi': 'USA', 'u.s. virgin islands': 'USA',
    'nyc': 'USA',
    'canada': 'CAN', 'can.': 'CAN', 'can': 'CAN',
    'england': 'GBR', 'scotland': 'GBR', 'wales': 'GBR',
    'northern ireland': 'GBR', 'great britain': 'GBR',
    'uk': 'GBR', 'u.k.': 'GBR',
    'south korea': 'KOR', 'korea': 'KOR',
    'trinidad': 'TTO', 'trinidad and tobago': 'TTO', 'tto': 'TTO',
    'ivory coast': 'CIV', "cote d'ivoire": 'CIV',
    'the bahamas': 'BHS', 'bahamas': 'BHS',
    'republic of ireland': 'IRL', 'ireland': 'IRL',
    # Country alias forms common on rosters but absent/different in WDI
    'aus': 'AUS', 'aus.': 'AUS',
    'nz': 'NZL', 'n.z.': 'NZL',
    'czech republic': 'CZE', 'czechia': 'CZE',
    'russia': 'RUS', 'russian federation': 'RUS',
    'turkey': 'TUR', 'turkiye': 'TUR',
    'slovakia': 'SVK', 'slovak republic': 'SVK',
    'egypt': 'EGY',
    'venezuela': 'VEN',
    'iran': 'IRN',
    'syria': 'SYR',
    'yemen': 'YEM',
    'hong kong': 'HKG',
    'bosnia': 'BIH', 'bosnia and herzegovina': 'BIH',
    'gambia': 'GMB',
    'st. kitts': 'KNA', 'saint kitts': 'KNA',
    'bvi': 'VGB', 'british virgin islands': 'VGB',
    # Caribbean and parish-level locations
    'st. elizabeth parish': 'JAM',
}


def build_wdi_dict():
    macro = pd.read_csv(os.path.join(RAW_DATA, 'macro_combined.csv'))
    wdi = macro[['country_code', 'country_name']].drop_duplicates()
    return dict(zip(wdi['country_name'].str.lower(), wdi['country_code']))


def _norm(s: str) -> str:
    """Lookup-normalize a key: lowercase, remove periods, collapse spaces."""
    return ' '.join(s.lower().replace('.', '').split())


# International region/state hints (e.g. Australian/Austrian/Mexican states
# that appear as the trailing comma-part on rosters).
INTL_REGIONS = {
    'tasmania': 'AUS', 'new south wales': 'AUS', 'queensland': 'AUS',
    'victoria': 'AUS', 'south australia': 'AUS', 'western australia': 'AUS',
    'northern territory': 'AUS', 'australian capital territory': 'AUS',
    'vorarlberg': 'AUT', 'tyrol': 'AUT', 'salzburg': 'AUT', 'styria': 'AUT',
    'yucatan': 'MEX', 'jalisco': 'MEX', 'sonora': 'MEX', 'chihuahua': 'MEX',
    'baja california': 'MEX', 'nuevo leon': 'MEX',
    'dubai': 'ARE', 'abu dhabi': 'ARE',
    'madrid': 'ESP', 'catalonia': 'ESP', 'andalusia': 'ESP',
    'taiwan': 'TWN',  # not in WDI, but useful as an output code
}


def make_mapper(wdi_country_dict):
    wdi_country_names = list(wdi_country_dict.keys())

    # Pre-normalize all sets/dicts so lookups can use a single normalized key.
    NORM_US_AP        = {_norm(s) for s in US_AP_ABBREVS}
    NORM_US_NAMES     = {_norm(s) for s in US_STATE_NAMES}
    NORM_PROVINCES    = {_norm(s) for s in CANADIAN_PROVINCES}
    NORM_ALIASES      = {_norm(k): v for k, v in COUNTRY_ALIASES.items()}
    NORM_INTL_REGIONS = {_norm(k): v for k, v in INTL_REGIONS.items()}
    NORM_WDI          = {_norm(k): v for k, v in wdi_country_dict.items()}

    # Trailing-comma typos like "Bronx, N.Y," produce an empty last part —
    # strip that. Also strip "africa", "asia", "europe" when used as a
    # continent suffix and try the preceding token as a country.
    CONTINENT_SUFFIX = {'africa', 'asia', 'europe', 'oceania', 'south america', 'north america'}

    def map_hometown_to_country(hometown):
        if pd.isna(hometown) or not str(hometown).strip():
            return None, 'missing'
        hometown = str(hometown).strip().rstrip(',')
        parts = [p.strip().rstrip(',') for p in hometown.split(',') if p.strip().rstrip(',')]
        # If the last token is a continent name, drop it and let the
        # preceding token act as the country (e.g. "Burkina Faso, Africa").
        if parts and parts[-1].lower() in CONTINENT_SUFFIX:
            parts = parts[:-1]
        if len(parts) >= 2:
            last_part = parts[-1].strip()
            last_upper = last_part.upper().rstrip('.')
            last_norm = _norm(last_part)  # lowercase, no periods, single spaces

            if last_upper in US_STATE_ABBREVS:
                return 'USA', 'exact'
            if last_upper.replace('.', '') in US_STATE_ABBREVS:
                return 'USA', 'exact'
            if last_norm in NORM_US_NAMES:
                return 'USA', 'exact'
            if last_norm in NORM_US_AP:
                return 'USA', 'exact'
            if last_norm in NORM_PROVINCES:
                return 'CAN', 'exact'
            if last_norm in NORM_WDI:
                return NORM_WDI[last_norm], 'exact'
            if last_norm in NORM_ALIASES:
                return NORM_ALIASES[last_norm], 'exact'
            if last_norm in NORM_INTL_REGIONS:
                return NORM_INTL_REGIONS[last_norm], 'exact'

            # Region+country fallback (e.g. "Western Australia" -> "australia").
            tokens = last_norm.split()
            tail_token = tokens[-1] if tokens else ''
            if tail_token and tail_token != last_norm:
                if tail_token in NORM_WDI:
                    return NORM_WDI[tail_token], 'exact'
                if tail_token in NORM_ALIASES:
                    return NORM_ALIASES[tail_token], 'exact'

            result = process.extractOne(last_norm, list(NORM_WDI.keys()), scorer=fuzz.ratio)
            if result and result[1] >= 80:
                return NORM_WDI[result[0]], 'fuzzy'

        if len(parts) == 1:
            # After dropping a continent suffix, the lone remaining token
            # might itself be a country name.
            only = _norm(parts[0])
            if only in NORM_WDI:
                return NORM_WDI[only], 'exact'
            if only in NORM_ALIASES:
                return NORM_ALIASES[only], 'exact'
            return 'USA', 'fuzzy'
        return None, 'unmatched'

    return map_hometown_to_country


def main():
    wdi_country_dict = build_wdi_dict()
    print(f"WDI dictionary: {len(wdi_country_dict)} countries")
    mapper = make_mapper(wdi_country_dict)

    rosters = [
        ('Soccer', os.path.join(RAW_DATA, 'soccer_rosters.csv')),
        ('TFRRS',  os.path.join(RAW_DATA, 'tfrrs_all_rosters.csv')),
        ('Track',  os.path.join(RAW_DATA, 'track_rosters.csv')),
    ]
    unmatched = set()

    for label, path in rosters:
        if not os.path.exists(path):
            print(f"\n  {label}: missing, skipping ({path})")
            continue
        df = pd.read_csv(path)
        if 'hometown' not in df.columns:
            print(f"\n  {label}: no hometown column, skipping")
            continue
        results = df['hometown'].apply(mapper)
        df['country_code'] = [r[0] for r in results]
        df['country_match_confidence'] = [r[1] for r in results]
        df.to_csv(path, index=False)
        n_mapped = df['country_code'].notna().sum()
        print(f"\n  {label}: mapped {n_mapped}/{len(df)} ({100 * n_mapped / max(len(df), 1):.1f}%)")
        print(df['country_match_confidence'].value_counts().to_string())
        for ht in df.loc[df['country_match_confidence'] == 'unmatched', 'hometown'].dropna().unique():
            unmatched.add(ht)

    if unmatched:
        out = os.path.join(RAW_DATA, 'unmatched_hometowns.csv')
        pd.DataFrame({'hometown': sorted(unmatched)}).to_csv(out, index=False)
        print(f"\nUnmatched hometowns ({len(unmatched)} unique) -> {out}")


if __name__ == '__main__':
    main()
