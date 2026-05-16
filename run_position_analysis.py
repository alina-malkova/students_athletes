"""
Position-level analyses for each major sport.

For each sport with substantial international athletes, we:
  (a) Parse the free-text position field into clean position groups.
  (b) Compute international share by position group.
  (c) Identify the top source countries for each position group.
  (d) Compute regional and country-level concentration (Herfindahl)
      by position.

Uses the student's full cross-section (raw_data/student_rosters_processed.csv)
because it has the broadest sport / division coverage.

Output: output/position_analysis.txt + output/position_table_{sport}.csv
"""
from __future__ import annotations

import os
import re
import warnings
import pandas as pd

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')


# Region mapping (same eight regions as §3.6 and §3.7 of the paper).
REGION = {
    **{c: 'SSA' for c in ['AGO','BDI','BEN','BFA','BWA','CAF','CIV','CMR','COD','COG',
                           'COM','CPV','DJI','ERI','ETH','GAB','GHA','GIN','GMB','GNB',
                           'GNQ','KEN','LBR','LSO','MDG','MLI','MOZ','MRT','MUS','MWI',
                           'NAM','NER','NGA','RWA','SDN','SEN','SLE','SOM','SSD','STP',
                           'SWZ','SYC','TCD','TGO','TZA','UGA','ZAF','ZMB','ZWE']},
    **{c: 'MENA' for c in ['ARE','BHR','DJI','DZA','EGY','IRN','IRQ','ISR','JOR','KWT',
                            'LBN','LBY','MAR','OMN','PSE','QAT','SAU','SYR','TUN','TUR',
                            'YEM']},
    **{c: 'SA' for c in ['AFG','BGD','BTN','IND','LKA','MDV','NPL','PAK']},
    **{c: 'EAP' for c in ['AUS','BRN','CHN','FJI','HKG','IDN','JPN','KHM','KIR','KOR',
                           'LAO','MAC','MMR','MNG','MYS','NCL','NZL','PHL','PLW','PNG',
                           'PRK','SGP','SLB','THA','TLS','TON','TWN','TUV','VNM','VUT',
                           'WSM']},
    **{c: 'LAC' for c in ['ARG','ATG','BHS','BLZ','BOL','BRA','BRB','CHL','COL','CRI',
                           'CUB','DMA','DOM','ECU','GRD','GTM','GUY','HND','HTI','JAM',
                           'KNA','LCA','MEX','NIC','PAN','PER','PRY','SLV','SUR','TTO',
                           'URY','VCT','VEN','VGB']},
    **{c: 'NA' for c in ['BMU','CAN','USA']},
    **{c: 'WEU' for c in ['AND','AUT','BEL','CHE','CYP','DEU','DNK','ESP','FIN','FRA',
                           'FRO','GBR','GIB','GRC','GRL','IMN','IRL','ISL','ITA','LIE',
                           'LUX','MCO','MLT','NLD','NOR','PRT','SMR','SWE','VAT']},
    **{c: 'EEU' for c in ['ALB','ARM','AZE','BGR','BIH','BLR','CZE','EST','GEO','HRV',
                           'HUN','KAZ','KGZ','LTU','LVA','MDA','MKD','MNE','POL','ROU',
                           'RUS','SRB','SVK','SVN','TJK','TKM','UKR','UZB','XKX']},
}


# ----------------------------------------------------------------
# Per-sport position classifiers
# ----------------------------------------------------------------

def classify_soccer(pos):
    if pd.isna(pos): return 'Unknown'
    s = str(pos).upper().strip()
    if re.search(r'\bGK|GOAL|KEEPER', s): return 'GK'
    if re.search(r'\bF\b|FW|FORWARD|STRIKER|ST', s): return 'F'
    if re.search(r'\bM\b|MID|MF', s): return 'M'
    if re.search(r'\bD\b|DEF|CB|FB|LB|RB|CENT', s): return 'D'
    return 'Unknown'


def classify_basketball(pos):
    if pd.isna(pos): return 'Unknown'
    s = str(pos).upper().strip()
    if re.search(r'^C\b|CENTER|^C/', s) and 'GUARD' not in s: return 'C'
    if re.search(r'^G/F|G-F|^GF|GUARD/FORWARD', s): return 'G/F'
    if re.search(r'^F/C|F-C|FORWARD/CENTER', s): return 'F/C'
    if re.search(r'^F\b|FORWARD|^PF|^SF', s): return 'F'
    if re.search(r'^G\b|GUARD|^PG|^SG', s): return 'G'
    return 'Unknown'


def classify_football(pos):
    if pd.isna(pos): return 'Unknown'
    s = str(pos).upper().strip()
    if re.search(r'\bQB|QUARTERBACK', s): return 'QB'
    if re.search(r'\bRB|HB|FB|RUN|BACK', s) and 'WIDE' not in s and 'DEFENS' not in s:
        return 'RB'
    if re.search(r'\bWR|WIDE\s*RECEIVER|REC', s): return 'WR'
    if re.search(r'\bTE\b|TIGHT', s): return 'TE'
    if re.search(r'\bOL|OFFENSIVE|OG|OT|OC|^G\b|^T\b', s): return 'OL'
    if re.search(r'\bDL|DEFENSIVE|DE\b|DT\b|NT\b', s): return 'DL'
    if re.search(r'\bLB|LINEBACKER', s): return 'LB'
    if re.search(r'\bDB|CB\b|^S\b|SAFETY|CORNER', s): return 'DB'
    if re.search(r'\bK\b|KICKER|^P$|PUNT|LS\b|SNAPPER', s): return 'ST'  # special teams
    return 'Unknown'


def classify_volleyball(pos):
    if pd.isna(pos): return 'Unknown'
    s = str(pos).upper().strip()
    if re.search(r'\bOH\b|OUTSIDE', s): return 'OH'
    if re.search(r'\bMB\b|MIDDLE', s): return 'MB'
    if re.search(r'\bOPP|RIGHT|RS\b', s): return 'OPP'
    if re.search(r'\bS\b|SETTER', s): return 'S'
    if re.search(r'\bL\b|LIBERO|DS\b|DEFENS', s): return 'L/DS'
    return 'Unknown'


def classify_baseball(pos):
    if pd.isna(pos): return 'Unknown'
    s = str(pos).upper().strip()
    if re.search(r'\bP\b|PITCH|RHP|LHP', s): return 'P'
    if re.search(r'\bC\b|CATCH', s): return 'C'
    if re.search(r'\bINF|1B|2B|3B|SS|INFIELD', s): return 'INF'
    if re.search(r'\bOF|OUT', s): return 'OF'
    if re.search(r'\bDH|UTIL', s): return 'UTIL'
    return 'Unknown'


def classify_icehockey(pos):
    if pd.isna(pos): return 'Unknown'
    s = str(pos).upper().strip()
    if re.search(r'\bG\b|GOAL', s): return 'G'
    if re.search(r'\bD\b|DEF', s): return 'D'
    if re.search(r'\bF\b|FORWARD|CENT|^C\b|^LW|^RW|WING', s): return 'F'
    return 'Unknown'


SPORT_CLASSIFIERS = {
    'Soccer':        classify_soccer,
    'Basketball':    classify_basketball,
    'Football':      classify_football,
    'Volleyball':    classify_volleyball,
    'Baseball':      classify_baseball,
    'Softball':      classify_baseball,  # similar positions
    'Ice Hockey':    classify_icehockey,
}


def position_breakdown(df, sport, classifier):
    sub = df[df['sport'] == sport].copy()
    sub['position_group'] = sub['position'].apply(classifier)
    sub = sub[sub['position_group'] != 'Unknown']
    sub['region'] = sub['country_code'].map(REGION).fillna('OTHER')

    rows = []
    for pos in sub['position_group'].unique():
        s = sub[sub['position_group'] == pos]
        n = len(s)
        n_intl = ((s['country_code'].notna()) & (s['country_code'] != 'USA')).sum()
        intl_pct = 100 * n_intl / n if n else 0
        intl = s[(s['country_code'].notna()) & (s['country_code'] != 'USA')]
        top_countries = intl['country_code'].value_counts().head(3)
        top_str = ', '.join(f'{c} {v}' for c, v in top_countries.items())
        # Regional Herfindahl
        if len(intl):
            rs = intl['region'].value_counts(normalize=True)
            hhi = (rs ** 2).sum()
            top_region = rs.idxmax(); top_region_share = rs.max()
        else:
            hhi = float('nan'); top_region = '-'; top_region_share = 0
        rows.append({
            'position':      pos,
            'n_total':       n,
            'n_intl':        n_intl,
            'intl_pct':      round(intl_pct, 1),
            'top_countries': top_str,
            'top_region':    f'{top_region} ({100*top_region_share:.0f}%)' if intl.shape[0] else '-',
            'reg_HHI':       round(hhi, 3) if intl.shape[0] else None,
        })
    out = pd.DataFrame(rows).sort_values('n_total', ascending=False)
    out.to_csv(os.path.join(OUTPUT, f'position_table_{sport.replace(" ","_")}.csv'),
                index=False)
    return out


def main():
    warnings.filterwarnings('ignore')
    df = pd.read_csv(os.path.join(RAW_DATA, 'student_rosters_processed.csv'),
                     low_memory=False)
    print(f'Loaded student cross-section: {df.shape}')

    fh = open(os.path.join(OUTPUT, 'position_analysis.txt'), 'w')
    fh.write('Position-level analyses by sport\n')
    fh.write(f'  All-sport sample: {len(df):,}\n')
    fh.write(f'  D-I + D-II + D-III, 2024-25 cross-section.\n')

    for sport, classifier in SPORT_CLASSIFIERS.items():
        out = position_breakdown(df, sport, classifier)
        print(f'\n=== {sport} ===')
        print(out.to_string(index=False))
        fh.write(f'\n\n=== {sport} ===\n')
        fh.write(out.to_string(index=False) + '\n')

    fh.close()
    print(f'\nWrote {os.path.join(OUTPUT, "position_analysis.txt")}')


if __name__ == '__main__':
    main()
