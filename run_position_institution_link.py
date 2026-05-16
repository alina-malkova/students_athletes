"""
Link position × country findings to institution characteristics.

For each (sport, position, intl-vs-domestic) cell, summarize the *kind of school*
that recruits there:
  - Carnegie classification (research / master's / baccalaureate)
  - Public vs private control
  - Athletic resources (EADA): total athletic revenue, sport-specific revenue,
    student-aid budget, recruiting expenditure.
  - Nonresident-alien share of total enrollment (EF2023a) — proxy for the
    school's broader international student exposure.

Inputs:
  raw_data/student_rosters_processed.csv   athlete-level (ipeds_unitid, position, ...)
  raw_data/HD2023.csv                       institution directory
  raw_data/EF2023a.csv                      enrollment (NRAL columns)
  raw_data/EADA_2023.xlsx                   athletic finances

Output: output/position_institution_link.txt + CSVs.
"""
from __future__ import annotations

import os
import re
import warnings
import pandas as pd
import numpy as np

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')

EADA_XLSX = os.path.join(RAW_DATA,
    'EADA_All_Data_Combined_2022-2023_SAS_SPSS_EXCEL', 'EADA_2023.xlsx')


# Map our position-analyzed sport keys to EADA sport-suffix tokens
SPORT_TO_EADA = {
    'Soccer':     'Soccer',
    'Basketball': 'Bskball',
    'Football':   'Football',
    'Volleyball': 'Vollball',
    'Baseball':   'Baseball',
    'Softball':   'Softball',
    'Ice Hockey': 'IceHcky',
}

# Same classifiers as run_position_analysis.py
def classify_soccer(p):
    if pd.isna(p): return 'Unknown'
    s = str(p).upper().strip()
    if re.search(r'\bGK|GOAL|KEEPER', s): return 'GK'
    if re.search(r'\bF\b|FW|FORWARD|STRIKER|ST', s): return 'F'
    if re.search(r'\bM\b|MID|MF', s): return 'M'
    if re.search(r'\bD\b|DEF|CB|FB|LB|RB|CENT', s): return 'D'
    return 'Unknown'

def classify_basketball(p):
    if pd.isna(p): return 'Unknown'
    s = str(p).upper().strip()
    if re.search(r'^C\b|CENTER|^C/', s) and 'GUARD' not in s: return 'C'
    if re.search(r'^G/F|G-F|^GF|GUARD/FORWARD', s): return 'G/F'
    if re.search(r'^F/C|F-C|FORWARD/CENTER', s): return 'F/C'
    if re.search(r'^F\b|FORWARD|^PF|^SF', s): return 'F'
    if re.search(r'^G\b|GUARD|^PG|^SG', s): return 'G'
    return 'Unknown'

def classify_football(p):
    if pd.isna(p): return 'Unknown'
    s = str(p).upper().strip()
    if re.search(r'\bQB|QUARTERBACK', s): return 'QB'
    if re.search(r'\bRB|HB|FB|RUN|BACK', s) and 'WIDE' not in s and 'DEFENS' not in s:
        return 'RB'
    if re.search(r'\bWR|WIDE\s*RECEIVER|REC', s): return 'WR'
    if re.search(r'\bTE\b|TIGHT', s): return 'TE'
    if re.search(r'\bOL|OFFENSIVE|OG|OT|OC|^G\b|^T\b', s): return 'OL'
    if re.search(r'\bDL|DEFENSIVE|DE\b|DT\b|NT\b', s): return 'DL'
    if re.search(r'\bLB|LINEBACKER', s): return 'LB'
    if re.search(r'\bDB|CB\b|^S\b|SAFETY|CORNER', s): return 'DB'
    if re.search(r'\bK\b|KICKER|^P$|PUNT|LS\b|SNAPPER', s): return 'ST'
    return 'Unknown'

def classify_volleyball(p):
    if pd.isna(p): return 'Unknown'
    s = str(p).upper().strip()
    if re.search(r'\bOH\b|OUTSIDE', s): return 'OH'
    if re.search(r'\bMB\b|MIDDLE', s): return 'MB'
    if re.search(r'\bOPP|RIGHT|RS\b', s): return 'OPP'
    if re.search(r'\bS\b|SETTER', s): return 'S'
    if re.search(r'\bL\b|LIBERO|DS\b|DEFENS', s): return 'L/DS'
    return 'Unknown'

def classify_baseball(p):
    if pd.isna(p): return 'Unknown'
    s = str(p).upper().strip()
    if re.search(r'\bP\b|PITCH|RHP|LHP', s): return 'P'
    if re.search(r'\bC\b|CATCH', s): return 'C'
    if re.search(r'\bINF|1B|2B|3B|SS|INFIELD', s): return 'INF'
    if re.search(r'\bOF|OUT', s): return 'OF'
    if re.search(r'\bDH|UTIL', s): return 'UTIL'
    return 'Unknown'

def classify_icehockey(p):
    if pd.isna(p): return 'Unknown'
    s = str(p).upper().strip()
    if re.search(r'\bG\b|GOAL', s): return 'G'
    if re.search(r'\bD\b|DEF', s): return 'D'
    if re.search(r'\bF\b|FORWARD|CENT|^C\b|^LW|^RW|WING', s): return 'F'
    return 'Unknown'

CLASSIFIERS = {
    'Soccer': classify_soccer, 'Basketball': classify_basketball,
    'Football': classify_football, 'Volleyball': classify_volleyball,
    'Baseball': classify_baseball, 'Softball': classify_baseball,
    'Ice Hockey': classify_icehockey,
}


# Carnegie 2021 basic — collapse to coarse buckets
def carnegie_bucket(code):
    if pd.isna(code): return 'Unknown'
    code = int(code)
    if code in (15, 16, 17): return 'R1/R2/D-PU'        # doctoral/research
    if code in (18, 19, 20, 21): return 'Master\'s'
    if code in (22, 23): return 'Baccalaureate'
    if code in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14): return 'Associate'
    return 'Special/Other'


def load_institution_panel():
    """Return one row per UNITID with school chars + EADA finances."""
    hd = pd.read_csv(os.path.join(RAW_DATA, 'HD2023.csv'), encoding='latin-1',
                     low_memory=False)
    hd = hd.rename(columns={hd.columns[0]: 'UNITID'})
    hd = hd[['UNITID', 'INSTNM', 'CONTROL', 'C21BASIC', 'STABBR', 'OBEREG',
             'INSTSIZE']].copy()
    hd['control_lbl'] = hd['CONTROL'].map({1: 'Public', 2: 'Private NP',
                                            3: 'Private FP'})
    hd['carnegie'] = hd['C21BASIC'].apply(carnegie_bucket)

    # EF2023a — nonresident alien share, using level-99 (grand total)
    ef = pd.read_csv(os.path.join(RAW_DATA, 'EF2023a.csv'), encoding='latin-1',
                     low_memory=False)
    # EFALEVEL == 1 is total/all students; some files use 99. We want one row per
    # institution. Inspect: take EFALEVEL == 1 (Total) if present, else min.
    if 'EFALEVEL' in ef.columns:
        tot = ef[ef['EFALEVEL'] == 1][['UNITID', 'EFTOTLT', 'EFNRALT']].copy()
        if tot.empty:
            tot = ef.groupby('UNITID', as_index=False)[
                ['EFTOTLT', 'EFNRALT']].sum()
    else:
        tot = ef.groupby('UNITID', as_index=False)[
            ['EFTOTLT', 'EFNRALT']].sum()
    tot['nral_share'] = tot['EFNRALT'] / tot['EFTOTLT'].replace(0, np.nan)
    tot = tot.rename(columns={'EFTOTLT': 'enrollment_total',
                              'EFNRALT': 'enrollment_nral'})

    # EADA — overall athletic finance + per-sport revenue/expense + scholarship
    eada = pd.read_excel(EADA_XLSX)
    eada = eada.rename(columns={'unitid': 'UNITID'})
    keep_overall = ['UNITID', 'STUDENTAID_TOTAL', 'RECRUITEXP_TOTAL',
                    'IL_TOTAL_REVENUE_ALL', 'GRND_TOTAL_REVENUE',
                    'IL_REVENUE_MENALL', 'IL_REVENUE_WOMENALL']
    keep_overall = [c for c in keep_overall if c in eada.columns]
    eada_ov = eada[keep_overall].copy()

    # Per-sport columns we'll attach inline (the merge function will pull them)
    eada_all = eada[['UNITID'] + [c for c in eada.columns if any(
        sfx in c for sfx in SPORT_TO_EADA.values())]].copy()

    inst = hd.merge(tot, on='UNITID', how='left')
    inst = inst.merge(eada_ov, on='UNITID', how='left')
    return inst, eada_all


def sport_revenue_col(eada_all, sport_key, gender):
    """Return per-sport revenue column for a sport+gender, NaN-fill else."""
    sfx = SPORT_TO_EADA[sport_key]
    g = 'MEN' if gender.lower() in ('m', 'men', 'male') else 'WOMEN'
    col = f'TOTAL_REV_MENWOMEN_{sfx}'   # combined men+women col is what EADA stores
    rev_col = f'TOTAL_REV_MENWOMEN_{sfx}'
    return rev_col if rev_col in eada_all.columns else None


def sport_team_rev(eada_all, sport_key):
    """Per-team total revenue (men+women combined) col for one sport."""
    sfx = SPORT_TO_EADA[sport_key]
    col = f'TOTAL_REV_MENWOMEN_{sfx}'
    return col if col in eada_all.columns else None


def sport_team_opexp(eada_all, sport_key):
    sfx = SPORT_TO_EADA[sport_key]
    col = f'TOTAL_OPEXP_MENWOMEN_{sfx}'
    return col if col in eada_all.columns else None


def summarize(df, group_cols, value_cols):
    out = df.groupby(group_cols).agg(
        n=(value_cols[0], 'size'),
        **{c: (c, 'mean') for c in value_cols},
    ).reset_index()
    return out


def main():
    warnings.filterwarnings('ignore')
    print('Loading institution panel...')
    inst, eada_all = load_institution_panel()
    print(f'  Institutions: {len(inst):,}')
    print(f'  NRAL share available for {inst["nral_share"].notna().sum():,}')
    print(f'  EADA rows: {len(eada_all):,}')

    print('Loading rosters...')
    df = pd.read_csv(os.path.join(RAW_DATA, 'student_rosters_processed.csv'),
                     low_memory=False)
    df = df[df['ipeds_unitid'].notna()].copy()
    df['UNITID'] = df['ipeds_unitid'].astype(int)
    df = df.merge(inst, on='UNITID', how='left')
    df = df.merge(eada_all, on='UNITID', how='left')
    df['is_intl'] = ((df['country_code'].notna()) &
                     (df['country_code'] != 'USA')).astype(int)
    print(f'  Athletes after merge: {len(df):,}')
    print(f'  with institution chars: '
          f'{df["nral_share"].notna().sum():,}')

    fh = open(os.path.join(OUTPUT, 'position_institution_link.txt'), 'w')
    fh.write('Position × country × institution-characteristics link\n')
    fh.write('=' * 64 + '\n')
    fh.write('Roster cross-section (2024-25, all NCAA divisions, all sports).\n')
    fh.write('Institution chars: IPEDS HD2023 + EF2023a (NRAL share) + EADA 2023.\n\n')

    for sport, classifier in CLASSIFIERS.items():
        sub = df[df['sport'] == sport].copy()
        if sub.empty:
            continue
        sub['pos_group'] = sub['position'].apply(classifier)
        sub = sub[sub['pos_group'] != 'Unknown']

        rev_col = sport_team_rev(eada_all, sport)
        opx_col = sport_team_opexp(eada_all, sport)

        rows = []
        for pos in sorted(sub['pos_group'].unique()):
            s = sub[sub['pos_group'] == pos]
            for intl_flag, label in [(0, 'Dom'), (1, 'Int')]:
                ss = s[s['is_intl'] == intl_flag]
                if ss.empty: continue
                rec = {
                    'pos':         pos,
                    'origin':      label,
                    'n':           len(ss),
                    'pct_priv':    100 * (ss['control_lbl']
                                          .isin(['Private NP', 'Private FP'])
                                          ).mean(),
                    'pct_R1R2':    100 * (ss['carnegie'] == 'R1/R2/D-PU').mean(),
                    'med_nral_pct': 100 * ss['nral_share'].median(),
                    'med_recruitK': ss['RECRUITEXP_TOTAL'].median()/1000,
                    'med_stuaidK': ss['STUDENTAID_TOTAL'].median()/1000,
                }
                if rev_col:
                    rec['med_sportRevK'] = ss[rev_col].median()/1000
                if opx_col:
                    rec['med_sportExpK'] = ss[opx_col].median()/1000
                rows.append(rec)

        tab = pd.DataFrame(rows)
        # round
        for c in tab.columns:
            if tab[c].dtype == float:
                tab[c] = tab[c].round(1)
        tab.to_csv(os.path.join(OUTPUT,
            f'position_institution_{sport.replace(" ","_")}.csv'), index=False)
        print(f'\n=== {sport} ===')
        print(tab.to_string(index=False))
        fh.write(f'\n=== {sport} ===\n')
        fh.write(tab.to_string(index=False) + '\n')

    # --- One pooled regression: P(international) on institution chars,
    # controlling for sport×position FE.  Just basketball as illustration.
    fh.write('\n\nDom-vs-Intl summary across all 7 sports (pooled)\n')
    pooled = df[df['sport'].isin(CLASSIFIERS.keys())].copy()
    pooled['pos_group'] = pooled.apply(
        lambda r: CLASSIFIERS[r['sport']](r['position']), axis=1)
    pooled = pooled[pooled['pos_group'] != 'Unknown']
    summ = pooled.groupby('is_intl').agg(
        n=('UNITID', 'size'),
        pct_priv=('control_lbl', lambda x: 100*x.isin(
            ['Private NP', 'Private FP']).mean()),
        pct_R1R2=('carnegie', lambda x: 100*(x == 'R1/R2/D-PU').mean()),
        med_nral_pct=('nral_share', lambda x: 100*x.median()),
        med_recruitK=('RECRUITEXP_TOTAL', lambda x: x.median()/1000),
        med_stuaidK=('STUDENTAID_TOTAL', lambda x: x.median()/1000),
        med_total_revK=('IL_TOTAL_REVENUE_ALL', lambda x: x.median()/1000),
    ).round(1)
    summ.index = summ.index.map({0: 'Domestic', 1: 'International'})
    print('\n=== Pooled across 7 sports ===')
    print(summ.to_string())
    fh.write(summ.to_string() + '\n')
    fh.close()
    print(f'\nWrote {os.path.join(OUTPUT, "position_institution_link.txt")}')


if __name__ == '__main__':
    main()
