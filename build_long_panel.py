"""
Build the long country × year × sport panel from the eight scraped
sport CSVs, then merge to time-varying country-year macro data.

Output: raw_data/country_year_sport_panel.csv with rows
(country_code, year, sport) and columns for:
  - athletes_count
  - log_athletes (log(1+count))
  - is_present (binary, athletes > 0)
  - gdp_per_capita_ppp_t          (time-varying)
  - political_stability_t          (time-varying)
  - gdp_shock_t                    (time-varying)
  - disaster_deaths_t              (time-varying)
  - disaster_events_t              (time-varying)
  - any_conflict_t                 (time-varying)
  - pop_15_24                      (time-invariant proxy, 2018-22 avg)
  - log_pop_15_24
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')

SPORT_FILES = {
    'track':      'track_rosters_panel.csv',
    'soccer':     'soccer_rosters_panel.csv',
    'basketball': 'basketball_rosters_panel.csv',
    'tennis':     'tennis_rosters_panel.csv',
    'golf':       'golf_rosters_panel.csv',
    'swimming':   'swimming_rosters_panel.csv',
    'icehockey':  'icehockey_rosters_panel.csv',
    'volleyball': 'volleyball_rosters_panel.csv',
}

YEARS = list(range(2019, 2024))


def main():
    # ---- (1) Stack sport CSVs --------------------------------------------
    parts = []
    for sport, fname in SPORT_FILES.items():
        p = os.path.join(RAW_DATA, fname)
        df = pd.read_csv(p, low_memory=False)
        df['sport'] = sport
        keep = ['country_code', 'year', 'sport', 'name', 'school_name',
                'gender', 'hometown']
        parts.append(df[[c for c in keep if c in df.columns]])
    stacked = pd.concat(parts, ignore_index=True)
    print(f'Stacked athlete rows: {len(stacked):,}')

    # Drop rows with no country (we still want them counted somewhere — they
    # are athletes with unparseable hometown — but they can't enter the
    # country panel). Keep for descriptive only.
    panel_rows = stacked.dropna(subset=['country_code', 'year']).copy()
    panel_rows['year'] = panel_rows['year'].astype(int)
    print(f'Rows with country code and year: {len(panel_rows):,}')

    # ---- (2) Aggregate to (country, year, sport) -------------------------
    counts = (panel_rows.groupby(['country_code', 'year', 'sport'])
                        .size().rename('athletes').reset_index())

    # Build the full balanced grid: every (country in panel) × year × sport
    countries = panel_rows['country_code'].unique()
    grid = pd.MultiIndex.from_product([countries, YEARS, list(SPORT_FILES)],
                                       names=['country_code', 'year', 'sport']
                                       ).to_frame(index=False)
    panel = grid.merge(counts, on=['country_code', 'year', 'sport'], how='left')
    panel['athletes'] = panel['athletes'].fillna(0).astype(int)
    panel['log_athletes'] = np.log1p(panel['athletes'])
    panel['is_present']   = (panel['athletes'] > 0).astype(int)
    print(f'Balanced country×year×sport panel: {len(panel):,} cells '
          f'({len(countries)} countries × {len(YEARS)} years × {len(SPORT_FILES)} sports)')

    # ---- (3) Merge time-varying macro covariates -------------------------
    macro = pd.read_csv(os.path.join(RAW_DATA, 'macro_combined.csv'))
    macro = macro[macro['year'].between(YEARS[0], YEARS[-1])][[
        'country_code', 'year', 'gdp_per_capita_ppp', 'gdp_growth',
        'unemployment'
    ]]

    polstab = pd.read_csv(os.path.join(RAW_DATA, 'political_stability.csv'))
    polstab = polstab.rename(columns={'iso_code': 'country_code'})
    polstab = polstab[polstab['year'].between(YEARS[0], YEARS[-1])][[
        'country_code', 'year', 'political_stability', 'rule_of_law',
        'control_corruption'
    ]]

    ms = pd.read_csv(os.path.join(RAW_DATA, 'macro_shocks.csv'))
    ms = ms[ms['year'].between(YEARS[0], YEARS[-1])][[
        'country_code', 'year', 'gdp_shock', 'currency_crisis',
        'unemployment_spike', 'any_econ_shock'
    ]]

    nd = pd.read_csv(os.path.join(RAW_DATA, 'natural_disasters.csv'))
    nd = nd.rename(columns={'iso_code': 'country_code'})
    nd = nd[nd['year'].between(YEARS[0], YEARS[-1])][[
        'country_code', 'year', 'n_events', 'deaths', 'total_affected'
    ]].rename(columns={'n_events': 'disaster_events',
                        'deaths': 'disaster_deaths',
                        'total_affected': 'disaster_affected'})

    cf = pd.read_csv(os.path.join(RAW_DATA, 'ucdp_conflict.csv'))
    cf = cf.rename(columns={'iso3': 'country_code'})
    cf = cf[cf['year'].between(YEARS[0], YEARS[-1])][[
        'country_code', 'year', 'any_conflict', 'n_conflicts'
    ]]

    cohort = pd.read_csv(os.path.join(RAW_DATA, 'wdi_population_15_24.csv'))
    # cohort is a time-invariant 2018-22 avg

    for d in (macro, polstab, ms, nd, cf):
        panel = panel.merge(d, on=['country_code', 'year'], how='left')

    panel = panel.merge(cohort, on='country_code', how='left')
    panel['log_pop_15_24'] = np.log(panel['pop_15_24_avg_2018_22'])
    panel['log_gdp_pc']    = np.log(panel['gdp_per_capita_ppp'])
    panel['log_disaster_events'] = np.log1p(panel['disaster_events'].fillna(0))
    panel['log_disaster_deaths'] = np.log1p(panel['disaster_deaths'].fillna(0))

    # Fill 0 for conflict/disaster missing rows (means "no event reported")
    for c in ['any_conflict', 'n_conflicts', 'disaster_events', 'disaster_deaths',
              'gdp_shock', 'currency_crisis', 'unemployment_spike', 'any_econ_shock']:
        panel[c] = panel[c].fillna(0)

    # COVID treatment indicator: 1 for sport-years where soccer or volleyball
    # were disrupted (the 2020 cohort had a +25% / +16% drop)
    panel['treated_sport'] = panel['sport'].isin(['soccer', 'volleyball']).astype(int)
    panel['post_2020']     = (panel['year'] >= 2020).astype(int)
    panel['did_term']      = panel['treated_sport'] * panel['post_2020']
    panel['treat_x_y2020'] = panel['treated_sport'] * (panel['year'] == 2020).astype(int)
    panel['treat_x_y2021'] = panel['treated_sport'] * (panel['year'] == 2021).astype(int)
    panel['treat_x_y2022'] = panel['treated_sport'] * (panel['year'] == 2022).astype(int)
    panel['treat_x_y2023'] = panel['treated_sport'] * (panel['year'] == 2023).astype(int)

    # Drop countries with effectively no time-varying covariates available
    needed = ['log_gdp_pc', 'political_stability']
    has_full = panel.groupby('country_code')[needed].apply(
        lambda g: g.notna().all().all())
    keep = has_full[has_full].index
    panel = panel[panel['country_code'].isin(keep)].copy()
    print(f'After dropping countries missing full macro coverage: {len(panel):,} rows')
    print(f'  countries: {panel["country_code"].nunique()}')

    out = os.path.join(RAW_DATA, 'country_year_sport_panel.csv')
    panel.to_csv(out, index=False)
    print(f'\nWrote {out}: {panel.shape}')

    # Quick descriptive
    print('\n=== Athletes per sport-year (aggregate over all countries) ===')
    agg = panel.groupby(['sport', 'year'])['athletes'].sum().unstack(fill_value=0)
    print(agg.to_string())
    print('\n=== Sport-specific 2019-vs-2020 (all-country sums) ===')
    delta = ((agg[2020] - agg[2019]) / agg[2019] * 100).round(1)
    for sport in agg.index:
        print(f'  {sport:<12} 2019={agg.loc[sport,2019]:>5,}  '
              f'2020={agg.loc[sport,2020]:>5,}  Δ={delta[sport]:+.1f}%')


if __name__ == '__main__':
    main()
