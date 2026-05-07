"""
Join the five exogenous shock panels onto analysis_dataset.csv via the
athlete's home country_code.

For each shock file, we add (a) a 2023 snapshot of the variables and
(b) recent-period aggregates over 2010-2023 (or the relevant years)
that capture the cumulative environment the athlete grew up in.

Updates raw_data/analysis_dataset.csv in place.
"""
from __future__ import annotations

import os
import warnings
import pandas as pd

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
ANALYSIS = os.path.join(RAW_DATA, 'analysis_dataset.csv')


def load_shock(filename: str, iso_col_candidates: list[str]) -> pd.DataFrame:
    """Load a shock CSV and rename its ISO column to country_code."""
    df = pd.read_csv(os.path.join(RAW_DATA, filename))
    for c in iso_col_candidates:
        if c in df.columns:
            if c != 'country_code':
                df = df.rename(columns={c: 'country_code'})
            return df
    raise KeyError(f"No ISO column in {filename}; tried {iso_col_candidates}")


def latest_year(df: pd.DataFrame, target_year: int = 2023) -> pd.DataFrame:
    """Return one row per country, preferring `target_year` then most recent."""
    df = df.dropna(subset=['country_code', 'year']).copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['_pref'] = (df['year'] == target_year).astype(int)
    df = df.sort_values(['country_code', '_pref', 'year'], ascending=[True, False, False])
    out = df.drop_duplicates('country_code', keep='first').drop(columns=['_pref'])
    return out


def main():
    warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
    df = pd.read_csv(ANALYSIS, low_memory=False)
    print(f"Loaded analysis_dataset: {df.shape}")
    cols_before = set(df.columns)

    # ---------- 1. Macro shocks (binary indicators) ----------
    ms = load_shock('macro_shocks.csv', ['country_code'])
    # Keep only the binary/composite columns to avoid duplicating gdp_per_capita_ppp etc.
    keep_ms = ['country_code', 'gdp_growth_zscore', 'gdp_shock',
               'exchange_rate_change', 'currency_crisis', 'currency_crisis_severe',
               'unemployment_change', 'unemployment_spike',
               'ppp_change', 'ppp_shock', 'econ_shock_index', 'any_econ_shock']
    ms_2023 = latest_year(ms[keep_ms + ['year']], 2023).drop(columns='year')
    df = df.merge(ms_2023, on='country_code', how='left', suffixes=('', '_msshock'))
    print(f"  + macro_shocks 2023: now {df.shape[1]} cols")

    # Recent-period aggregates: count of any_econ_shock and currency_crisis 2010-2023
    ms_agg = (ms[(ms['year'] >= 2010) & (ms['year'] <= 2023)]
                .groupby('country_code')
                .agg(econ_shocks_2010_23=('any_econ_shock', 'sum'),
                     currency_crises_2010_23=('currency_crisis', 'sum'),
                     gdp_shocks_2010_23=('gdp_shock', 'sum'))
                .reset_index())
    df = df.merge(ms_agg, on='country_code', how='left')
    print(f"  + macro_shocks 2010-23 aggregates: now {df.shape[1]} cols")

    # ---------- 2. UCDP conflict ----------
    cf = load_shock('ucdp_conflict.csv', ['iso3'])
    cf_2023 = latest_year(cf[['country_code', 'year', 'any_conflict', 'n_conflicts',
                              'max_intensity_level']], 2023)
    cf_2023 = cf_2023.rename(columns={
        'any_conflict':      'conflict_2023',
        'n_conflicts':       'n_conflicts_2023',
        'max_intensity_level': 'conflict_intensity_2023',
    }).drop(columns='year')
    df = df.merge(cf_2023, on='country_code', how='left')
    # UCDP missing rows = no conflict
    df['conflict_2023'] = df['conflict_2023'].fillna(0)
    df['n_conflicts_2023'] = df['n_conflicts_2023'].fillna(0)
    df['conflict_intensity_2023'] = df['conflict_intensity_2023'].fillna(0)

    cf_agg = (cf[(cf['year'] >= 2010) & (cf['year'] <= 2023)]
                .groupby('country_code')
                .agg(years_with_conflict_2010_23=('any_conflict', 'sum'),
                     conflict_onsets_2010_23=('any_onset', 'sum'))
                .reset_index())
    df = df.merge(cf_agg, on='country_code', how='left')
    df['years_with_conflict_2010_23'] = df['years_with_conflict_2010_23'].fillna(0)
    df['conflict_onsets_2010_23'] = df['conflict_onsets_2010_23'].fillna(0)
    print(f"  + UCDP conflict (2023 + 2010-23 aggregates): now {df.shape[1]} cols")

    # ---------- 3. Natural disasters ----------
    nd = load_shock('natural_disasters.csv', ['iso_code'])
    nd_2023 = latest_year(nd[['country_code', 'year', 'n_events', 'deaths',
                              'total_affected']], 2023)
    nd_2023 = nd_2023.rename(columns={
        'n_events':        'disaster_events_2023',
        'deaths':          'disaster_deaths_2023',
        'total_affected':  'disaster_affected_2023',
    }).drop(columns='year')
    df = df.merge(nd_2023, on='country_code', how='left')

    nd_agg = (nd[(nd['year'] >= 2010) & (nd['year'] <= 2023)]
                .groupby('country_code')
                .agg(disaster_deaths_2010_23=('deaths', 'sum'),
                     disaster_events_2010_23=('n_events', 'sum'))
                .reset_index())
    df = df.merge(nd_agg, on='country_code', how='left')
    df['disaster_deaths_2010_23'] = df['disaster_deaths_2010_23'].fillna(0)
    df['disaster_events_2010_23'] = df['disaster_events_2010_23'].fillna(0)
    print(f"  + Natural disasters (2023 + 2010-23 aggregates): now {df.shape[1]} cols")

    # ---------- 4. COVID ----------
    cv = load_shock('covid_by_country_year.csv', ['iso_code'])
    # Use peak-stringency aggregate (mean over 2020-2021) and 2023 cumulative cases/deaths.
    cv_peak = (cv[cv['year'].isin([2020, 2021])]
                .groupby('country_code')
                .agg(stringency_2020_21=('stringency_index', 'mean'))
                .reset_index())
    df = df.merge(cv_peak, on='country_code', how='left')
    cv_latest = latest_year(cv[['country_code', 'year', 'total_cases_per_million',
                                'total_deaths_per_million']], 2023)
    cv_latest = cv_latest.rename(columns={
        'total_cases_per_million':  'covid_cases_per_million',
        'total_deaths_per_million': 'covid_deaths_per_million',
    }).drop(columns='year')
    df = df.merge(cv_latest, on='country_code', how='left')
    print(f"  + COVID (peak stringency 2020-21 + cumulative 2023): now {df.shape[1]} cols")

    # ---------- 5. Political stability / WGI ----------
    ps = load_shock('political_stability.csv', ['iso_code'])
    ps_2023 = latest_year(ps[['country_code', 'year', 'political_stability',
                              'govt_effectiveness', 'rule_of_law',
                              'control_corruption']], 2023)
    ps_2023 = ps_2023.drop(columns='year')
    df = df.merge(ps_2023, on='country_code', how='left')
    print(f"  + Political stability 2023: now {df.shape[1]} cols")

    # ---------- Save ----------
    df.to_csv(ANALYSIS, index=False)
    new_cols = sorted(set(df.columns) - cols_before)
    print(f"\nSaved {ANALYSIS}")
    print(f"  shape: {df.shape}")
    print(f"  added {len(new_cols)} columns:")
    for c in new_cols:
        print(f"    {c}")

    # ---------- Quick coverage check ----------
    print("\n=== Coverage on international athletes ===")
    intl = df[df['is_international'] == 1]
    for c in new_cols:
        if intl[c].dtype != 'O':
            cov = intl[c].notna().sum()
            pct = 100 * cov / max(len(intl), 1)
            print(f"  {c:35s} {cov:>5}/{len(intl)}  ({pct:.1f}%)")


if __name__ == '__main__':
    main()
