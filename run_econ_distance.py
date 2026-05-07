"""
Compute economic-distance metrics for each athlete relative to the US 2023
baseline, then save back to raw_data/analysis_dataset.csv. Also writes
descriptive stats and plots to output/.

Metrics added:
  is_international          1 if non-USA, 0 if USA, NaN if no country
  gdp_ratio                 home_gdp_pc_ppp / us_gdp_pc_ppp
  log_gdp_ratio             ln(gdp_ratio)
  gdp_distance              us_gdp_pc_ppp - home_gdp_pc_ppp (absolute $)
  unemployment_diff         home_unemployment - us_unemployment (pp)
  price_level_relative      home_price_level / us_price_level
"""
from __future__ import annotations

import os
import math
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')
os.makedirs(OUTPUT, exist_ok=True)


def main():
    warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
    df = pd.read_csv(os.path.join(RAW_DATA, 'analysis_dataset.csv'),
                     low_memory=False)
    print(f"Loaded analysis dataset: {df.shape}")

    # US 2023 baseline from the row(s) where country_code == 'USA'
    macro = pd.read_csv(os.path.join(RAW_DATA, 'macro_combined.csv'))
    us = macro[(macro['country_code'] == 'USA') & (macro['year'] == 2023)].iloc[0]
    us_gdp = us['gdp_per_capita_ppp']
    us_unemp = us['unemployment']
    us_plr = us['price_level_ratio'] if 'price_level_ratio' in us else 1.0
    print(f"\nUS 2023 baseline: GDP/cap PPP = ${us_gdp:,.0f}, unemployment = {us_unemp:.2f}%, "
          f"price-level ratio = {us_plr:.3f}")

    # is_international
    df['is_international'] = df['country_code'].apply(
        lambda c: np.nan if pd.isna(c) else int(c != 'USA')
    )

    # GDP-based metrics
    df['gdp_ratio']      = df['gdp_per_capita_ppp'] / us_gdp
    df['gdp_distance']   = us_gdp - df['gdp_per_capita_ppp']
    with np.errstate(divide='ignore', invalid='ignore'):
        df['log_gdp_ratio'] = np.log(df['gdp_ratio'].where(df['gdp_ratio'] > 0))

    # Unemployment + price level
    df['unemployment_diff'] = df['unemployment'] - us_unemp
    if 'price_level_ratio' in df.columns:
        df['price_level_relative'] = df['price_level_ratio'] / us_plr

    df.to_csv(os.path.join(RAW_DATA, 'analysis_dataset.csv'), index=False)
    print(f"\nSaved updated analysis_dataset.csv (now {df.shape[1]} columns)")

    # ----------------- descriptive stats -----------------
    print("\n=== Descriptive statistics ===\n")

    n_total      = len(df)
    n_country    = df['country_code'].notna().sum()
    n_intl       = (df['is_international'] == 1).sum()
    n_intl_pct   = 100 * n_intl / max(n_country, 1)
    print(f"Total athletes:        {n_total:,}")
    print(f"With country_code:     {n_country:,} ({100*n_country/n_total:.1f}%)")
    print(f"International:         {n_intl:,} ({n_intl_pct:.1f}% of those with country)")

    # Sport × international
    print("\nSport × international:")
    print(pd.crosstab(df['sport'], df['is_international'].fillna('unknown'),
                      margins=True, margins_name='Total').to_string())

    # International rate by NCAA conference (top 15 conferences by international count)
    if 'ncaa_conference' in df.columns:
        conf = (df.dropna(subset=['ncaa_conference'])
                  .groupby('ncaa_conference')
                  .agg(athletes=('school_name', 'count'),
                       intl=('is_international', 'sum'))
                  .assign(intl_pct=lambda x: 100 * x['intl'] / x['athletes'])
                  .sort_values('intl', ascending=False)
                  .head(15))
        print("\nTop 15 conferences by international athlete count:")
        print(conf.to_string())

    # Mean / median GDP per capita PPP for international athletes' home countries
    intl = df[df['is_international'] == 1].dropna(subset=['gdp_per_capita_ppp'])
    print("\nInternational athletes' home-country economics (n =", len(intl), "):")
    print(f"  Mean home GDP/cap PPP:     ${intl['gdp_per_capita_ppp'].mean():,.0f}")
    print(f"  Median home GDP/cap PPP:   ${intl['gdp_per_capita_ppp'].median():,.0f}")
    print(f"  US 2023 baseline:          ${us_gdp:,.0f}")
    print(f"  Mean GDP ratio (home/US):  {intl['gdp_ratio'].mean():.3f}")
    print(f"  Median GDP ratio:          {intl['gdp_ratio'].median():.3f}")
    print(f"  Mean log GDP ratio:        {intl['log_gdp_ratio'].mean():.3f}")

    # Top countries with their GDP and athlete counts
    print("\nTop 25 countries with athletes and home GDP (international only):")
    top25 = (intl.groupby('country_code')
                 .agg(athletes=('school_name', 'count'),
                      gdp_per_capita_ppp=('gdp_per_capita_ppp', 'first'),
                      gdp_ratio=('gdp_ratio', 'first'))
                 .sort_values('athletes', ascending=False)
                 .head(25))
    print(top25.round(3).to_string())

    # Save the country-level summary
    top25.to_csv(os.path.join(OUTPUT, 'top_countries_summary.csv'))

    # ----------------- plots -----------------
    print("\n=== Plots ===")

    # Plot 1: histogram of home-country GDP for international athletes
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(intl['gdp_per_capita_ppp'].dropna(), bins=40, color='steelblue', edgecolor='white')
    ax.axvline(us_gdp, color='red', linestyle='--', label=f'US 2023 (${us_gdp:,.0f})')
    ax.set_xlabel('Home country GDP per capita, PPP (constant intl $, 2023)')
    ax.set_ylabel('International athletes')
    ax.set_title('Where international NCAA athletes come from, by home-country GDP')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'fig_intl_home_gdp_hist.png'), dpi=150)
    plt.close(fig)
    print(f"  wrote {os.path.join(OUTPUT, 'fig_intl_home_gdp_hist.png')}")

    # Plot 2: top 20 source countries (bar chart)
    top20 = top25.head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top20.index[::-1], top20['athletes'][::-1], color='steelblue')
    ax.set_xlabel('Number of athletes')
    ax.set_title(f"Top 20 source countries for international NCAA athletes (n={len(intl)})")
    for b, v in zip(bars, top20['athletes'][::-1]):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, str(v), va='center')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'fig_top_countries_bar.png'), dpi=150)
    plt.close(fig)
    print(f"  wrote {os.path.join(OUTPUT, 'fig_top_countries_bar.png')}")

    # Plot 3: scatter of country GDP vs # athletes (one dot per country, log scale)
    cy = intl.groupby('country_code').agg(
        athletes=('school_name', 'count'),
        gdp=('gdp_per_capita_ppp', 'first')).dropna()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(cy['gdp'], cy['athletes'], alpha=0.65, s=40, color='steelblue', edgecolor='white')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('GDP per capita PPP (log scale)')
    ax.set_ylabel('Athletes (log scale)')
    ax.set_title('NCAA international athletes vs home-country GDP per capita')
    # Annotate top countries
    for code, row in cy.sort_values('athletes', ascending=False).head(15).iterrows():
        ax.annotate(code, (row['gdp'], row['athletes']),
                    xytext=(4, 4), textcoords='offset points', fontsize=8)
    ax.axvline(us_gdp, color='red', linestyle='--', alpha=0.5, label='US 2023')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'fig_athletes_vs_gdp_scatter.png'), dpi=150)
    plt.close(fig)
    print(f"  wrote {os.path.join(OUTPUT, 'fig_athletes_vs_gdp_scatter.png')}")

    # Plot 4: log-GDP-ratio histogram (econ-distance density)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(intl['log_gdp_ratio'].dropna(), bins=40, color='darkorange', edgecolor='white')
    ax.axvline(0, color='red', linestyle='--', label='Parity with US')
    ax.set_xlabel('log(home GDP / US GDP) — negative = poorer than US')
    ax.set_ylabel('International athletes')
    ax.set_title('Econ-distance distribution for international NCAA athletes')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'fig_log_gdp_ratio_hist.png'), dpi=150)
    plt.close(fig)
    print(f"  wrote {os.path.join(OUTPUT, 'fig_log_gdp_ratio_hist.png')}")


if __name__ == '__main__':
    main()
