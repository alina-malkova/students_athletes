"""
Country x entry-cohort pseudo-panel from class_year, then COVID event study.

Roster snapshots are 2024-25/2025-26 single-period. We back out an
implied freshman year from each athlete's class_year:

  Fr -> 2024, So -> 2023, Jr -> 2022, Sr -> 2021, Gr -> 2020
  R-Fr -> 2023, R-So -> 2022, R-Jr -> 2021, R-Sr -> 2020
  Freshman/Sophomore/... -> same as abbreviations

Athletes whose class_year doesn't parse cleanly are dropped (~5%).

This gives a country x entry-year panel for 2020-2024 (5 cohorts).
We then identify the COVID effect:

  log(1 + entrants_{c,t}) = alpha_c + gamma_t
                            + sum_{k=2020}^{2024} beta_k 1{t=k} x StringencyZ_c
                            + epsilon_{c,t}

Treatment intensity StringencyZ_c is the country's standardized
stringency 2020-21 (z-score across countries). Cohort 2020 is the
omitted reference (decisions made in 2019 = pre-COVID). Coefficient
beta_k traces the effect on entrants whose recruitment decisions were
made k-1 years out.

Outputs: output/event_study.txt, output/fig_event_study.png
"""
from __future__ import annotations

import os
import re
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, 'raw_data')
OUTPUT   = os.path.join(HERE, 'output')


CLASS_TO_OFFSET = {
    'fr':   0, 'freshman': 0, 'fy': 0, 'first year': 0, '1l': 0,
    'so':   1, 'sophomore': 1, '2l': 1, 'second year': 1,
    'jr':   2, 'junior':    2, '3l': 2, 'third year': 2,
    'sr':   3, 'senior':    3, '4l': 3, 'fourth year': 3,
    'gr':   4, 'graduate':  4, 'graduate student': 4,
    '5th':  4, '5th-year senior': 4, '5th year': 4,
    'r-fr':  1, 'redshirt freshman':   1,
    'r-so':  2, 'redshirt sophomore':  2,
    'r-jr':  3, 'redshirt junior':     3,
    'r-sr':  4, 'redshirt senior':     4,
    'r sr.': 4,
}


def parse_class_year(s) -> int | None:
    """Return years-since-freshman offset (Fr=0, So=1, ..., Gr=4); None if unparseable."""
    if not isinstance(s, str) or not s.strip():
        return None
    raw = s.strip().lower().rstrip('.').replace('.', '').strip()
    raw = re.sub(r'\s+', ' ', raw)
    # "fr./fr." or "sr./sr." -- collapse
    if '/' in raw:
        raw = raw.split('/')[0].strip()
    if raw in CLASS_TO_OFFSET:
        return CLASS_TO_OFFSET[raw]
    # Try first 2 chars: "fr" "so" "jr" "sr" "gr"
    head = raw.replace(' ', '')[:2]
    if head in CLASS_TO_OFFSET:
        return CLASS_TO_OFFSET[head]
    return None


def main():
    warnings.filterwarnings('ignore')
    df = pd.read_csv(os.path.join(RAW_DATA, 'analysis_dataset.csv'),
                     low_memory=False)
    intl = df[(df['is_international'] == 1) & df['country_code'].notna()].copy()
    print(f'International athletes: {len(intl)}')

    # Most rosters are 2024-25 season; assume current = 2024.
    CURRENT_YEAR = 2024
    intl['year_offset'] = intl['class_year'].apply(parse_class_year)
    intl['entry_year'] = CURRENT_YEAR - intl['year_offset']
    parsed = intl.dropna(subset=['entry_year'])
    print(f'  with parseable class_year: {len(parsed)} '
          f'({100 * len(parsed) / len(intl):.1f}%)')
    parsed['entry_year'] = parsed['entry_year'].astype(int)
    print('\nEntry-year distribution:')
    print(parsed['entry_year'].value_counts().sort_index().to_string())

    # ------- Country x entry-year panel -------
    # Use country-level covid stringency from analysis_dataset (already at
    # athlete level; just take first value per country).
    country_chars = (parsed.groupby('country_code')
                            .agg(stringency_2020_21=('stringency_2020_21', 'first'),
                                 pop_15_24=('pop_15_24_avg_2018_22', 'first'),
                                 gdp_per_capita_ppp=('gdp_per_capita_ppp', 'first'),
                                 political_stability=('political_stability', 'first'))
                            .reset_index())

    # Build a balanced panel for entry_year 2020..2024 over countries with
    # at least one athlete in any of those years AND stringency observed.
    years = list(range(2020, 2025))
    countries = country_chars.dropna(subset=['stringency_2020_21'])['country_code'].tolist()
    panel = pd.MultiIndex.from_product([countries, years],
                                        names=['country_code', 'entry_year']).to_frame(index=False)
    counts = (parsed.groupby(['country_code', 'entry_year'])
                     .size().rename('athletes_entering').reset_index())
    panel = panel.merge(counts, on=['country_code', 'entry_year'], how='left')
    panel['athletes_entering'] = panel['athletes_entering'].fillna(0).astype(int)
    panel = panel.merge(country_chars, on='country_code', how='left')
    panel['log_entrants'] = np.log1p(panel['athletes_entering'])

    print(f'\nCountry x entry-year panel: {len(panel)} cells '
          f'({len(countries)} countries x {len(years)} cohorts)')
    print(f'  cells with at least one entrant: {(panel["athletes_entering"]>0).sum()}')

    # Standardize stringency (z-score across countries)
    s = country_chars['stringency_2020_21']
    sm_, ss_ = s.mean(), s.std()
    panel['stringency_z'] = (panel['stringency_2020_21'] - sm_) / ss_

    panel.to_csv(os.path.join(OUTPUT, 'event_study_panel.csv'), index=False)

    fh = open(os.path.join(OUTPUT, 'event_study.txt'), 'w')

    # ------- (1) Static DID: high vs low stringency, treated cohorts 2021-22 -------
    fh.write('NCAA international athlete pseudo-panel: COVID event study\n')
    fh.write(f'  N countries:      {len(countries)}\n')
    fh.write(f'  N cohorts:        {len(years)}\n')
    fh.write(f'  N panel cells:    {len(panel)}\n\n')

    panel['post_covid'] = panel['entry_year'].isin([2021, 2022]).astype(int)
    panel['treat_x_post'] = panel['stringency_z'] * panel['post_covid']

    # Build country and year fixed effects via dummies (drop one each)
    cdums = pd.get_dummies(panel['country_code'], prefix='c', drop_first=True).astype(float)
    ydums = pd.get_dummies(panel['entry_year'].astype(int), prefix='y', drop_first=True).astype(float)
    X = pd.concat([panel[['treat_x_post']], cdums, ydums], axis=1)
    X = sm.add_constant(X)
    sub = panel.dropna(subset=['stringency_z']).copy()
    Xfit = X.loc[sub.index]
    yfit = sub['log_entrants']
    m_static = sm.OLS(yfit, Xfit).fit(cov_type='cluster', cov_kwds={'groups': sub['country_code']})
    fh.write('=== Static DID: Stringency_z x 1{entry_year in 2021,2022} ===\n')
    fh.write(f'(Country + year FE; SE clustered at country)\n')
    fh.write(f'  treat_x_post coef = {m_static.params["treat_x_post"]:+.3f}\n')
    fh.write(f'  SE              = {m_static.bse["treat_x_post"]:.3f}\n')
    fh.write(f'  t               = {m_static.tvalues["treat_x_post"]:.2f}\n')
    fh.write(f'  p               = {m_static.pvalues["treat_x_post"]:.3f}\n\n')
    print('\n=== Static DID ===')
    print(f'  Stringency_z x post(2021-22) coef = {m_static.params["treat_x_post"]:+.3f}'
          f'  (SE {m_static.bse["treat_x_post"]:.3f},'
          f'  t={m_static.tvalues["treat_x_post"]:.2f},'
          f'  p={m_static.pvalues["treat_x_post"]:.3f})')

    # ------- (2) Dynamic event study: stringency_z x year dummies, base 2020 -------
    panel['stringency_z'] = (panel['stringency_2020_21'] - sm_) / ss_
    sub2 = panel.dropna(subset=['stringency_z']).copy()
    interactions = {}
    for y in years:
        if y == 2020:
            continue  # omitted base
        col = f'sZ_x_y{y}'
        interactions[col] = sub2['stringency_z'] * (sub2['entry_year'] == y).astype(int)
    sub2 = pd.concat([sub2, pd.DataFrame(interactions, index=sub2.index)], axis=1)
    cdums2 = pd.get_dummies(sub2['country_code'], prefix='c', drop_first=True).astype(float)
    ydums2 = pd.get_dummies(sub2['entry_year'].astype(int), prefix='y', drop_first=True).astype(float)
    X2 = pd.concat([sub2[list(interactions.keys())], cdums2, ydums2], axis=1)
    X2 = sm.add_constant(X2)
    m_dyn = sm.OLS(sub2['log_entrants'], X2).fit(
        cov_type='cluster', cov_kwds={'groups': sub2['country_code']})
    fh.write('=== Dynamic event study: Stringency_z x cohort-year ===\n')
    fh.write('(Base year = 2020; SE clustered at country)\n')
    print('\n=== Dynamic event study (base = 2020) ===')
    coefs, ses = {}, {}
    for col in interactions:
        b  = m_dyn.params[col]
        se = m_dyn.bse[col]
        p  = m_dyn.pvalues[col]
        stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
        fh.write(f'  {col}: {b:+.3f}  (SE {se:.3f}, p={p:.3f}) {stars}\n')
        print(f'  {col}: {b:+.3f}  (SE {se:.3f}, t={b/se:5.2f}) {stars}')
        coefs[col] = b
        ses[col]   = se
    fh.write('\n')
    fh.close()

    # ------- Plot leads/lags -------
    plot_years = sorted(years)
    pe = []
    pl = []
    pu = []
    for y in plot_years:
        if y == 2020:
            pe.append(0.0); pl.append(0.0); pu.append(0.0)
        else:
            col = f'sZ_x_y{y}'
            b  = coefs[col]
            se = ses[col]
            pe.append(b); pl.append(b - 1.96 * se); pu.append(b + 1.96 * se)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(plot_years, pe,
                yerr=[[pe[i] - pl[i] for i in range(len(pe))],
                      [pu[i] - pe[i] for i in range(len(pe))]],
                fmt='o-', color='steelblue', capsize=4, markersize=8)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.axvspan(2020.5, 2022.5, color='red', alpha=0.10,
               label='Recruitment decisions during COVID (2020-21)')
    ax.set_xticks(plot_years)
    ax.set_xlabel('Cohort entry year')
    ax.set_ylabel('Stringency$_z$ × 1{entry_year=t} coefficient')
    ax.set_title('Event study: NCAA international entrants\n'
                 'by home-country COVID stringency, base year 2020')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'fig_event_study.png'), dpi=150)
    plt.close(fig)
    print(f'\nWrote {os.path.join(OUTPUT, "fig_event_study.png")}')
    print(f'Wrote {os.path.join(OUTPUT, "event_study.txt")}')
    print(f'Wrote {os.path.join(OUTPUT, "event_study_panel.csv")}')


if __name__ == '__main__':
    main()
