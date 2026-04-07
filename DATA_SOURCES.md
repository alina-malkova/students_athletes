# Data Sources — Exogenous Shock Variables

This document describes all data sources used to construct exogenous shock variables for the Student Athletes Economic Distance project. All files are stored in `raw_data/`.

---

## 1. Macroeconomic Shocks (constructed from WDI)

**File:** `macro_shocks.csv`
**Source:** World Bank World Development Indicators (WDI), downloaded via `wbgapi` Python package
**Coverage:** 214 countries, 2010-2023 (annual)
**Unit of observation:** Country-year

### Variables

| Variable | Description | Construction |
|----------|-------------|-------------|
| `gdp_growth_num` | GDP per capita growth rate (%) | WDI indicator `NY.GDP.PCAP.KD.ZG` |
| `gdp_growth_zscore` | GDP growth deviation from country mean, in standard deviations | `(growth - country_mean) / country_sd` |
| `gdp_shock` | Binary: GDP growth recession shock | `1` if growth < 0 AND growth < country_mean - 1.5*SD |
| `exchange_rate_num` | Official exchange rate (LCU per US$) | WDI indicator `PA.NUS.FCRF` |
| `exchange_rate_change` | Year-over-year exchange rate change (%) | `(rate_t - rate_{t-1}) / rate_{t-1}` |
| `currency_crisis` | Binary: currency depreciation > 15% | `1` if exchange_rate_change > 0.15 |
| `currency_crisis_severe` | Binary: currency depreciation > 30% | `1` if exchange_rate_change > 0.30 |
| `unemployment_num` | Unemployment rate (%) | WDI indicator `SL.UEM.TOTL.ZS` |
| `unemployment_change` | Year-over-year change in unemployment (pp) | `rate_t - rate_{t-1}` |
| `unemployment_spike` | Binary: unemployment increase > 3pp | `1` if unemployment_change > 3 |
| `ppp_num` | PPP conversion factor (LCU per intl $) | WDI indicator `PA.NUS.PPP` |
| `ppp_change` | Year-over-year PPP change (%) | `(ppp_t - ppp_{t-1}) / ppp_{t-1}` |
| `ppp_shock` | Binary: PPP increase > 20% | `1` if ppp_change > 0.20 |
| `gdp_per_capita_ppp` | GDP per capita, PPP (constant intl $) | WDI indicator `NY.GDP.PCAP.PP.KD` |
| `price_level_ratio` | Price level ratio of PPP to exchange rate | WDI indicator `PA.NUS.PPPC.RF` |
| `econ_shock_index` | Composite: sum of 4 binary shock indicators (0-4) | `gdp_shock + currency_crisis + unemployment_spike + ppp_shock` |
| `any_econ_shock` | Binary: any economic shock occurred | `1` if econ_shock_index > 0 |

### Shock Frequencies (2010-2023)

| Shock | Country-years | % |
|-------|--------------|---|
| GDP shock | 203 | 6.8% |
| Currency crisis (>15%) | 262 | 8.7% |
| Currency crisis severe (>30%) | 92 | 3.1% |
| Unemployment spike (>3pp) | 28 | 0.9% |
| PPP shock (>20%) | 98 | 3.3% |
| Any economic shock | 492 | 16.4% |

### Citation
World Bank. "World Development Indicators." Washington, D.C.: The World Bank Group. Accessed April 2026.

---

## 2. Armed Conflict

**File:** `ucdp_conflict.csv`
**Source:** Uppsala Conflict Data Program (UCDP) / Peace Research Institute Oslo (PRIO) Armed Conflict Dataset v24.1
**Coverage:** 119 countries with conflicts, 1946-2023
**Unit of observation:** Country-year
**API:** `https://ucdpapi.pcr.uu.se/api/ucdpprio/24.1`

### Variables

| Variable | Description | Values |
|----------|-------------|--------|
| `country_name` | Primary country of conflict | Text |
| `iso3` | ISO 3166-1 alpha-3 country code | 3-letter code |
| `year` | Year of observation | 1946-2023 |
| `any_conflict` | Binary: any active armed conflict | 0/1 |
| `n_conflicts` | Number of active conflicts | Integer |
| `max_intensity_level` | Highest intensity of any conflict | 1 = minor (25-999 battle deaths/yr), 2 = war (1000+ deaths/yr) |
| `max_intensity_label` | Text label for intensity | "minor" or "war" |
| `conflict_types` | Types of conflict active | "interstate", "intrastate", "internationalized intrastate", "extrasystemic" |
| `any_onset` | Binary: new conflict started this year | 0/1 |
| `incompatibilities` | What is being fought over | "government", "territory", or both |
| `conflict_ids` | UCDP conflict identifiers | Semicolon-separated IDs |

### Notes
- Countries with **no entry** in a given year had **no active conflict** — these are implicit zeros.
- When merging with roster data, non-matched country-years should be coded as `any_conflict = 0`.
- Conflict type definitions:
  - **Interstate:** Between two or more states
  - **Intrastate:** Between government and non-state group
  - **Internationalized intrastate:** Intrastate with foreign state intervention
  - **Extrasystemic:** Between state and non-state group outside its territory (colonial wars)

### Citation
Gleditsch, Nils Petter; Peter Wallensteen, Mikael Eriksson, Margareta Sollenberg & Havard Strand (2002) Armed Conflict 1946–2001: A New Dataset. *Journal of Peace Research* 39(5): 615–637.

Davies, Shawn; Therese Pettersson & Magnus Oberg (2024). Organized violence 1989-2023, and the return of conflict between states. *Journal of Peace Research* 61(4).

---

## 3. Natural Disasters

**File:** `natural_disasters.csv`
**Source:** Our World in Data / EM-DAT (Emergency Events Database), International Disaster Database, CRED, UCLouvain
**Coverage:** ~200 countries, 2000-2023
**Unit of observation:** Country-year

### Variables

| Variable | Description |
|----------|-------------|
| `country` | Country name |
| `iso_code` | ISO 3166-1 alpha-3 country code |
| `year` | Year of observation |
| `death_rate_per_100k` | Deaths from natural disasters per 100,000 population |
| `deaths` | Total deaths from natural disasters |

### Disaster Types Included
Geophysical (earthquakes, volcanic activity), meteorological (storms, extreme temperature), hydrological (floods, landslides), climatological (droughts, wildfires), biological (epidemics).

### Notes
- Death counts include both direct and indirect deaths.
- Economic damage data is available from the full EM-DAT database (requires registration at emdat.be) but is not included in this extract.
- To construct a disaster shock variable, consider: `disaster_shock = 1 if deaths > country_median * 5` or use the death_rate_per_100k with a threshold.

### Citation
EM-DAT, CRED / UCLouvain, Brussels, Belgium. www.emdat.be.
Ritchie, H., Rosado, P. and Roser, M. (2022). "Natural Disasters." *Our World in Data*.

---

## 4. COVID-19 Pandemic

**File:** `covid_by_country_year.csv`
**Source:** Our World in Data (OWID), compiled from Johns Hopkins University CSSE, national government reports, WHO
**Coverage:** 236 countries/territories, 2020-2023
**Unit of observation:** Country-year (annual aggregates)

### Variables

| Variable | Description |
|----------|-------------|
| `iso_code` | ISO 3166-1 alpha-3 country code |
| `year` | Year (2020-2023) |
| `total_cases_per_million` | Cumulative confirmed cases per million population (end of year) |
| `total_deaths_per_million` | Cumulative confirmed deaths per million population (end of year) |
| `stringency_index` | Oxford COVID-19 Government Response Stringency Index (0-100, annual average). Measures strictness of lockdown policies: school closures, workplace closures, travel bans, etc. |

### Notes
- The **stringency index** is particularly useful as an exogenous shock variable because it captures policy restrictions (school closures, travel bans) that directly affect students' ability to travel and enroll.
- Cases and deaths are cumulative within each year — to get annual incidence, compute `cases_year_t - cases_year_{t-1}`.
- Testing intensity varies across countries, making case counts less comparable than death counts.

### Suggested Shock Variables
- `covid_lockdown`: `1` if stringency_index > 50 (strict restrictions)
- `covid_severe`: `1` if total_deaths_per_million > 1000
- `covid_year`: Simple dummy for 2020-2021 (peak pandemic period)

### Citation
Mathieu, E., Ritchie, H., Rodés-Guirao, L. et al. (2021). "A global database of COVID-19 vaccinations." *Nature Human Behaviour*.
Hale, T. et al. (2021). "A global panel database of pandemic policies (Oxford COVID-19 Government Response Tracker)." *Nature Human Behaviour*.

---

## 5. Political Stability & Governance

**File:** `political_stability.csv`
**Source:** World Bank Worldwide Governance Indicators (WGI)
**Coverage:** ~200 countries, 2015-2023
**Unit of observation:** Country-year

### Variables

| Variable | Description | Range |
|----------|-------------|-------|
| `iso_code` | ISO 3166-1 alpha-3 country code | |
| `country_name` | Country name | |
| `year` | Year of observation | |
| `political_stability` | Political Stability and Absence of Violence/Terrorism | -2.5 (unstable) to +2.5 (stable) |
| `govt_effectiveness` | Government Effectiveness | -2.5 to +2.5 |
| `rule_of_law` | Rule of Law | -2.5 to +2.5 |
| `control_corruption` | Control of Corruption | -2.5 to +2.5 |

### Notes
- All indicators are standardized with mean 0 and standard deviation ~1 across countries.
- Negative values indicate below-average governance; positive values indicate above-average.
- **Political Stability** is the most relevant for this project — it captures perceptions of the likelihood of political instability and politically-motivated violence, including terrorism.
- To construct a shock variable: `political_shock = 1 if political_stability < -1.5` (bottom ~7% of countries).
- Year-over-year drops > 0.5 points indicate rapid destabilization.

### Suggested Shock Variables
- `political_instability`: `1` if political_stability < -1.0
- `political_destabilization`: `1` if YoY change in political_stability < -0.5
- `governance_shock`: Composite of drops in multiple WGI dimensions

### Citation
Kaufmann, Daniel, Aart Kraay and Massimo Mastruzzi (2011). "The Worldwide Governance Indicators: Methodology and Analytical Issues." *Hague Journal on the Rule of Law*, 3(2): 220-246.

---

## Summary of All Shock Data Files

| File | Source | Countries | Years | Key Variables |
|------|--------|-----------|-------|---------------|
| `macro_shocks.csv` | World Bank WDI | 214 | 2010-2023 | GDP shock, currency crisis, unemployment spike, PPP shock |
| `ucdp_conflict.csv` | UCDP/PRIO | 119 (with conflict) | 1946-2023 | Conflict active, intensity, onset, type |
| `natural_disasters.csv` | OWID/EM-DAT | ~200 | 2000-2023 | Deaths, death rate per 100k |
| `covid_by_country_year.csv` | OWID/Oxford | 236 | 2020-2023 | Cases, deaths per million, stringency index |
| `political_stability.csv` | World Bank WGI | ~200 | 2015-2023 | Political stability, governance effectiveness, rule of law |
| `macro_combined.csv` | World Bank WDI | 214 | 2010-2023 | GDP/cap PPP, exchange rate, unemployment, PPP, price level |

### Merge Key
All files use ISO 3166-1 alpha-3 country codes (`iso3`, `iso_code`, or `country_code`) and `year` as the primary merge keys. When merging:
1. Rename the code column to a common name (e.g., `country_code`)
2. Left-join on `(country_code, year)`
3. For UCDP: non-matched country-years = no conflict (fill `any_conflict` with 0)
4. For disasters: non-matched country-years = no disaster deaths (fill with 0)
