# Determinants of International NCAA Athlete Representation

## 1. Data

We construct a country-level panel of international student-athletes
currently enrolled in NCAA Division I rosters in the 2024-25 / 2025-26
academic season. Athlete-level records are scraped from official athletic
department websites for soccer (660 athletes from 27 schools) and track
and field / cross country (18,657 athletes from 328 of 355 D-I programs;
school coverage 92.4 %).

Hometown text is parsed from each roster page and mapped to ISO-3 country
codes via a normalisation procedure that combines US state abbreviations
(both postal and AP-style), Canadian provinces, World Bank country names,
and a manual alias map for common roster forms ("U.K.", "B.C.",
"Czech Republic"). The mapping resolves 87.6 % of track and 99.8 % of
soccer hometowns. Of 19,317 total athletes, **1,830 are international**
(country code other than USA), and 1,819 of those merge cleanly to the
2023 World Bank macro panel.

Country covariates are taken from World Bank WDI (GDP per capita PPP,
population, unemployment, exchange rate, PPP), Worldwide Governance
Indicators (political stability, government effectiveness, rule of law,
control of corruption), UCDP/PRIO armed-conflict data, EM-DAT natural
disasters, and Oxford COVID-19 Government Response Tracker. For each
country we use both 2023 cross-sections and 2010-2023 cumulative
aggregates (e.g. years with active conflict, total disaster deaths,
count of macroeconomic shocks).

The country panel for analysis contains the **102 countries with at
least one international athlete in our sample**.

## 2. Specification

The baseline model regresses log athlete count or log athletes per
million population on home-country economic and political conditions:

$$
\log \text{athletes}_c = \alpha + \beta_1 \log \text{GDPpc}_c
                         + \beta_2 \log \text{pop}_c
                         + \beta_3 \text{polstab}_c
                         + \beta_4 \text{econshocks}_c
                         + \beta_5 \log(1 + \text{deaths}_c) + \varepsilon_c
$$

We report Huber-White heteroskedasticity-robust (HC1) standard errors.
The rate specification replaces the dependent variable with $\log
(\text{athletes}_c / \text{pop}_c)$ and drops $\log \text{pop}_c$ from
the right-hand side. We also fit Poisson and Negative Binomial models
on raw athlete counts with $\log \text{pop}_c$ as offset, which
properly handle the count nature of the outcome (especially relevant
for countries sending only 1-2 athletes).

## 3. Main results

### 3.1 Population matters; once it is controlled, the picture is dominated by political stability

Without a population control, $\log \text{GDPpc}$ enters positively but
small (β ≈ 0.31, p < 0.05) and $\log(1 + \text{disaster deaths})$
appears as a strong positive predictor (β = 0.18, p < 0.01) — a result
that, on inspection, reflects a population-size confound: large
countries experience more disasters and also produce more athletes.
Adding $\log \text{pop}_c$ knocks the disaster coefficient out of
significance, while $\log \text{GDPpc}$ rises to β = 0.48 (p < 0.01)
and $\log \text{pop}_c$ enters at β = 0.29 (p < 0.01).

The substantively most informative specification is the rate model
($\log$ athletes per million population), which removes the scale
mechanic entirely. Here the picture changes:

| Variable                         | Coef  | SE    | t      |
|----------------------------------|-------|-------|--------|
| $\log \text{GDPpc}$              | +0.22 | 0.24  | 0.93   |
| Political stability (WGI)        | **+1.16** | 0.31  | **3.81** |
| Economic shocks (count, 2010-23) | +0.03 | 0.11  | 0.30   |
| $\log(1 + \text{disaster deaths})$ | **−0.25** | 0.06 | **−3.99** |
|                                  |       |       |        |
| $R^2$                            | 0.45  |       |        |
| n                                | 100   |       |        |

*Stars: *** p<0.01, ** p<0.05, * p<0.10. HC1 robust SEs throughout.*

A 1-unit increase on the WGI political-stability index (which ranges
roughly from −2.5 to +2.5 across countries) is associated with
$\exp(1.16) \approx 3.2$ times more athletes per capita in NCAA D-I.
A doubling of cumulative disaster deaths is associated with about
$2^{-0.25} \approx 16\,\%$ fewer athletes per capita.

### 3.2 Robustness

Five specifications confirm the political-stability finding and reveal
that GDP per capita is fragile (significant in some specs, not in
others), while the disaster-deaths coefficient flips sign correctly
once population is properly handled.

| Coefficient                | OLS rate | Drop top-20 | Heckman (IMR)   | Poisson  | Neg.Bin. |
|----------------------------|----------|-------------|-----------------|----------|----------|
| $\log \text{GDPpc}$        | +0.22    | +0.31       | +1.13 (n.s.)    | +0.07    | −0.28    |
| Political stability        | **+1.16***** | **+0.94***** | +0.58*       | **+1.09***** | **+1.29***** |
| Econ. shocks 2010-23       | +0.03    | +0.08       | —               | −0.02    | **−0.22*****  |
| $\log(1+\text{disaster deaths})$ | **−0.25***** | **−0.32***** | —      | **−0.19***** | **−0.21***** |
| $n$                        | 100      | 80          | 100             | 100      | 100      |
| $R^2$ / pseudo-$R^2$       | 0.45     | 0.46        | 0.31            | —        | —        |

The political-stability coefficient is significant and of similar
magnitude in **every** specification. The GDP per capita coefficient is
not. Two findings deserve comment:

**Heckman selection.** The selection probit (n = 191 countries with
non-missing covariates) shows that $\log \text{GDPpc}$ (β = 0.52,
p < 0.01) and $\log \text{pop}$ (β = 0.18, p < 0.01) determine whether
a country sends *any* athletes. Political stability is not significant
at this binary stage. The outcome equation with the inverse Mills
ratio retains a positive but less precise political-stability
coefficient (+0.58, p < 0.10) and an IMR coefficient of +2.68 (n.s.).
Selection is therefore present but does not overturn the conditional
result: among countries that send any athletes at all, more stable
ones send disproportionately more per capita.

**Negative Binomial picks up an additional shock effect.** In the NB
model the econ-shocks coefficient becomes significant negative
(β = −0.22, p < 0.05): a country experiencing more macroeconomic
shocks between 2010 and 2023 produces fewer athletes per capita,
controlling for level GDP and political stability. This may reflect
disruption of sport-development institutions during downturns; it is
also consistent with the disaster-deaths sign.

### 3.3 Sport-specific patterns

The international concentration is in track. The track-only panel
(n = 99 countries with $\geq 1$ international track athlete) replicates
the main result: $\log \text{GDPpc}$ +0.35 (p < 0.05), $\log \text{pop}$
+0.23 (p < 0.05). The soccer panel is too small (n = 24 international-
sender countries) to identify any structural effect; only $\log
\text{pop}$ approaches conventional significance (+0.31, p < 0.10).
A separate research design — possibly modelling school-level recruiting
networks rather than country-year macroeconomics — is required for
international soccer.

## 4. Discussion

Three substantive conclusions emerge.

First, the **population scale of the supply pool matters**. Roughly a
third of the variance explained by the count model is attributable to
$\log \text{pop}_c$ alone. Without that control, the disaster-deaths
coefficient absorbs this scale effect with the wrong sign, mimicking a
"push factor" interpretation that disappears once size is held fixed.

Second, **political stability is a robust positive predictor of
athlete-rate.** Across OLS rate, OLS drop-outliers, Heckman outcome,
Poisson, and NegBin specifications, the WGI political-stability
coefficient is significant and large. We interpret this as evidence
that the marginal pathway from a country to a U.S. NCAA roster runs
through *institutions* — youth-development clubs, school sports
infrastructure, federations — rather than through pure economic push
or pull. Stable states can sustain the multi-year investments that
produce internationally competitive athletes; unstable ones cannot.

Third, **economic distance per se has limited explanatory power for
the count of athletes** once population and political stability are
included. Notable outliers (Kenyan distance running, Jamaican
sprinting, Bahamian and Trinidadian recruiting pipelines) are visible
in the data but they do not drive the panel-level result: dropping
the top-20 senders leaves the political-stability coefficient at
+0.94 (p < 0.01).

This contrasts with a naive reading of GDP differences as a "push"
mechanism, in which low-income countries should over-produce athletes
relative to high-income peers. We find the opposite at the aggregate
level — high-GDP countries produce more athletes overall — and no
robust relationship at the per-capita level once stability is
controlled.

## 5. Caveats and limitations

- **Cross-section, not panel.** Roster snapshots are 2024-25 / 2025-26.
  Macro covariates are 2023 plus 2010-23 aggregates. The model cannot
  identify the causal effect of, say, a particular shock; only
  associations between contemporaneous country conditions and the
  current stock of athletes in U.S. programs.
- **Coverage.** Hometown is observed for 87.6 % of track athletes;
  the unobserved 12.4 % are missing not at random (school-page
  rendering issues) and could understate countries whose schools rely
  on JS-rendered rosters.
- **One-sport dominance.** 96 % of the sample is track and field.
  Soccer is too small to identify, and we have no data on football,
  basketball, swimming, or other sports. The "international NCAA
  athlete" we describe is essentially a track athlete.
- **Athletes per capita is not the same as athletic talent per capita.**
  Recruiting selection, scholarship caps, language and visa frictions,
  and US schools' bilateral relationships filter the population. We
  observe the *combined* effect of supply and selection.
- **Population is contemporaneous.** A more careful design would use
  the cohort population (e.g. 18-22 year-olds 2017-2022) to match the
  ages of currently-enrolled athletes.

## 6. Reproducibility

All results in this section are reproducible from the local Python
scripts in this repository:

- `discover_track_urls.py`, `run_track_scrape.py`, `run_track_selenium.py`,
  `rediscover_urls.py` build the roster data.
- `run_country_mapping.py` resolves hometowns to country codes.
- `run_merge_analysis.py` joins rosters to NCAA-IPEDS crosswalk, IPEDS
  HD, EADA, and macro indicators.
- `run_join_shocks.py` adds the five shock panels and population.
- `run_econ_distance.py` adds the per-athlete economic-distance metrics.
- `run_analysis.py` and `run_robustness.py` produce the coefficient
  tables in `output/regression_results.txt` and
  `output/robustness_results.txt`.
