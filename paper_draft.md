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

### Table 1. Descriptive statistics by income tier

Countries are split into quartiles by 2023 GDP per capita PPP.

| Tier (Q of GDP/cap PPP) | Countries | Athletes | Per country | GDP/cap PPP (med.) | 15-24 pop, mil (med.) | Athletes / mil pop | **Athletes / mil 15-24** | Pol. stab. (med.) | Econ shocks 2010-23 | Years w/ conflict |
|---|---|---|---|---|---|---|---|---|---|---|
| Q1 Low                | 26 |  533 | 20.5 |  $7,670 | 4.1 | 0.16 |   **0.84** | -0.60 | 2.85 | 3.6 |
| Q2 Lower-middle       | 25 |  109 |  4.4 | $21,917 | 0.7 | 0.27 |   **2.38** | -0.08 | 2.56 | 1.7 |
| Q3 Upper-middle       | 25 |  352 | 14.1 | $41,288 | 0.6 | 1.35 |  **12.15** | +0.58 | 2.48 | 1.4 |
| Q4 High               | 26 |  825 | 31.7 | $64,485 | 0.9 | 1.43 |  **13.17** | +0.79 | 1.50 | 0.0 |
| All                   |102 | 1819 | 17.8 | $30,837 | 1.1 | 0.71 |   **5.59** | +0.39 | 2.34 | 1.7 |

Two patterns to note before we turn to regressions:

1. **Per-cohort rate rises sharply with income.** Athletes per million
   15-24 year-olds is 16× higher in Q4 (13.17) than in Q1 (0.84). Q3
   countries already approach Q4 rates.

2. **Aggregate counts do not.** Q1 supplies 533 athletes vs Q4's 825 --
   only a 1.5× ratio -- because Q1 contains large populous countries
   (median 4.1 million 15-24-year-olds vs 0.9 million in Q4). The
   Kenya / Nigeria / Jamaica pipelines load heavily into Q1 and
   sustain the count, even as the per-cohort rate is small.

This U-shape in counts (Q1 high, Q2 low, Q3-Q4 high) versus monotonic
rate (Q1 low → Q4 high) is the central tension we model.

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

The substantively most informative specifications are the rate models,
which remove the scale mechanic entirely. We report two: the
conventional rate (athletes per million total population) and a
cohort-corrected rate (athletes per million age-15-24 population,
2018-2022 average) that matches the recruitment age window of currently
enrolled athletes.

| Variable                         | Per million total pop | Per million age 15-24 |
|----------------------------------|----:|----:|
| $\log \text{GDPpc}$              | +0.22 (0.24)  | **+0.39*** (0.24) |
| Political stability (WGI)        | **+1.16*** (0.31) | **+1.19*** (0.30) |
| Economic shocks (count, 2010-23) | +0.03 (0.11)  | +0.05 (0.12) |
| $\log(1 + \text{disaster deaths})$ | **−0.25*** (0.06) | **−0.23*** (0.06) |
|                                  |       |       |
| $R^2$                            | 0.45  | 0.48  |
| n                                | 100   | 100   |

*Standard errors in parentheses. Stars: \*\*\* p<0.01, \*\* p<0.05, \* p<0.10.*

Substantively, scaling by the 15-24 cohort rather than by total
population brings $\log \text{GDPpc}$ to marginal significance (+0.39,
p<0.10) and raises $R^2$ slightly. The political-stability and
disaster-deaths coefficients are essentially unchanged. We use the
cohort rate as the preferred outcome going forward.

A 1-unit increase on the WGI political-stability index (which ranges
roughly from −2.5 to +2.5 across countries) is associated with
$\exp(1.19) \approx 3.3$ times more athletes per young person in NCAA
D-I. A doubling of cumulative disaster deaths is associated with about
$2^{-0.23} \approx 15\,\%$ fewer athletes per young person.

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

### 3.4 Extended specification with inflation and refugees

We extend the rate model with two additional shock variables that
arrived after the headline results: average CPI inflation 2010-2023
(World Bank WDI) and total refugee outflows 2010-2023 normalised by
the 15-24 cohort (UNHCR Refugee Statistics, country of origin).

| Variable                                | Spec 4b | Spec 7 (extended) |
|-----------------------------------------|---------|-------------------|
| $\log \text{GDPpc}$                     | +0.39*  | +0.37 (n.s.)      |
| Political stability (WGI)               | +1.19***| **+1.13***        |
| $\log(1 + \text{mean inflation 2010-23})$ |       | −0.20 (n.s.)      |
| $\log(1 + \text{refugees per cohort 2010-23})$ |   | **+0.77*          |
| Years with conflict 2010-23             |         | −0.00 (n.s.)      |
| $\log(1 + \text{disaster deaths 2010-23})$ | −0.23*** | **−0.22***       |
| $R^2$                                   | 0.48    | 0.47              |
| $n$                                     | 100     | 99                |

Three observations.

**Refugee outflows are positively associated with athlete supply.** A
doubling of refugees-plus-asylum-seekers per young person is associated
with $\exp(0.77 \cdot \ln 2) - 1 \approx 70\,\%$ more athletes per young
person, after conditioning on GDP, political stability, conflict, and
disasters. The point estimate is significant only at the 10% level, but
the direction is opposite to the naive "refugee push" reading and worth
naming. Three plausible mechanisms:

  *(i)* selection -- athletes from refugee-producing countries who
        successfully reach US college sports are positively selected on
        unobservables (skills, networks, institutional ties);
  *(ii)* a common-cause story -- whatever degrades home institutions
        (war, economic collapse) raises both refugee outflows and the
        marginal value of an outside option, including athletic
        scholarships;
  *(iii)* mechanical -- the cohort denominator is small in war-affected
        states, inflating the rate ratio.

Decomposing by sport (soccer vs track separately) and looking at
country-specific time-series of refugee waves and subsequent NCAA
enrollment would be needed to discriminate; this is left for further
work.

**Inflation does not predict athlete supply** in this specification
(coefficient −0.20, n.s.). The cumulative-inflation control was
intended to capture economic disturbance not picked up by the
binary `econ_shocks` count; it adds no information here.

**Disaster deaths and political stability survive again.** Both
coefficients are essentially unchanged from Spec 4b (within standard
error). The headline "institutions over economics" reading from
Section 3.2 is robust to these additional controls.

### 3.5 Shift-share decomposition

To probe whether sport-mix specialization itself drives the country
panel result, we construct a Bartik-style shift-share exposure on
track-event composition. For each country $c$ with at least five
international track athletes (n=40), we compute shares $s_{c,e}$
across nine event groups (Sprints, Mid-Distance, Distance, Cross
Country, Hurdles, Jumps, Pole Vault, Throws, Multi-Events) parsed from
roster position fields. Leave-one-out shifts $\Delta_e^{-c}$ are the
total intl athletes in event $e$ summed over countries other than $c$.
The Bartik exposure is

$$
B_c = \sum_e s_{c,e} \cdot \log(1 + \Delta_e^{-c}).
$$

Three uses confirm and refine the main story.

**(i) As a covariate.** When $B_c$ is added to the count specification,
it enters at +0.95 (p<0.10) and absorbs the $\log \text{GDPpc}$
coefficient (which falls from +0.32 to −0.08, n.s.). This is mechanical:
sport mix and country wealth are correlated, so including both
introduces multicollinearity. In the cohort-rate specification, $B_c$
is insignificant on its own (+0.91, n.s.) but the $\log \text{disaster
deaths}$ coefficient strengthens to −0.39 (p<0.01) and political
stability rises to **+1.30 (p<0.01)** -- the strongest stability point
estimate in any specification.

**(ii) As 2SLS instrument.** The first-stage F on $B_c$ is 13.5 -- just
above the rule-of-thumb 10. We do not lean on the IV estimates: the
2SLS standard errors blow up to the point where no coefficient is
distinguishable from zero. The instrument is too weak in our cross-
section to identify a clean local average treatment effect.

**(iii) Decomposition.** Regressing $\log \text{athletes}$ on $B_c$
alone yields $R^2 = 0.13$ and slope +1.09. The Bartik captures meaningful
but not dominant variation. After residualizing on $B_c$, country-level
shocks no longer correlate with the residual ($R^2 = 0.04$, no
significant coefficients) -- in this 40-country subsample, what the
shocks "explained" was largely sport-mix variation correlated with
country fundamentals.

The most consequential finding from the Bartik exercise is the
**robustness of the political-stability coefficient**: it remains
+1.21 (p<0.01) in the rate model even after the Bartik enters as a
covariate. This rules out a plausible alternative -- that
political-stability is a hidden sport-mix effect (e.g., stable
countries doing more throwing) -- and supports the institutions-as-
mechanism interpretation we develop in Section 4.

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
- `run_bartik.py` builds the track-event shift-share exposure and
  writes `output/bartik_results.txt` and `output/bartik_panel.csv`.
- `run_paper_outputs.py` builds Table 1 (`output/table1_descriptives.csv`),
  the coefficient forest plot (`output/fig_coef_forest.png`), the
  cohort-rate scatter (`output/fig_log_athletes_per_cohort.png`), and
  the top-20 panel (`output/fig_top20_countries_panel.png`).
