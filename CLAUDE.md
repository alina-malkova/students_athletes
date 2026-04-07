# Student Athletes Research Project

## Overview

Economics research project studying international student-athletes at NCAA institutions. The project examines the "economic distance" between athletes' home countries and U.S. colleges — linking macroeconomic indicators (GDP, PPP, exchange rates, unemployment) to enrollment patterns and athletic participation.

**Internal project name**: `RA_Project_EconDistance`
**Primary environment**: Google Colab notebook with Google Drive storage
**Language**: Python (pandas, requests, BeautifulSoup, Selenium)

## Research Question

How do macroeconomic conditions in athletes' home countries relate to their enrollment at U.S. NCAA institutions? The project focuses on nonresident alien (international) student-athletes, particularly in soccer.

## Data Sources

### IPEDS (Integrated Postsecondary Education Data System)
- `HD2023.csv` — Institution directory (names, sectors, locations); key column: `UNITID`; has BOM in header (`ï»¿UNITID`)
- `EF2023a.csv` — Enrollment data; contains `NRAL` (nonresident alien) columns; columns starting with `X` are imputation flags
- `SFA2223.csv` — Student financial aid
- Filter to 4-year institutions: `SECTOR` in [1, 2]
- Encoding: `latin-1`

### EADA (Equity in Athletics Disclosure Act)
- `EADA_All_Data_Combined_2022-2023_SAS_SPSS_EXCEL/` — Athletic program data
- Available as `.xlsx`, `.sas7bdat`, `.sav`, `.doc`
- Merged with IPEDS via `UNITID` (inner join)

### World Bank WDI (World Development Indicators)
- Downloaded via `wbgapi` Python package for years 2010-2023
- Six indicators stored as `WDI_*.csv`:
  - `PPP_conversion_factor` (PA.NUS.PPP) -> `ppp`
  - `Official_exchange_rate` (PA.NUS.FCRF) -> `exchange_rate`
  - `Unemployment_rate` (SL.UEM.TOTL.ZS) -> `unemployment`
  - `GDP_per_capita_growth` (NY.GDP.PCAP.KD.ZG) -> `gdp_growth`
  - `GDP_per_capita_PPP` (NY.GDP.PCAP.PP.KD) -> `gdp_per_capita_ppp`
  - `Price_level_ratio` (PA.NUS.PPPC.RF) -> `price_level_ratio`
- Reshaped from wide (YR columns) to long format, merged into a single `macro` DataFrame on `['country_code', 'country_name', 'year']`

### NCAA Directory
- Scraped from `web3.ncaa.org/directory/api/directory/memberList` (JSON API with CSRF token)
- Contains division, conference, state for member institutions
- Crosswalk file: `ncaa_ipeds_crosswalk.csv` / `ncaa_ipeds_crosswalk_verified.csv` — links NCAA schools to IPEDS UNITIDs using fuzzy matching (`thefuzz`)

### Athletic Roster Data (Soccer)
Web-scraped from college athletics websites. Three platform-specific scrapers:

1. **Sidearm Sports** — HTML table-based; uses `requests` + `BeautifulSoup`
2. **WMT Digital** — Card-based layout; uses `requests` with Selenium fallback
3. **PrestoSports** — Similar card/table approach

**Output files:**
- `soccer_rosters.csv` — Successfully scraped rosters (columns: `school_name`, `jersey_number`, `name`, `position`, `class_year`, `height`, `hometown`, `high_school_club`, `gender`)
- `soccer_roster_failures.csv` — Schools where scraping failed
- `school_website_platforms.csv` — Platform metadata and roster URLs per school/gender

### Track & Field Roster Data (TFRRS)
- `tfrrs_team_urls.csv` — Team page URLs
- `tfrrs_all_rosters.csv` — Scraped roster data
- `tfrrs_failures.csv` / `tfrrs_still_failed.csv` — Failed scrapes

## Project Structure (Google Drive)

```
RA_Project_EconDistance/
├── raw_data/           # All source and scraped data
│   ├── HD2023.csv, EF2023a.csv, SFA2223.csv  (IPEDS)
│   ├── WDI_*.csv                               (World Bank)
│   ├── ncaa_ipeds_crosswalk*.csv                (Crosswalk)
│   ├── soccer_rosters.csv                       (Scraped rosters)
│   ├── school_website_platforms.csv             (Platform metadata)
│   ├── tfrrs_all_rosters.csv                    (Track & field)
│   ├── EADA_All_Data_Combined_2022-2023_SAS_SPSS_EXCEL/
│   └── debug_html/                              (Saved HTML for debugging scrapers)
├── cleaned_data/
├── output/
└── code/
```

## Local Directory (this folder)

```
Students Athletes/
├── CLAUDE.md                    # This file
├── Untitled Notebook.ipynb      # Main Colab notebook (91 cells)
└── Raw Data Apr 07 2026.zip     # Snapshot of raw_data/ folder
```

## Key Technical Notes

- **BOM in IPEDS headers**: `HD2023.csv` has `ï»¿UNITID` — always rename to `UNITID` before merging
- **Encoding**: IPEDS CSVs use `latin-1` encoding
- **Fuzzy matching**: NCAA-to-IPEDS crosswalk uses `thefuzz` for institution name matching
- **Scraper delays**: All scrapers use 2-3 second delays between requests
- **Selenium fallback**: WMT Digital scraper tries `requests` first, falls back to headless Chrome
- **Deduplication**: Roster CSVs use `append_to_roster_csv()` which deduplicates on `['school_name', 'name']`
- **Notebook language**: Code comments and print statements are in Portuguese (Brazilian), reflecting RA collaboration

## Workflow

1. Load & filter IPEDS data (4-year institutions)
2. Download WDI macroeconomic indicators, reshape to long format, merge into `macro`
3. Scrape NCAA directory, build NCAA-IPEDS crosswalk via fuzzy matching
4. Load EADA athletics data, merge with IPEDS
5. Scrape soccer/track rosters from school websites (by platform type)
6. Link athlete hometown/nationality data to macroeconomic indicators
7. Analyze relationship between economic distance and enrollment patterns
