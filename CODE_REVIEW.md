# Code Review — Student Athletes Notebook

**Reviewer:** Prof. Malkova
**Date:** April 7, 2026
**File reviewed:** `Untitled Notebook.ipynb` (91 cells)

---

## Summary

The overall approach and logic are solid. The data pipeline — IPEDS for institutions, WDI for macro indicators, NCAA directory for the crosswalk, EADA for athletics financials, and scraped rosters for athlete-level data — is well-designed and appropriate for the research question.

The code works and the results are being saved correctly. The main issues are (1) the notebook has accumulated a lot of test/draft code that makes it hard to follow, and (2) there are some bugs and data quality risks that should be fixed before we move to analysis.

Below is detailed feedback organized by section.

---

## 1. Notebook Organization

**Problem:** The notebook has 91 cells, but many are duplicates, abandoned experiments, or diagnostic checks. This makes it difficult to understand which code is the "real" version and which is a test.

### What should be removed

| Cells | Description | Why remove |
|-------|-------------|------------|
| 0 | `print("Hello! My Colab environment is working.")` | Test cell, not needed |
| 9-12 | `os.listdir()` and `os.path.exists()` checks | Diagnostic cells used during setup — no longer needed |
| 14 | Verifying `ef` and `sfa` DataFrames exist | Diagnostic |
| 24 | Duplicate of cell 22 (IPEDS analysis: total institutions, NRAL enrollment, top 10) | Exact copy — keep only one |
| 28-31 | Three separate attempts at NCAA directory scraping (`try_requests_approach`, `try_selenium_approach`) before the final working version in cell 37 | Superseded by cell 37 |
| 35-36 | Random college sampling (10 random colleges, 10 from different conferences) | Exploratory tests, not part of pipeline |
| 38 | Random colleges from different division/conference/state combos | Exploratory test |
| 51-54 | Four iterations of platform detection (`detect_platform`) with increasing sophistication | Only the final version (cell 54) is needed |
| 58-62 | Five versions of Sidearm scraper (cells 59-62 plus cell 61 with VERSÃO ALTERNATIVA SEM SELENIUM) | Only the final "SIDEARM 1" version (cell 76) or the "SCRAPER FINAL" (cell 71) is needed |
| 63 | Standalone "SCRAPER WMT DIGITAL" | Superseded by "WMT DIGITAL OFICIAL" (cell 88) |
| 64 | WMT scraper version 1 | Superseded |
| 78-80 | Three intermediate WMT scraper versions | Superseded by "WMT DIGITAL OFICIAL" (cell 88) |
| 81-82 | `apt-get install chromium-chromedriver` (appears 3 times in notebook) | Keep only once, in setup |

### What should be kept (proposed structure)

```
Section 1: SETUP (1 cell)
  - All pip installs, apt-get, imports, Drive mount, constants

Section 2: LOAD IPEDS DATA (2-3 cells)
  - Load HD2023, EF2023a, SFA2223
  - Filter to 4-year institutions (SECTOR 1 or 2)
  - Basic summary stats + missing values

Section 3: LOAD EADA (1 cell)
  - Load EADA xlsx, merge with IPEDS on UNITID

Section 4: WDI MACRO DATA (3 cells)
  - Download 6 indicators from World Bank API
  - Reshape from wide to long
  - Merge into single `macro` DataFrame

Section 5: NCAA DIRECTORY (1 cell)
  - The working CSRF+API scraper (cell 37)

Section 6: NCAA-IPEDS CROSSWALK (3 cells)
  - Fuzzy matching with thefuzz
  - Score analysis
  - Manual corrections + verified crosswalk save

Section 7: PLATFORM DETECTION (1 cell)
  - Final version with fingerprints + confidence scores (cell 54)

Section 8: ROSTER SCRAPING — shared utilities (1 cell)
  - append_to_roster_csv()
  - setup_driver()

Section 9: TFRRS SCRAPER (2 cells)
  - Main scraper (cell 48)
  - Retry with Playwright fallback (cell 49)

Section 10: SOCCER SCRAPERS (3 cells)
  - Sidearm scraper — final version only
  - WMT Digital scraper — "WMT DIGITAL OFICIAL" only
  - PrestoSports scraper — "PRESTOSPORTS FINAL" only

Section 11: ORCHESTRATOR (1 cell)
  - The main loop that dispatches by platform (cell 73)

Section 12: ANALYSIS (2-3 cells)
  - NRAL enrollment analysis
  - Missing values in macro
  - Any other stats
```

This brings the notebook from ~91 cells down to ~20-22 cells.

---

## 2. Code Bugs

### Bug 1: Deduplication key is too narrow

**Where:** `append_to_roster_csv()` (defined 8 times, same bug each time)

```python
def append_to_roster_csv(new_df, filepath, key_columns=['school_name', 'name']):
    ...
    combined = combined.drop_duplicates(subset=key_columns, keep='last')
```

**Problem:** If a men's team and a women's team at the same school both have a player named "Alex Smith", this will drop one of them. Two different people get merged into one row.

**Fix:** Change the default key to include gender:
```python
key_columns=['school_name', 'name', 'gender']
```

### Bug 2: Dead code in `scrape_wmt_roster_parsed()`

**Where:** Cell ~54 (one of the WMT scraper versions)

```python
def scrape_wmt_roster_parsed(soup, school_name):
    roster_data = []
    player_cards = soup.find_all(...)
    for card in player_cards:
        # (código de extração igual ao anterior)
        pass    # <-- THIS DOES NOTHING
    ...
    return pd.DataFrame(roster_data)  # always returns empty
```

**Problem:** The function has a `pass` placeholder inside the extraction loop, so it never actually extracts any data. Any school that hits this fallback path silently returns zero athletes.

**Fix:** Either remove this function or copy the actual extraction logic into it.

### Bug 3: Bare `except: pass` swallows errors silently

**Where:** JSON-LD parsing in the Sidearm scraper (cell 61)

```python
for script in scripts:
    try:
        data = json.loads(script.string)
        ...
    except:
        pass  # <-- swallows ALL exceptions silently
```

**Problem:** If there's a bug in the JSON parsing (e.g., a `KeyError`, `TypeError`), you'll never know. The code silently moves on and you might miss valid data.

**Fix:**
```python
except (json.JSONDecodeError, TypeError):
    continue  # only skip JSON parse errors, not logic bugs
```

### Bug 4: Selenium driver not closed on exception

**Where:** Multiple WMT and Sidearm scraper versions

```python
def scrape_wmt_selenium(url, school_name):
    try:
        driver = setup_driver()
        driver.get(url)
        ...
        driver.quit()         # only reached on success
        return pd.DataFrame(roster_data)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame() # driver.quit() never called!
```

**Problem:** If the scraper crashes mid-execution, the Chrome process stays alive. After scraping many schools, this can exhaust Colab's memory (each Chrome instance uses ~100-200MB).

**Fix:** Use `try/finally`:
```python
driver = setup_driver()
try:
    driver.get(url)
    ...
    return pd.DataFrame(roster_data)
except Exception as e:
    print(f"Error: {e}")
    return pd.DataFrame()
finally:
    driver.quit()  # always runs
```

---

## 3. Data Quality Concerns

### 3a. Hometown parsing misses international locations

**Where:** All roster scrapers use this regex for hometown detection:

```python
hometown_pattern = r'[A-Z][a-zA-Z\s\.\-]+\s*,\s*([A-Z]{2}|[A-Z]\.[A-Z]\.|[A-Z][a-z]{1,2}\.)'
```

**Problem:** This pattern is designed for U.S. locations like "Austin, TX" or "Portland, Ore." but will miss:
- International locations without commas: "London England", "São Paulo Brazil"
- Locations with non-ASCII characters: "München, Germany", "Malmö, Sweden"
- Locations formatted as "City / Country" (common on some rosters)

**Why this matters:** This project specifically studies international student-athletes. If the hometown field is empty for many international players, we lose the ability to link them to their home country's macro indicators.

**Recommendation:** After scraping is complete, run a check:
```python
# How many athletes have no hometown?
rosters = pd.read_csv('soccer_rosters.csv')
print(rosters['hometown'].isna().sum(), "missing out of", len(rosters))
# Check a sample of the missing ones manually
```
If the miss rate is high for international athletes, we may need to parse the hometown differently per platform or scrape individual athlete profile pages.

### 3b. Club keywords list is too specific

**Where:** WMT and Sidearm scrapers

```python
club_keywords = ['academy', 'school', 'club', 'fc', 'sc', 'united',
    'breakers', 'solar', 'phoenix', 'atlanta', 'austin',
    'cedar stars', 'morris united', 'fremad amager', ...]
```

**Problem:** This mixes generic keywords (`academy`, `club`, `fc`) with very specific club names (`cedar stars`, `fremad amager`, `grorud`). The specific names only help for a handful of known athletes and will miss all other clubs. Also, city names like `atlanta` and `austin` will false-match on hometown lines that contain those cities.

**Recommendation:** Keep only the generic keywords:
```python
club_keywords = ['academy', 'school', 'club', 'fc', 'sc', 'united',
                 'high school', 'prep', 'college', 'institute']
```

### 3c. TFRRS rosters are missing hometown and event data

**Where:** TFRRS scraper (cells 47-49)

```python
athletes.append({
    'athlete_name': name,
    'class_year':   class_year,
    'hometown':     None,   # only on individual athlete pages
    'event':        None,   # only on individual athlete pages
    ...
})
```

The code correctly notes that hometown is only available on individual athlete profile pages. The `profile_url` is saved, so a follow-up scraper could visit each athlete's page to get this data. This is a known limitation, not a bug — but it means **the TFRRS data cannot yet be linked to macro indicators** without the follow-up scrape.

**Recommendation:** Add a cell (or a separate notebook) that:
1. Loads `tfrrs_all_rosters.csv`
2. Iterates through unique `profile_url` values
3. Scrapes each athlete page for hometown
4. Updates the roster CSV

---

## 4. Performance / Efficiency

### 4a. Fuzzy matching crosswalk is O(n * m) — very slow

**Where:** Cell 41

```python
for index_ncaa, row_ncaa in ncaa.iterrows():       # ~1,100 NCAA schools
    for index_hd, row_hd in hd.iterrows():          # ~6,500 IPEDS schools
        score = fuzz.token_set_ratio(ncaa_name, hd_name)
```

This runs ~7 million string comparisons. On Colab this probably takes 30+ minutes.

**Optimization (optional, for future runs):**
```python
from thefuzz import process

# This is ~10x faster because process.extractOne uses internal optimizations
results = []
for _, row in ncaa.iterrows():
    match, score, idx = process.extractOne(
        row['cleaned_name'],
        hd['INSTNM_cleaned'].tolist(),
        scorer=fuzz.token_set_ratio
    )
    results.append({...})
```

Since the crosswalk is already saved to CSV and manually verified, this is not urgent — but worth knowing for future runs.

### 4b. `append_to_roster_csv()` re-reads the entire CSV on every school

**Where:** Every scraper calls this after each school:

```python
def append_to_roster_csv(new_df, filepath, ...):
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)           # reads entire file
        combined = pd.concat([existing_df, new_df])   # copies everything
        combined.to_csv(filepath, ...)                 # rewrites entire file
```

For 100+ schools, the file is read and rewritten 100+ times, growing each time. With ~5,000 athletes this works fine but will slow down significantly at scale.

**Better approach (for future):** Accumulate all results in memory, then write once at the end:
```python
all_results = []
for school in schools:
    df = scrape(...)
    all_results.append(df)

final = pd.concat(all_results)
final.to_csv(SUCCESS_FILE, index=False)
```

---

## 5. Things Done Well

These are worth highlighting — good practices to keep:

1. **Crosswalk manual corrections** (cell 44): The verified crosswalk with manual corrections for known mismatches (Manhattan vs. Manhattanville, Pomona-Pitzer expansion, Canadian school removal) is careful and thorough work.

2. **Platform detection with confidence scores** (cell 54): The fingerprint-based detection with HIGH/MEDIUM/LOW confidence and a separate review file for uncertain matches is a smart approach.

3. **Failure logging**: Every scraper saves failures to a separate CSV with the reason. This makes debugging and retrying easy.

4. **TFRRS retry with Playwright fallback** (cell 49): Using requests first (fast) then Playwright (slow but handles JS) is the right strategy.

5. **Progress reporting**: The TFRRS scraper prints progress every 10 teams with success/failure counts. This is helpful for long-running scrapes.

6. **Saving `profile_url`** in TFRRS data: Even though hometown isn't scraped yet, saving the URL means we can come back for it later without re-scraping team pages.

---

## 6. Recommended Next Steps

1. **Clean up the notebook** — Remove duplicate/test cells, keep only final versions. Rename from "Untitled Notebook" to something descriptive (e.g., `01_data_pipeline.ipynb`).

2. **Fix the deduplication key** — Add `'gender'` to avoid dropping same-name athletes across men's/women's teams.

3. **Fix the Selenium resource leak** — Add `finally: driver.quit()` to all Selenium fallback functions.

4. **Check hometown coverage** — Run the missing-data check suggested in section 3a. If coverage is low for international athletes, we need a different parsing strategy.

5. **Scrape TFRRS athlete pages for hometown** — This is needed before TFRRS data can be linked to macro indicators.

6. **Push cleaned code to Git** — The repo `git@github.com:alina-malkova/students_athletes.git` is set up. Once the notebook is reorganized, push it there so we have version history.
