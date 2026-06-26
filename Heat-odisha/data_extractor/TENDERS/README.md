# Tenders

Public procurement data is scraped from the [Odisha Tenders](https://tendersodisha.gov.in/nicgep/app) website and used to identify, classify and geotag **heat-response tenders** -- works procured to prepare for or respond to heatwaves (cool roofs, water kiosks, heat-stroke wards, shade shelters, awareness drives, etc.).

## Methodology

### 1. Aggregate raw tenders
`count_tenders.py` concatenates the monthly AOC (Acceptance of Contract) tender CSVs scraped from the source website into a single combined file and reports basic totals (e.g. tenders with `Status == "Accepted-AOC"`).

### 2. Classify heat-response tenders (`heat_tenders.py` + `heat_tenders_config.py`)
All keyword lists used for classification live in `heat_tenders_config.py`; `heat_tenders.py` contains only the matching/classification logic.

For every tender, the `tender_externalreference`, `tender_title` and `Work Description` fields are concatenated into a single slug, lower-cased and stripped of punctuation (`clean_text`). Keywords are normalised through the same cleaner before matching, so hyphenated and spaced forms of a term (e.g. "heat-wave" / "heat wave") both match. Multi-word keywords are matched as whole tokens via regex (`\b...\b`), with internal whitespace collapsed to `\s+`.

- **`POSITIVE_KEYWORDS`** (derived from `THEMATIC_KEYWORD_GROUPS`) -- broad heat/disaster-adjacent vocabulary (drinking water, shelters, cooling infrastructure, awareness campaigns, scheme acronyms, etc.).
- **`NEGATIVE_KEYWORDS`** -- terms that should suppress a match even if a positive keyword fires (e.g. "stadium", "football", "crematorium").
- **`HEAT_SPECIFIC_KEYWORDS`** -- a curated whitelist of terms that are heat-specific on their own (e.g. "heatwave", "heat stroke", "cool roof", "water kiosk"). This list deliberately excludes generic/dual-use terms (saline, ambulance, first aid, climate resilient, etc.) that produced false positives in testing.

Two gates are computed per tender and both are written to the output:
- **`is_heat_tender`** (loose gate): any positive keyword fired and no negative keyword fired.
- **`is_strong_heat_tender`** (strong gate): a `HEAT_SPECIFIC_KEYWORDS` term fired; a negative keyword can only suppress a *weak* match, not a strong one.
- **`heat_signal`**: convenience column, `"strong"` or `"weak"`, derived from the strong gate.

`KEEP_ONLY_STRONG` (in `heat_tenders.py`) controls which rows are written to the output file -- by default the full loose set is kept so weak matches can still be inspected/filtered later via `heat_signal`. Tenders from departments listed in `EXCLUDED_DEPARTMENTS` are dropped regardless of keyword match.

Each surviving tender is further enriched with:
- **`Season`**: published month mapped to a season bucket via `SEASON_MONTHS` (Odisha's heatwave season runs roughly March-June).
- **`Scheme`**: funding scheme acronyms (`SCHEME_KEYWORDS`, e.g. RIDF, SDRF, CMRF, MGNREGA) found as whole tokens in the tender text, matched deterministically (sorted, not arbitrary).
- **`Response Type`** / **`Heat Response - Themes`** / **`Heat Response - Subhead`**: the tender is checked against each group in `THEMATIC_KEYWORD_GROUPS` (e.g. "Heatwave & Emergency Response", "Drinking Water & Hydration", "Cooling & Reflective Infrastructure", "Greening & Urban Forestry", "Shelter & Rest Infrastructure", "Health System Strengthening", "Awareness & Capacity Building"). The first matching group (in declared priority order) becomes `Response Type`; all matching groups and their matched keywords are recorded in `Heat Response - Themes` / `Heat Response - Subhead`. Tenders matching no theme are labelled `"Others"`.

`heat_tenders.py` can run either against a single concatenated CSV (`--input`/`--output`, used for testing) or against the original monthly-folder layout (`data/monthly_tenders/*.csv` -> `data/heat_tenders/*.csv` -> concatenated into `data/heat_tenders_all.csv`).

### 3. Geotag tenders (`geocode_district.py`, `geocode_blocks.py`)
Tender `location` text is fuzzy-matched (via `difflib.SequenceMatcher`) against the Odisha village/sub-district master list to resolve a district (`geocode_district.py`), then narrowed to a block within that district using village/GP/block name dictionaries built from the same master list (`geocode_blocks.py`), producing `DISTRICT_FINALISED` / `BLOCK_FINALISED` columns.

### 4. Build indicators (`transformer.py`)
The block-geotagged heat tenders are merged with the Odisha block shapefile (matching on upper-cased, stripped district/block names) to attach a block `object_id`. Indicators are then aggregated by `month` x `object_id` and written as one CSV per month per indicator:
1. `total_tender_awarded_value`: total awarded value of all heat tenders.
2. One `<scheme>_tenders_awarded_value` indicator per funding `Scheme` (e.g. `sdrf_tenders_awarded_value`, `ridf_tenders_awarded_value`).
3. One `<response-type>_tenders_awarded_value` indicator per heat-response `Response Type` (excluding `"Others"`).

Set `ONLY_STRONG_SIGNAL = True` in `transformer.py` to restrict indicator-building to `heat_signal == "strong"` rows only; the default builds indicators from the full (loose + strong) set.

## Project Structure
- `scripts`: scripts used to scrape, classify, geotag and transform the data
    - `count_tenders.py`: concatenates monthly scraped tenders into a single file and reports totals
    - `heat_tenders.py`: heat-tender classification logic (loose/strong gate, scheme/season/theme tagging)
    - `heat_tenders_config.py`: all keyword lists and lookup tables used by `heat_tenders.py` (positive/negative/heat-specific keywords, thematic groups, scheme acronyms, excluded departments, season-month map)
    - `geocode_district.py`: geocodes tenders to a district using fuzzy keyword matching against the village master list
    - `geocode_blocks.py`: geocodes district-tagged tenders to a block using the village/block shapefile
    - `transformer.py`: builds the final block x month indicator CSVs from the geotagged heat tenders
    - `scraper`: scraping logic for the source tenders portal
- `data`: datasets generated by the scripts above
    - `monthly_tenders`: all AOC tenders scraped from the tender website, organized by month
    - `heat_tenders`: subset of `monthly_tenders` identified as heat-response tenders
    - `heat_tenders_all.csv`: concatenation of all monthly `heat_tenders` files
    - `variables`: one folder per indicator listed above, each containing month-wise CSVs keyed by block `object_id`
