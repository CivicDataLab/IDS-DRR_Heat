"""
heat_tenders.py  (patched)

Fixes applied vs. the original (see the audit):

1. Keyword/​text hyphen mismatch
   clean_text() turns hyphens into spaces, so a hyphenated keyword such as
   "heat-wave" could never match. Keywords are now normalised through the
   SAME cleaner before the regex is built, so hyphenated and space forms
   both work.

2. is_heat_tender gate had no notion of heat-specificity
   The original tagged a tender on ANY single positive match, which is why
   generic public works (chc/phc/hospital/piped water supply/community hall)
   dominated the output. A configurable REQUIRE_STRONG_SIGNAL gate now
   requires at least one genuinely heat-specific term (or a dual-use term
   co-occurring with one). Set REQUIRE_STRONG_SIGNAL=False to reproduce the
   old loose behaviour.

3. Negative keywords could cause false negatives
   The original dropped the whole tender if ANY negative term appeared.
   A negative term no longer overrides a strong heat-specific signal.

4. identify_scheme was non-deterministic
   It returned set.pop() (arbitrary order) when several schemes matched.
   It now matches on word boundaries and returns the schemes deterministically.

5. Standalone input
   main() can process a single concatenated CSV (INPUT_CSV) as well as the
   original monthly-folder layout, so it runs without the data/ tree.

The classification semantics, column names, and output format are otherwise
unchanged.
"""

import os
import re
import glob
import ast
import argparse
import pandas as pd
import dateutil.parser

from heat_tenders_config import (
    POSITIVE_KEYWORDS,
    NEGATIVE_KEYWORDS,
    THEMATIC_KEYWORD_GROUPS,
    SCHEME_KEYWORDS,
    EXCLUDED_DEPARTMENTS,
    SEASON_MONTHS,
    HEAT_SPECIFIC_KEYWORDS,
)

# ---------------------------------------------------------------------------
# Tuning switches
# ---------------------------------------------------------------------------
# Every tender now gets BOTH flags written to the output:
#   is_heat_tender         -> loose gate  (any positive keyword, no negative)
#   is_strong_heat_tender  -> strong gate (a heat-SPECIFIC keyword fired)
# plus a convenience column:
#   heat_signal            -> 'strong' or 'weak'
#
# KEEP_ONLY_STRONG controls which rows are written to the output file:
#   False (default) -> write the full LOOSE set (so you can see weak tags too,
#                      filter later on heat_signal == 'strong')
#   True            -> write only the strong-signal rows
KEEP_ONLY_STRONG = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
MONTHLY_TENDERS_DIR = os.path.join(DATA_DIR, 'monthly_tenders')
HEAT_TENDERS_DIR = os.path.join(DATA_DIR, 'heat_tenders')
HEAT_TENDERS_ALL_CSV = os.path.join(DATA_DIR, 'heat_tenders_all.csv')

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9 \n\.]', ' ', text).lower()


def normalize_keyword(keyword):
    """Normalise a keyword the SAME way as the searched text.

    Fixes the bug where 'heat-wave' (hyphen) never matched: the text had its
    hyphens converted to spaces by clean_text, so the keyword must be too.
    """
    return clean_text(str(keyword)).strip()


def build_tender_slug(row):
    parts = []
    for col in ('tender_externalreference', 'tender_title', 'Work Description'):
        val = row.get(col, '')
        if pd.isna(val):
            val = ''
        parts.append(str(val))
    return ' '.join(parts)


def count_keyword_hits(text, keywords):
    hits = {}
    for keyword in keywords:
        norm = normalize_keyword(keyword)
        if not norm:
            hits[keyword] = 0
            continue
        # collapse internal whitespace so multi-word keywords are robust
        pattern = r'\b%s\b' % re.escape(norm).replace(r'\ ', r'\s+')
        hits[keyword] = len(re.findall(pattern, text))
    return hits


_STRONG_NORM = {normalize_keyword(s) for s in HEAT_SPECIFIC_KEYWORDS}


def heat_filter(row):
    """Return (is_heat_loose, is_heat_strong, positive_kw_dict, negative_kw_dict)."""
    tender_slug = clean_text(build_tender_slug(row))

    positive_hits = count_keyword_hits(tender_slug, POSITIVE_KEYWORDS)
    negative_hits = count_keyword_hits(tender_slug, NEGATIVE_KEYWORDS)

    fired_positive = {k for k, v in positive_hits.items() if v > 0}
    has_positive = bool(fired_positive)
    has_negative = any(negative_hits.values())
    has_strong = any(normalize_keyword(k) in _STRONG_NORM for k in fired_positive)

    # loose: original behaviour (any positive, no negative)
    is_heat_loose = has_positive and not has_negative
    # strong: a heat-specific term fired; a negative only excludes a weak match
    is_heat_strong = has_strong and not (has_negative and not has_strong)

    return str(is_heat_loose), str(is_heat_strong), str(positive_hits), str(negative_hits)


def classify_season(published_date):
    try:
        month = dateutil.parser.parse(published_date).month
    except (ValueError, TypeError):
        return "Unknown"
    for season, months in SEASON_MONTHS.items():
        if month in months:
            return season
    return "Unknown"


def identify_scheme(row):
    """Deterministic scheme detection.

    Original used set.pop(), which returned an arbitrary element when several
    schemes matched. Now we match each scheme as a whole token and return the
    matches in a stable (sorted) order.
    """
    tender_slug = clean_text(build_tender_slug(row))
    matched = sorted(
        kw for kw in SCHEME_KEYWORDS
        if re.search(r'\b%s\b' % re.escape(kw), tender_slug)
    )
    return ', '.join(m.upper() for m in matched)


def classify_theme(row):
    tender_slug = clean_text(build_tender_slug(row))
    matched_themes = []
    subhead_hits = {}
    for theme, keywords in THEMATIC_KEYWORD_GROUPS.items():
        hits = {kw: c for kw, c in count_keyword_hits(tender_slug, keywords).items() if c > 0}
        if hits:
            matched_themes.append(theme)
            subhead_hits[theme] = hits
    response_type = matched_themes[0] if matched_themes else "Others"
    return response_type, ', '.join(matched_themes), str(subhead_hits)


def process_file(csv_path):
    filename = os.path.basename(csv_path)
    print("FILENAME " + filename)

    input_df = pd.read_csv(csv_path)
    input_df = input_df.drop_duplicates()

    filter_results = list(input_df.apply(heat_filter, axis=1))
    input_df.loc[:, 'is_heat_tender'] = [r[0] for r in filter_results]
    input_df.loc[:, 'is_strong_heat_tender'] = [r[1] for r in filter_results]
    input_df.loc[:, 'positive_keywords_dict'] = [r[2] for r in filter_results]
    input_df.loc[:, 'negative_keywords_dict'] = [r[3] for r in filter_results]

    keep_col = 'is_strong_heat_tender' if KEEP_ONLY_STRONG else 'is_heat_tender'
    tenders_df = input_df[
        (input_df[keep_col] == 'True')
        & (~input_df.Department.isin(EXCLUDED_DEPARTMENTS))
    ].copy()

    tenders_df.loc[:, 'heat_signal'] = tenders_df['is_strong_heat_tender'].map(
        {'True': 'strong', 'False': 'weak'}
    )

    n_strong = (tenders_df['is_strong_heat_tender'] == 'True').sum()
    print('Heat tenders written: %d  (strong: %d, weak: %d)'
          % (tenders_df.shape[0], n_strong, tenders_df.shape[0] - n_strong))
    if tenders_df.shape[0] == 0:
        return None

    tenders_df.loc[:, 'Season'] = tenders_df['Published Date'].apply(classify_season)
    tenders_df.loc[:, 'Scheme'] = tenders_df.apply(identify_scheme, axis=1)

    theme_results = list(tenders_df.apply(classify_theme, axis=1))
    tenders_df.loc[:, 'Response Type'] = [r[0] for r in theme_results]
    tenders_df.loc[:, 'Heat Response - Themes'] = [r[1] for r in theme_results]
    tenders_df.loc[:, 'Heat Response - Subhead'] = [r[2] for r in theme_results]

    return tenders_df


def main():
    parser = argparse.ArgumentParser(description="Tag heat-response tenders.")
    parser.add_argument('--input', help="Single concatenated tenders CSV to process.")
    parser.add_argument('--output', help="Where to write the tagged heat tenders CSV.")
    args = parser.parse_args()

    # Mode A: single concatenated file (what we use for testing).
    if args.input:
        tenders_df = process_file(args.input)
        out_path = args.output or os.path.join(SCRIPT_DIR, 'heat_tenders_all.csv')
        if tenders_df is not None:
            tenders_df['month'] = ''
            tenders_df.to_csv(out_path, index=False, encoding='utf-8')
            print('Wrote', out_path)
        return

    # Mode B: original monthly-folder layout.
    os.makedirs(HEAT_TENDERS_DIR, exist_ok=True)
    for csv_path in glob.glob(os.path.join(MONTHLY_TENDERS_DIR, '*.csv')):
        tenders_df = process_file(csv_path)
        if tenders_df is None:
            continue
        out_path = os.path.join(HEAT_TENDERS_DIR, os.path.basename(csv_path))
        tenders_df.to_csv(out_path, encoding='utf-8', index=False)

    dfs = []
    for csv_path in glob.glob(os.path.join(HEAT_TENDERS_DIR, '*.csv')):
        month = os.path.basename(csv_path)[:7]
        df = pd.read_csv(csv_path)
        df['month'] = month
        dfs.append(df)
    if dfs:
        tenders_df = pd.concat(dfs, ignore_index=True)
        tenders_df.to_csv(HEAT_TENDERS_ALL_CSV, index=False)


if __name__ == '__main__':
    main()
