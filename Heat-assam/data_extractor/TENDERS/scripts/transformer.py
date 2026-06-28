"""
transformer_heat.py

Heat-tender counterpart of transformer.py (which builds indicators from
Assam's FLOOD tenders). Ports the patched, IO-light indicator builder from
Heat-odisha's transformer.py: tolerant Awarded Value parsing, one indicator
per funding Scheme, and one per heat-response theme. Merges onto the
revenue-circle geometry (assam_rc_2024-11.geojson) instead of Odisha's block
geometry, since Assam's administrative unit below district is the revenue
circle, not the block.
"""

import os
import pandas as pd
import geopandas as gpd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE = os.getcwd()
DATA_PATH = os.path.join(BASE, 'Heat-assam', 'data_extractor', 'TENDERS', 'data')
SHAPEFILE = os.path.join(BASE, 'Heat-assam', 'data_extractor', 'Maps',
                         'Geojson', 'assam_rc_2024-11.geojson')
GEOTAGGED_CSV = os.path.join(DATA_PATH, 'heattenders_RCgeotagged.csv')

# Merge keys (tender side -> shapefile side)
TENDER_DISTRICT_COL, TENDER_RC_COL = 'DISTRICT_FINALISED', 'REVENUE_CIRCLE_FINALISED'
SHAPE_DISTRICT_COL, SHAPE_RC_COL = 'dtname', 'revenue_ci'

# Build indicators only from strong-signal heat tenders when that information
# is available (heat_signal == 'strong', written by heat_tenders.py).
# Set to False to use every row in the geotagged file.
ONLY_STRONG_SIGNAL = False


def parse_awarded_value(df):
    """Return a clean float 'Awarded Value' column, tolerant of either source
    column name, embedded commas, and NaN / already-numeric values."""
    if 'Awarded Price in ₹' in df.columns and 'Awarded Value' not in df.columns:
        df = df.rename(columns={'Awarded Price in ₹': 'Awarded Value'})
    if 'Awarded Value' not in df.columns:
        raise KeyError("Neither 'Awarded Value' nor 'Awarded Price in ₹' found in the geotagged file.")
    df['Awarded Value'] = (
        df['Awarded Value'].astype(str)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    df['Awarded Value'] = pd.to_numeric(df['Awarded Value'], errors='coerce')
    return df


def write_monthly(variable_df, variable, value_col):
    """Write one CSV per month for a single indicator."""
    out_dir = os.path.join(DATA_PATH, 'variables', variable)
    os.makedirs(out_dir, exist_ok=True)
    for year_month in variable_df['month'].dropna().unique():
        monthly = variable_df[variable_df['month'] == year_month][['object_id', value_col]]
        monthly.to_csv(os.path.join(out_dir, '{}_{}.csv'.format(variable, year_month)),
                       index=False)


def build_indicators(df):
    """Core transformation: takes the merged (geotagged) heat tenders frame and
    writes all indicators. Kept IO-light so it can be unit-tested."""
    # 1. Total awarded value per revenue circle per month
    variable = 'total_tender_awarded_value'
    total_df = (df.groupby(['month', 'object_id'])[['Awarded Value']]
                  .sum().reset_index()
                  .rename(columns={'Awarded Value': variable}))
    write_monthly(total_df, variable, variable)
    print('  wrote indicator:', variable)

    # 2. One indicator per heat-response theme (skip NaN / "Others")
    if 'Response Type' in df.columns:
        for rtype in df['Response Type'].dropna().unique():
            if str(rtype).strip() in ('', 'nan', 'None', 'Others'):
                continue
            rdf = df[df['Response Type'] == rtype]
            rdf = rdf.groupby(['month', 'object_id'])[['Awarded Value']].sum().reset_index()
            variable = '{}_tenders_awarded_value'.format(str(rtype).strip())
            rdf = rdf.rename(columns={'Awarded Value': variable})
            write_monthly(rdf, variable, variable)
            print('  wrote indicator:', variable)


def main():
    rc_gdf = gpd.read_file(SHAPEFILE)
    heat_df = pd.read_csv(GEOTAGGED_CSV)

    if ONLY_STRONG_SIGNAL and 'heat_signal' in heat_df.columns:
        before = len(heat_df)
        heat_df = heat_df[heat_df['heat_signal'] == 'strong'].copy()
        print('Restricted to strong-signal heat tenders: {} -> {} rows'.format(before, len(heat_df)))
    elif ONLY_STRONG_SIGNAL:
        print("WARNING: ONLY_STRONG_SIGNAL=True but no 'heat_signal' column found; "
              "using all rows in the geotagged file.")

    # Normalize case before merging, same as Odisha's block merge: tender-side
    # district/RC values and the shapefile's dtname/revenue_ci can differ in
    # case (e.g. "Kokrajhar" vs "KOKRAJHAR").
    heat_df['_merge_district'] = heat_df[TENDER_DISTRICT_COL].astype(str).str.upper().str.strip()
    heat_df['_merge_rc'] = heat_df[TENDER_RC_COL].astype(str).str.upper().str.strip()
    rc_gdf['_merge_district'] = rc_gdf[SHAPE_DISTRICT_COL].astype(str).str.upper().str.strip()
    rc_gdf['_merge_rc'] = rc_gdf[SHAPE_RC_COL].astype(str).str.upper().str.strip()

    heat_df = heat_df.merge(
        rc_gdf,
        on=['_merge_district', '_merge_rc'],
        how='left',
    )
    heat_df = parse_awarded_value(heat_df)
    build_indicators(heat_df)
    print('Done.')


if __name__ == '__main__':
    main()
