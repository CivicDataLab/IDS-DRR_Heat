import ee
import geopandas as gpd
import pandas as pd

# =================================================
# CONFIG
# =================================================
# Target years for analysis
YEARS = range(2020, 2025)

# Baseline period for 90th percentile calculation (30-year climatology)
BASELINE_START = "1991-01-01"
BASELINE_END = "2020-12-31"

# Input GeoJSON
GEOJSON_FILE = "Bihar_District_Boundary_final.geojson"

# District name field
DISTRICT_FIELD = "dtname"

# Output files
DAILY_OUTPUT = "bihar_daily_temp_districts_2020_2025.csv"
YEARLY_OUTPUT = "bihar_heatwave_counts_districts_2020_2025.csv"
MONTHLY_OUTPUT = "bihar_heatwave_counts_monthly_2020_2025.csv"
THRESHOLD_OUTPUT = "bihar_district_heatwave_thresholds.csv"

# =================================================
# INITIALIZE EARTH ENGINE
# =================================================
ee.Initialize()

# =================================================
# LOAD DISTRICTS
# =================================================
print("Loading district boundaries...")
gdf = gpd.read_file(GEOJSON_FILE)
gdf = gdf.to_crs(epsg=4326)

if DISTRICT_FIELD not in gdf.columns:
    raise ValueError(f"❌ Missing required column: {DISTRICT_FIELD}")

print(f"Loaded {len(gdf)} districts")

# =================================================
# HELPER: SHAPELY → EE GEOMETRY
# =================================================
def shapely_to_ee(geom):
    """Convert Shapely geometry to Earth Engine geometry"""
    if geom.geom_type == 'Polygon':
        return ee.Geometry.Polygon(list(geom.exterior.coords))
    elif geom.geom_type == 'MultiPolygon':
        return ee.Geometry.MultiPolygon(
            [list(p.exterior.coords) for p in geom.geoms]
        )
    else:
        return None

# =================================================
# STEP 1: CALCULATE 90TH PERCENTILE THRESHOLDS
# =================================================
print("\n" + "="*60)
print("STEP 1: Calculating 90th percentile thresholds per district")
print("="*60)

# Load baseline ERA5 data
era5_baseline = (
    ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    .filterDate(BASELINE_START, BASELINE_END)
    .select("temperature_2m_max")
)

# Convert Kelvin to Celsius
def kelvin_to_celsius(img):
    return img.subtract(273.15).copyProperties(img, img.propertyNames())

era5_baseline = era5_baseline.map(kelvin_to_celsius)

district_thresholds = {}

for idx, row in gdf.iterrows():
    ee_geom = shapely_to_ee(row.geometry)
    if ee_geom is None:
        print(f"⚠️ Skipping {row[DISTRICT_FIELD]} - invalid geometry")
        continue
    
    district_name = str(row[DISTRICT_FIELD]).upper()
    
    try:
        # Calculate 90th percentile for this district
        percentile_90 = (
            era5_baseline
            .reduce(ee.Reducer.percentile([90]))
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_geom,
                scale=10000,
                maxPixels=1e9
            )
            .getInfo()
        )
        
        threshold = percentile_90.get("temperature_2m_max_p90")
        
        if threshold is not None:
            # IMD criterion: For plains, minimum 40°C
            threshold = max(threshold, 40.0)
            district_thresholds[district_name] = round(threshold, 2)
            print(f"  ✓ {district_name}: {threshold:.2f}°C (90th percentile)")
        else:
            print(f"  ⚠️ {district_name}: Using default 40°C")
            district_thresholds[district_name] = 40.0
            
    except Exception as e:
        print(f"  ⚠️ {district_name}: Error - {e}, using default 40°C")
        district_thresholds[district_name] = 40.0

# Save thresholds to CSV
threshold_df = pd.DataFrame([
    {'district': k, 'threshold_90p_c': v} 
    for k, v in district_thresholds.items()
])
threshold_df.to_csv(THRESHOLD_OUTPUT, index=False)
print(f"\n✅ Thresholds saved to: {THRESHOLD_OUTPUT}")

# =================================================
# STEP 2: PROCESS DAILY DATA FOR EACH DISTRICT & YEAR
# =================================================
print("\n" + "="*60)
print("STEP 2: Processing daily temperature data")
print("="*60)

def process_district_year(geom, props, year, threshold):
    """Process one district for one year with district-specific threshold"""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    district_name = str(props.get(DISTRICT_FIELD, "UNKNOWN")).upper()

    # Convert geometry to EE
    ee_geom = shapely_to_ee(geom)
    if ee_geom is None:
        return pd.DataFrame()

    district_fc = ee.FeatureCollection([ee.Feature(ee_geom, props)])

    # ERA5 dataset
    dataset = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start_date, end_date)
        .filterBounds(ee_geom)
    )

    # Add Tmin, Tmax, Heatwave flag (district-specific threshold)
    def add_bands(img):
        tmin_c = img.select('temperature_2m_min').subtract(273.15).rename('tmin_c')
        tmax_c = img.select('temperature_2m_max').subtract(273.15).rename('tmax_c')
        heat_flag = tmax_c.gte(threshold).rename('heatwave_day')
        return img.addBands([tmin_c, tmax_c, heat_flag])

    dataset = dataset.map(add_bands)

    # Zonal statistics for this district
    def zonal_stats(img):
        stats = img.reduceRegions(
            collection=district_fc,
            reducer=ee.Reducer.mean(),
            scale=10000
        ).map(lambda f: f.set('date', img.date().format('YYYY-MM-dd')))
        return stats

    try:
        zonal = dataset.map(zonal_stats).flatten()
        features = zonal.getInfo()['features']

        rows = []
        for f in features:
            p = f['properties']
            rows.append({
                'date': p['date'],
                'year': year,
                'district': district_name,
                'threshold_90p_c': threshold,
                'tmax_c': round(p.get('tmax_c', None), 2) if p.get('tmax_c') is not None else None,
                'tmin_c': round(p.get('tmin_c', None), 2) if p.get('tmin_c') is not None else None,
                'heatwave_day': int(round(p.get('heatwave_day', 0))) if p.get('heatwave_day') is not None else 0
            })

        return pd.DataFrame(rows)
    
    except Exception as e:
        print(f"  ⚠️ Error processing {district_name} for {year}: {e}")
        return pd.DataFrame()

# =================================================
# LOOP THROUGH ALL DISTRICTS & YEARS
# =================================================
all_districts = []

for idx, row in gdf.iterrows():
    district_name = str(row[DISTRICT_FIELD]).upper()
    threshold = district_thresholds.get(district_name, 40.0)
    
    print(f"\nProcessing {district_name} (threshold: {threshold}°C)...")
    
    for yr in YEARS:
        print(f"  → Year {yr}...", end=" ")
        df_year = process_district_year(
            row['geometry'], 
            row.drop('geometry').to_dict(), 
            yr,
            threshold
        )
        all_districts.append(df_year)
        print(f"✓ ({len(df_year)} days)")

# =================================================
# STEP 3: COMBINE AND SAVE RESULTS
# =================================================
print("\n" + "="*60)
print("STEP 3: Saving results")
print("="*60)

df = pd.concat(all_districts, ignore_index=True)
df['date'] = pd.to_datetime(df['date'])

# Save daily dataset
df.to_csv(DAILY_OUTPUT, index=False)
print(f"✅ Daily dataset saved: {DAILY_OUTPUT}")

# =================================================
# STEP 4: YEARLY HEATWAVE COUNTS
# =================================================
yearly_counts = (
    df.groupby(['district', 'year'])['heatwave_day']
    .sum()
    .reset_index()
)
yearly_counts.rename(columns={'heatwave_day': 'heatwave_days_count'}, inplace=True)

# Add threshold information
yearly_counts = yearly_counts.merge(
    threshold_df, 
    on='district', 
    how='left'
)

yearly_counts.to_csv(YEARLY_OUTPUT, index=False)
print(f"✅ Yearly heatwave counts saved: {YEARLY_OUTPUT}")

# =================================================
# STEP 5: MONTHLY HEATWAVE COUNTS
# =================================================
df['month'] = df['date'].dt.month

monthly_counts = (
    df.groupby(['district', 'year', 'month'])['heatwave_day']
    .sum()
    .reset_index()
)
monthly_counts.rename(columns={'heatwave_day': 'heatwave_days_count'}, inplace=True)

# Add threshold information
monthly_counts = monthly_counts.merge(
    threshold_df, 
    on='district', 
    how='left'
)

monthly_counts.to_csv(MONTHLY_OUTPUT, index=False)
print(f"✅ Monthly heatwave counts saved: {MONTHLY_OUTPUT}")

# =================================================
# SUMMARY STATISTICS
# =================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total districts processed: {len(gdf)}")
print(f"Years analyzed: {min(YEARS)}-{max(YEARS)}")
print(f"Total daily records: {len(df):,}")
print(f"\nThreshold range: {threshold_df['threshold_90p_c'].min():.2f}°C - {threshold_df['threshold_90p_c'].max():.2f}°C")
print(f"Average threshold: {threshold_df['threshold_90p_c'].mean():.2f}°C")
print(f"\nTotal heatwave days (all districts, all years): {df['heatwave_day'].sum():,.0f}")
print(f"Average heatwave days per district per year: {yearly_counts['heatwave_days_count'].mean():.1f}")

# Top 5 districts by heatwave days
print("\nTop 5 districts by total heatwave days (2020-2024):")
top_districts = (
    yearly_counts.groupby('district')['heatwave_days_count']
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
for dist, count in top_districts.items():
    print(f"  {dist}: {count:.0f} days")

print("\n" + "="*60)
print("✅ ALL PROCESSING COMPLETE")
print("="*60)