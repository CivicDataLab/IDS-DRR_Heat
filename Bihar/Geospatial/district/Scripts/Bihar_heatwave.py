import ee
import datetime

# Initialize Earth Engine
ee.Initialize()

# ----------------------------
# 1. Load Bihar boundary
# ----------------------------
# FAO GAUL level1 (India states)
states = ee.FeatureCollection("FAO/GAUL/2015/level1")
bihar = states.filter(ee.Filter.eq('ADM1_NAME', 'Bihar')).geometry()


start_date = '2025-03-01'
end_date   = '2025-06-30'

dataset = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
           .filterDate(start_date, end_date)
           .filterBounds(bihar))


def extract_stats(img):
    tmin_c = img.select('temperature_2m_min').subtract(273.15)
    tmax_c = img.select('temperature_2m_max').subtract(273.15)

    # Heatwave threshold (Tmax >= 40°C)
    heat_flag = tmax_c.gte(40)

    stats = img.addBands(tmin_c.rename('tmin_c')) \
               .addBands(tmax_c.rename('tmax_c')) \
               .addBands(heat_flag.rename('heatwave_day')) \
               .reduceRegion(
                   reducer=ee.Reducer.mean(),
                   geometry=bihar,
                   scale=10000
               )

    return ee.Feature(None, stats).set('date', img.date().format('YYYY-MM-dd'))

# Apply function to all days
daily_stats = dataset.map(extract_stats)

# ----------------------------
# 5. Convert to list + fetch
# ----------------------------
features = ee.FeatureCollection(daily_stats).getInfo()['features']

# ----------------------------
# 6. Python-side Heatwave Event Detection
# ----------------------------
heatwave_data = []
for f in features:
    props = f['properties']
    date  = props['date']
    tmin  = props['tmin_c']
    tmax  = props['tmax_c']
    flag  = int(props['heatwave_day'])
    heatwave_data.append([date, tmin, tmax, flag])

import pandas as pd
df = pd.DataFrame(heatwave_data, columns=['date','tmin_c','tmax_c','heatwave_day'])
df['date'] = pd.to_datetime(df['date'])

# Identify consecutive heatwave sequences
df['event_group'] = (df['heatwave_day'].diff().ne(0)).cumsum()
heat_events = df[df['heatwave_day']==1].groupby('event_group')

# Build event-level summary
events_summary = []
for _, group in heat_events:
    if len(group) >= 2:  # IMD requires >=2 days
        events_summary.append({
            'start_date': group['date'].iloc[0],
            'end_date': group['date'].iloc[-1],
            'duration_days': len(group),
            'mean_tmax': group['tmax_c'].mean(),
            'max_tmax': group['tmax_c'].max()
        })

events_df = pd.DataFrame(events_summary)

# ----------------------------
# 7. Save outputs
# ----------------------------
df.to_csv("bihar_daily_temp.csv", index=False)
events_df.to_csv("bihar_heatwave_events.csv", index=False)

print("✅ Bihar daily temps + heatwave events saved!")