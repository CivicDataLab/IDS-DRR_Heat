import os
import pandas as pd
from pathlib import Path
import re
import glob

# main_directory = Path.cwd() / "data_extractor"
main_directory = Path.cwd()
# print(main_directory)



# ONLY keep these two sources
KEEP = ['BHARATMAPS', 'ANTYODAYA', 'WORLDPOP']


# Iterate through all folders and sub-folders
# Iterate through all folders and sub-folders
for root, dirs, files in os.walk(main_directory):
    root_path = Path(root)

    # Skip everything except the desired sources
    if not any(k in root_path.parts for k in KEEP):
        continue

    if 'variables' not in root_path.parts:
        continue

    csv_files = list(root_path.glob("*.csv"))
    dfs = []

    for csv in csv_files:

        print(csv)

        if 'WORLDPOP' in csv.parts:
            print("WORLDPOP")
        elif any(folder in csv.parts for folder in ['BHARATMAPS', 'ANTYODAYA']):
            print("BHARATMAPS/ANTYODAYA")
        elif any("SENTINEL" in str(parent) for parent in csv.parents):
            print("SENTINEL")
        else:
            print("DEFAULT")

        # ------------------------------------
        # Determine file_name and timeperiod
        # ------------------------------------
        if 'WORLDPOP' in csv.parts:

            year_match = re.findall(r'\d{4}', csv.name)
            if not year_match:
                raise ValueError(f"No year found in {csv.name}")

            timeperiod = year_match[0]
            file_name = csv.stem.replace(f"_{timeperiod}", "")

        elif any(folder in csv.parts for folder in ['BHARATMAPS', 'ANTYODAYA']):

            timeperiod = ""
            file_name = csv.stem

        elif any("SENTINEL" in str(parent) for parent in csv.parents):

            date_match = re.findall(r'\d{4}-\d{2}-\d{2}', csv.name)
            if not date_match:
                raise ValueError(f"No Sentinel date found in {csv.name}")

            timeperiod = date_match[0][:-3].replace("-", "_")
            file_name = csv.stem

        else:

            date_match = re.findall(r'\d{4}_\d{2}', csv.name)
            if not date_match:
                raise ValueError(f"No timeperiod found in {csv.name}")

            timeperiod = date_match[0]
            file_name = csv.stem[:-8]

        print("file:", file_name)
        print("timeperiod:", timeperiod)

        df = pd.read_csv(csv)
        df["timeperiod"] = timeperiod
        dfs.append(df)

    if len(dfs) == 0:
        continue

    master_df = pd.concat(dfs, ignore_index=True)

    print(file_name)
    print(master_df["timeperiod"].unique())
    print(master_df["timeperiod"].value_counts().sort_index())

    master_df.to_csv(
        main_directory / f"data_extractor/master/{file_name}.csv",
        index=False
    )

# # IMD (unchanged)
# path = main_directory / 'IMD/data/rain/csv'
# csvs = glob.glob(str(path / '*.csv'))
# dfs = []

# for csv in csvs:
#     month = re.findall(r'\d{4}_\d{2}', csv)[0]
#     df = pd.read_csv(csv)
#     df['timeperiod'] = month
#     dfs.append(df)

# master_df = pd.concat(dfs)
# master_df = master_df.rename(columns={'max': 'max_rain', 'mean':'mean_rain', 'sum':'sum_rain'})
# master_df.to_csv(main_directory / 'master/rainfall.csv', index=False)


# # BHUVAN (unchanged)
# path = main_directory / 'BHUVAN/data/variables/inundation_pct'
# csvs = glob.glob(str(path / '*.csv'))
# dfs = []

# for csv in csvs:
#     month = re.findall(r'\d{4}_\d{2}', csv)[0]
#     df = pd.read_csv(csv)
#     df['timeperiod'] = month
#     dfs.append(df)

# if dfs:
#     master_df = pd.concat(dfs)
#     master_df.to_csv(main_directory / 'master/inundation.csv', index=False)


# # NRSC (unchanged)
# path = main_directory / 'NRSC/data/variables/runoff'
# csvs = glob.glob(str(path / '*.csv'))
# dfs = []

# for csv in csvs:
#     month = re.findall(r'\d{4}_\d{2}', csv)[0]
#     df = pd.read_csv(csv)
#     df['timeperiod'] = month
#     dfs.append(df)

# if dfs:
#     master_df = pd.concat(dfs)
#     master_df.to_csv(main_directory / 'master/runoff.csv', index=False)

# ERA5-LAND HEAT DAYS
df = pd.read_csv(
    main_directory / 'data_extractor/era5_land/data/variables/heatdays.csv'
)

df.to_csv(
    main_directory / 'data_extractor/master/heatdays.csv',
    index=False
)


# PLFS
df = pd.read_csv(
    main_directory / 'data_extractor/plfs/data/variables/plfs_sunexposed_pct.csv'
)

df.to_csv(
    main_directory / 'data_extractor/master/plfs_sunexposed_pct.csv',
    index=False
)

# NFHS
df = pd.read_csv(
    main_directory / 'data_extractor/nfhs/data/variables/nfhs_ncd_pct.csv'
)

df.to_csv(
    main_directory / 'data_extractor/master/nfhs_ncd_pct.csv',
    index=False
)

# lst_naming
df = pd.read_csv(
    main_directory / 'data_extractor/modis_aqua/data/variables/lst_raster.csv'
)

df.to_csv(
    main_directory / 'data_extractor/master/lst_raster.csv',
    index=False
)

df = pd.read_csv(
    main_directory / 'data_extractor/modis_aqua/data/variables/land_surface_temperature.csv'
    )
df.to_csv(
    main_directory / 'data_extractor/master/land_surface_temperature.csv', 
    index=False)


# ...existing code...

# TENDERS - total_tender_awarded_value
path = main_directory / 'data_extractor/TENDERS/data/variables/total_tender_awarded_value'
tender_csvs = list(path.glob('total_tender_awarded_value_*.csv'))
tender_dfs = []


for csv in tender_csvs:
    date_match = re.findall(r'(\d{4}_\d{2})', csv.name)
    if date_match:
        timeperiod = date_match[0]
        df = pd.read_csv(csv)
        df['timeperiod'] = timeperiod
        tender_dfs.append(df)

if tender_dfs:
    tender_master_df = pd.concat(tender_dfs)
    tender_master_df.to_csv(main_directory / 'data_extractor/master/total_tender_awarded_value.csv', index=False)

# ...existing code...