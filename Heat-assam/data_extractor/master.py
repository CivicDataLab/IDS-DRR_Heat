import os
import pandas as pd
from pathlib import Path
import re
import glob

main_directory = Path.cwd() / 'Heat-assam/data_extractor/'
#print(main_directory)


# Iterate through all folders and sub-folders
for root, dirs, files in os.walk(main_directory):
    root_path = Path(root)
    #print(root_path)
    
    if 'variables' in root_path.parts:
        csv_files = list(root_path.glob('*.csv'))  # No need for '**/*.csv'
        dfs = []
        for csv in csv_files:
            
                #ignore plfs
            if 'plfs' in str(root_path).lower():
                continue

            #csv = csv.resolve()
                #one-time
            if any(folder in str(csv.parts) for folder in ['BHARATMAPS', 'ANTYODAYA']):
                timeperiod = ''
                file_name = csv.stem
                # Annual 
            elif any(folder in str(csv.parts) for folder in ['WORLDPOP']):
                year_match = re.findall(r'\d{4}', csv.name)
                #if year_match:
                timeperiod = year_match[0]
                file_name = csv.stem[:-5]
                
            # elif any("SENTINEL" in str(parent) for parent in csv.parents):
            #     date_match = re.findall(r'\d{4}-\d{2}-\d{2}', csv.name)
            #     #if date_match:
            #     timeperiod = date_match[0][:-3].replace('-', '_')
            #     file_name = csv.stem
                            
            else:
                date_match = re.findall(r'\d{4}_\d{2}', csv.name)
                
                if date_match:
                    timeperiod = date_match[0]
                #if len(csv.stem) < 16:
                #    file_name = csv.stem  # No slicing if the filename is short
                #else:
                #    file_name = csv.stem[:-8] # Use full stem if there's no underscore

                file_name = csv.stem[:-8]
                

            print("file: ", file_name)
            df = pd.read_csv(csv)
            df['timeperiod'] = timeperiod
            dfs.append(df)

        if dfs:  # Check if there are any dataframes to concatenate
            master_df = pd.concat(dfs)
            master_df.to_csv(main_directory / f'master/{file_name}.csv', index=False)

# IMD
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

# BHUVAN
# path = main_directory / 'BHUVAN/data/variables/inundation_pct'
# csvs = glob.glob(str(path / '*.csv'))
# dfs = []
# for csv in csvs:
#     month = re.findall(r'\d{4}_\d{2}', csv)[0]
#     df = pd.read_csv(csv)
#     df['timeperiod'] = month
#     dfs.append(df)

# master_df = pd.concat(dfs)
# master_df.to_csv(main_directory / 'master/inundation.csv', index=False)


# ERA5-LAND HEAT DAYS
df = pd.read_csv(
    main_directory / 'era5_land/data/variables/heatdays.csv'
)

df.to_csv(
    main_directory / 'master/heatdays.csv',
    index=False
)

#  PLFS
df = pd.read_csv(
    main_directory / 'plfs/data/variables/plfs_sunexposed_pct.csv'
)

df.to_csv(
    main_directory / 'master/plfs_sunexposed_pct.csv',
    index=False
)

# NFHS
df = pd.read_csv(
    main_directory / 'nfhs/data/variables/nfhs_ncd_pct.csv'
)

df.to_csv(
    main_directory / 'master/nfhs_ncd_pct.csv',
    index=False
)

# lst
df = pd.read_csv(
    main_directory / 'modis_aqua/data/variables/lst_raster.csv'
)
df.to_csv(
    main_directory / 'master/lst_raster.csv',
    index=False
)


df = pd.read_csv(
    main_directory / 'modis_aqua/data/variables/land_surface_temperature.csv'
)

df.to_csv(
    main_directory / 'master/land_surface_temperature.csv',
    index=False
)
