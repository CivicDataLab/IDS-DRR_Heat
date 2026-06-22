import pandas as pd
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
input_dir = Path("/Users/stephensmathew/assignment/heat_files")

output_csv = "assam_heat_aggregated_by_object_id.csv"

required_cols = [
    "object_id",
    "tmax_abs_c",
    "heatwave_days"
]

# --------------------------------------------------
# 1. READ & CONCATENATE ALL CSV FILES
# --------------------------------------------------
dfs = []

for csv_file in input_dir.glob("*.csv"):
    print(f"📄 Reading: {csv_file.name}")
    df = pd.read_csv(csv_file)

    # Standardise column names
    df.columns = df.columns.str.strip().str.lower()

    # Validate required columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing columns {missing} in {csv_file.name}")

    dfs.append(df[required_cols])

if not dfs:
    raise ValueError("❌ No CSV files found in the directory.")

combined_df = pd.concat(dfs, ignore_index=True)

# --------------------------------------------------
# 2. ENSURE NUMERIC TYPES
# --------------------------------------------------
combined_df["tmax_abs_c"] = pd.to_numeric(
    combined_df["tmax_abs_c"], errors="coerce"
)

combined_df["heatwave_days"] = pd.to_numeric(
    combined_df["heatwave_days"], errors="coerce"
)

# --------------------------------------------------
# 3. GROUP BY OBJECT_ID & AGGREGATE (CORRECT LOGIC)
# --------------------------------------------------
agg_df = (
    combined_df
    .groupby("object_id", as_index=False)
    .agg({
        "tmax_abs_c": "max",        # ✅ peak heat intensity
        "heatwave_days": "sum"     # ✅ cumulative exposure
    })
)

# --------------------------------------------------
# 4. SAVE OUTPUT
# --------------------------------------------------
agg_df.to_csv(output_csv, index=False)

print(f"✅ Aggregated CSV saved at:\n{output_csv}")
print(f"🧮 Total objects: {len(agg_df)}")
