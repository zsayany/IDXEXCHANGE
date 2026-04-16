import pandas as pd
from pathlib import Path
import re

# =========================================================
# WEEK 1 - MONTHLY DATASET AGGREGATION
# Combines all monthly MLS files from Jan 2024 through the
# most recently completed calendar month, filters to
# PropertyType == "Residential", and saves new CSVs.
# =========================================================

# Your project structure (from screenshot):
# IDX2/
#   raw/   <-- most CSV files live here
#   sql/
#   IDX_Exchange_Intern_Handbook_Final-...
#
# So we point data_dir to the raw folder:
data_dir = Path("raw")

output_dir = data_dir / "combined_outputs"
output_dir.mkdir(exist_ok=True)

today = pd.Timestamp.today()
most_recent_completed_month = (today.replace(day=1) - pd.DateOffset(months=1)).strftime("%Y%m")

sold_pattern = re.compile(r"CRMLSSold(\d{6})\.csv$", re.IGNORECASE)
listing_pattern = re.compile(r"CRMLSListing(\d{6})\.csv$", re.IGNORECASE)

def get_matching_files(folder, pattern, start_yyyymm="202401", end_yyyymm=most_recent_completed_month):
    matched = []
    for file in folder.glob("*.csv"):
        match = pattern.match(file.name)
        if match:
            yyyymm = match.group(1)
            if start_yyyymm <= yyyymm <= end_yyyymm:
                matched.append((yyyymm, file))
    matched.sort(key=lambda x: x[0])
    return [file for _, file in matched]

sold_files = get_matching_files(data_dir, sold_pattern)
listing_files = get_matching_files(data_dir, listing_pattern)

print("=" * 60)
print("FILES FOUND")
print("=" * 60)
print(f"Most recently completed month used: {most_recent_completed_month}")
print(f"Sold files found: {len(sold_files)}")
print(f"Listing files found: {len(listing_files)}")
print()

if len(sold_files) == 0:
    raise FileNotFoundError("No sold files found in 'raw'. Check file names.")
if len(listing_files) == 0:
    raise FileNotFoundError("No listing files found in 'raw'. Check file names.")

def combine_files(file_list, dataset_name):
    all_dfs = []
    monthly_row_counts = []

    print("=" * 60)
    print(f"LOADING {dataset_name.upper()} FILES")
    print("=" * 60)

    for file in file_list:
        df = pd.read_csv(file, low_memory=False)
        row_count = len(df)
        monthly_row_counts.append((file.name, row_count))
        all_dfs.append(df)
        print(f"{file.name}: {row_count:,} rows")

    total_rows_before_concat = sum(rows for _, rows in monthly_row_counts)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    total_rows_after_concat = len(combined_df)

    print()
    print(f"{dataset_name} total rows before concatenation: {total_rows_before_concat:,}")
    print(f"{dataset_name} total rows after concatenation:  {total_rows_after_concat:,}")
    print()

    return combined_df, monthly_row_counts, total_rows_before_concat, total_rows_after_concat

sold_df, sold_monthly_counts, sold_before_concat, sold_after_concat = combine_files(sold_files, "sold")
listing_df, listing_monthly_counts, listing_before_concat, listing_after_concat = combine_files(listing_files, "listing")

if "PropertyType" not in sold_df.columns:
    raise KeyError("PropertyType column not found in sold dataset.")
if "PropertyType" not in listing_df.columns:
    raise KeyError("PropertyType column not found in listing dataset.")

sold_before_filter = len(sold_df)
listing_before_filter = len(listing_df)

sold_residential = sold_df[sold_df["PropertyType"].astype(str).str.strip().str.lower() == "residential"].copy()
listing_residential = listing_df[listing_df["PropertyType"].astype(str).str.strip().str.lower() == "residential"].copy()

sold_after_filter = len(sold_residential)
listing_after_filter = len(listing_residential)

print("=" * 60)
print("RESIDENTIAL FILTER RESULTS")
print("=" * 60)
print(f"Sold rows before Residential filter:    {sold_before_filter:,}")
print(f"Sold rows after Residential filter:     {sold_after_filter:,}")
print(f"Listings rows before Residential filter:{listing_before_filter:,}")
print(f"Listings rows after Residential filter: {listing_after_filter:,}")
print()

sold_output_file = output_dir / f"CRMLSSold_Combined_Residential_202401_to_{most_recent_completed_month}.csv"
listing_output_file = output_dir / f"CRMLSListing_Combined_Residential_202401_to_{most_recent_completed_month}.csv"

sold_residential.to_csv(sold_output_file, index=False)
listing_residential.to_csv(listing_output_file, index=False)

summary_lines = [
    "WEEK 1 ROW COUNT SUMMARY",
    "=" * 60,
    f"Most recently completed month used: {most_recent_completed_month}",
    "",
    "SOLD DATASET",
    f"Files included: {len(sold_files)}",
    f"Rows before concatenation (sum of all monthly files): {sold_before_concat:,}",
    f"Rows after concatenation: {sold_after_concat:,}",
    f"Rows before Residential filter: {sold_before_filter:,}",
    f"Rows after Residential filter: {sold_after_filter:,}",
    "",
    "LISTING DATASET",
    f"Files included: {len(listing_files)}",
    f"Rows before concatenation (sum of all monthly files): {listing_before_concat:,}",
    f"Rows after concatenation: {listing_after_concat:,}",
    f"Rows before Residential filter: {listing_before_filter:,}",
    f"Rows after Residential filter: {listing_after_filter:,}",
    "",
    "MONTHLY SOLD FILES",
]

for file_name, rows in sold_monthly_counts:
    summary_lines.append(f"{file_name}: {rows:,} rows")

summary_lines.append("")
summary_lines.append("MONTHLY LISTING FILES")

for file_name, rows in listing_monthly_counts:
    summary_lines.append(f"{file_name}: {rows:,} rows")

summary_lines.append("")
summary_lines.append(f"Sold output file: {sold_output_file}")
summary_lines.append(f"Listing output file: {listing_output_file}")

summary_file = output_dir / "week1_row_count_summary.txt"
with open(summary_file, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("=" * 60)
print("FILES SAVED")
print("=" * 60)
print(f"Sold combined file:    {sold_output_file}")
print(f"Listing combined file: {listing_output_file}")
print(f"Summary file:          {summary_file}")
print()
print("Done.")