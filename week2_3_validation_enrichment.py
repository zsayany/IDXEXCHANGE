import pandas as pd
import numpy as np
from pathlib import Path

# WEEKS 2-3 

data_dir = Path("raw")
combined_dir = data_dir / "combined_outputs"
output_dir = data_dir / "week2_3_outputs"
output_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 1. Load Week 1 combined CSVs
# ---------------------------------------------------------
sold_files = sorted(combined_dir.glob("CRMLSSold_Combined_Residential_*.csv"))
listing_files = sorted(combined_dir.glob("CRMLSListing_Combined_Residential_*.csv"))

if not sold_files:
    raise FileNotFoundError("No Week 1 sold combined file found in raw/combined_outputs")
if not listing_files:
    raise FileNotFoundError("No Week 1 listing combined file found in raw/combined_outputs")

sold_file = sold_files[-1]
listing_file = listing_files[-1]

sold = pd.read_csv(sold_file, low_memory=False)
listings = pd.read_csv(listing_file, low_memory=False)

print("=" * 70)
print("FILES LOADED")
print("=" * 70)
print(f"Sold file loaded: {sold_file}")
print(f"Listing file loaded: {listing_file}")
print(f"Sold shape: {sold.shape}")
print(f"Listings shape: {listings.shape}")
print()

# ---------------------------------------------------------
# 2. Document unique property types + filtering logic
# ---------------------------------------------------------
sold_property_types = sorted(sold["PropertyType"].dropna().astype(str).str.strip().unique().tolist()) if "PropertyType" in sold.columns else []
listing_property_types = sorted(listings["PropertyType"].dropna().astype(str).str.strip().unique().tolist()) if "PropertyType" in listings.columns else []

print("=" * 70)
print("PROPERTY TYPE CHECK")
print("=" * 70)
print("Sold PropertyType values:", sold_property_types)
print("Listing PropertyType values:", listing_property_types)
print()

# Even though Week 1 already filtered to Residential, we confirm again here
sold_filtered = sold[sold["PropertyType"].astype(str).str.strip().str.lower() == "residential"].copy()
listings_filtered = listings[listings["PropertyType"].astype(str).str.strip().str.lower() == "residential"].copy()

print("Sold rows before Residential confirmation filter:", len(sold))
print("Sold rows after Residential confirmation filter: ", len(sold_filtered))
print("Listings rows before Residential confirmation filter:", len(listings))
print("Listings rows after Residential confirmation filter: ", len(listings_filtered))
print()

sold = sold_filtered
listings = listings_filtered

# ---------------------------------------------------------
# 3. Null-count summary helper
# ---------------------------------------------------------
def build_null_summary(df, dataset_name):
    summary = pd.DataFrame({
        "column": df.columns,
        "null_count": df.isnull().sum().values,
        "missing_pct": (df.isnull().mean().values * 100)
    })
    summary["flag_above_90_pct_missing"] = summary["missing_pct"] > 90
    summary = summary.sort_values(["missing_pct", "null_count"], ascending=[False, False]).reset_index(drop=True)

    out_file = output_dir / f"{dataset_name}_null_summary.csv"
    summary.to_csv(out_file, index=False)

    print("=" * 70)
    print(f"{dataset_name.upper()} NULL SUMMARY")
    print("=" * 70)
    print(summary.head(20).to_string(index=False))
    print()
    print(f"Saved null summary to: {out_file}")
    print()

    return summary

sold_null_summary = build_null_summary(sold, "sold")
listing_null_summary = build_null_summary(listings, "listings")

# ---------------------------------------------------------
# 4. Columns above 90% missing
# ---------------------------------------------------------
sold_above_90 = sold_null_summary[sold_null_summary["flag_above_90_pct_missing"]]
listing_above_90 = listing_null_summary[listing_null_summary["flag_above_90_pct_missing"]]

print("=" * 70)
print("COLUMNS ABOVE 90% MISSING")
print("=" * 70)
print("Sold columns >90% missing:")
print(sold_above_90[["column", "null_count", "missing_pct"]].to_string(index=False))
print()
print("Listing columns >90% missing:")
print(listing_above_90[["column", "null_count", "missing_pct"]].to_string(index=False))
print()

# ---------------------------------------------------------
# 5. Numeric summary helper
# ---------------------------------------------------------
def numeric_summary(df, cols, dataset_name):
    available_cols = [c for c in cols if c in df.columns]
    if not available_cols:
        return pd.DataFrame()

    summary_rows = []

    for col in available_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        row = {
            "dataset": dataset_name,
            "column": col,
            "non_null_count": series.notna().sum(),
            "min": series.min(),
            "p01": series.quantile(0.01),
            "p05": series.quantile(0.05),
            "p25": series.quantile(0.25),
            "median": series.median(),
            "mean": series.mean(),
            "p75": series.quantile(0.75),
            "p95": series.quantile(0.95),
            "p99": series.quantile(0.99),
            "max": series.max()
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    out_file = output_dir / f"{dataset_name}_numeric_summary.csv"
    summary_df.to_csv(out_file, index=False)

    print("=" * 70)
    print(f"{dataset_name.upper()} NUMERIC SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print()
    print(f"Saved numeric summary to: {out_file}")
    print()

    return summary_df

required_numeric_cols = ["ClosePrice", "LivingArea", "DaysOnMarket"]

sold_numeric_summary = numeric_summary(sold, required_numeric_cols, "sold")
listing_numeric_summary = numeric_summary(listings, required_numeric_cols, "listings")

# ---------------------------------------------------------
# 6. Fetch mortgage rate data from FRED
# ---------------------------------------------------------
fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(fred_url)

mortgage.columns = ["date", "rate30yrfixed"]
mortgage["date"] = pd.to_datetime(mortgage["date"], errors="coerce")
mortgage["rate30yrfixed"] = pd.to_numeric(mortgage["rate30yrfixed"], errors="coerce")

mortgage = mortgage.dropna(subset=["date", "rate30yrfixed"]).copy()
mortgage["yearmonth"] = mortgage["date"].dt.to_period("M")

mortgage_monthly = (
    mortgage.groupby("yearmonth", as_index=False)["rate30yrfixed"]
    .mean()
)

mortgage_monthly["yearmonth"] = mortgage_monthly["yearmonth"].astype(str)
mortgage_monthly_out = output_dir / "mortgage_monthly_avg.csv"
mortgage_monthly.to_csv(mortgage_monthly_out, index=False)

print("=" * 70)
print("MORTGAGE DATA")
print("=" * 70)
print(mortgage_monthly.head(12).to_string(index=False))
print()
print(f"Saved monthly mortgage rate file to: {mortgage_monthly_out}")
print()

# ---------------------------------------------------------
# 7. Create yearmonth keys on sold and listings
# ---------------------------------------------------------
def add_yearmonth_key(df, date_col, dataset_name):
    if date_col not in df.columns:
        raise KeyError(f"{date_col} not found in {dataset_name} dataset")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["yearmonth"] = df[date_col].dt.to_period("M").astype(str)

    print(f"{dataset_name}: created yearmonth from {date_col}")
    print(f"{dataset_name}: null {date_col} values after parsing = {df[date_col].isnull().sum():,}")
    print(f"{dataset_name}: null yearmonth values = {df['yearmonth'].isnull().sum():,}")
    print()

    return df

sold = add_yearmonth_key(sold, "CloseDate", "sold")
listings = add_yearmonth_key(listings, "ListingContractDate", "listings")

# ---------------------------------------------------------
# 8. Merge monthly mortgage rates
# ---------------------------------------------------------
sold_enriched = sold.merge(mortgage_monthly, on="yearmonth", how="left")
listings_enriched = listings.merge(mortgage_monthly, on="yearmonth", how="left")

# ---------------------------------------------------------
# 9. Validate mortgage merge completeness
# ---------------------------------------------------------
sold_null_rates = sold_enriched["rate30yrfixed"].isnull().sum()
listing_null_rates = listings_enriched["rate30yrfixed"].isnull().sum()

print("=" * 70)
print("MORTGAGE MERGE VALIDATION")
print("=" * 70)
print(f"Sold rows with null mortgage rate: {sold_null_rates:,}")
print(f"Listing rows with null mortgage rate: {listing_null_rates:,}")
print()

preview_cols_sold = [c for c in ["CloseDate", "yearmonth", "ClosePrice", "rate30yrfixed"] if c in sold_enriched.columns]
preview_cols_listings = [c for c in ["ListingContractDate", "yearmonth", "ListPrice", "rate30yrfixed"] if c in listings_enriched.columns]

print("Sold preview:")
print(sold_enriched[preview_cols_sold].head().to_string(index=False))
print()
print("Listings preview:")
print(listings_enriched[preview_cols_listings].head().to_string(index=False))
print()

# ---------------------------------------------------------
# 10. Save enriched datasets
# ---------------------------------------------------------
sold_enriched_file = output_dir / "sold_residential_enriched_week2_3.csv"
listings_enriched_file = output_dir / "listings_residential_enriched_week2_3.csv"

sold_enriched.to_csv(sold_enriched_file, index=False)
listings_enriched.to_csv(listings_enriched_file, index=False)

# ---------------------------------------------------------
# 11. Save summary report text file
# ---------------------------------------------------------
report_lines = [
    "WEEKS 2-3 DATASET STRUCTURING AND VALIDATION REPORT",
    "=" * 70,
    "",
    f"Sold file loaded: {sold_file}",
    f"Listing file loaded: {listing_file}",
    "",
    "FILTERING LOGIC",
    "Confirmed datasets were filtered to PropertyType == 'Residential'.",
    f"Sold rows after Residential confirmation filter: {len(sold):,}",
    f"Listing rows after Residential confirmation filter: {len(listings):,}",
    "",
    "UNIQUE PROPERTY TYPES - SOLD",
]
report_lines.extend([f"- {x}" for x in sold_property_types] if sold_property_types else ["- None found"])

report_lines.append("")
report_lines.append("UNIQUE PROPERTY TYPES - LISTINGS")
report_lines.extend([f"- {x}" for x in listing_property_types] if listing_property_types else ["- None found"])

report_lines.append("")
report_lines.append("SOLD COLUMNS ABOVE 90% MISSING")
if sold_above_90.empty:
    report_lines.append("None")
else:
    for _, row in sold_above_90.iterrows():
        report_lines.append(f"- {row['column']}: {row['missing_pct']:.2f}% missing")

report_lines.append("")
report_lines.append("LISTING COLUMNS ABOVE 90% MISSING")
if listing_above_90.empty:
    report_lines.append("None")
else:
    for _, row in listing_above_90.iterrows():
        report_lines.append(f"- {row['column']}: {row['missing_pct']:.2f}% missing")

report_lines.append("")
report_lines.append(f"Sold rows with null mortgage rate after merge: {sold_null_rates:,}")
report_lines.append(f"Listing rows with null mortgage rate after merge: {listing_null_rates:,}")
report_lines.append("")
report_lines.append(f"Saved sold enriched dataset: {sold_enriched_file}")
report_lines.append(f"Saved listing enriched dataset: {listings_enriched_file}")

report_file = output_dir / "week2_3_report.txt"
with open(report_file, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("=" * 70)
print("FILES SAVED")
print("=" * 70)
print(f"Sold enriched dataset: {sold_enriched_file}")
print(f"Listings enriched dataset: {listings_enriched_file}")
print(f"Report file: {report_file}")
print("Done.")