import pandas as pd
import numpy as np
from pathlib import Path
import re

raw_dir = Path("raw")
out_dir = raw_dir / "final_outputs"
out_dir.mkdir(exist_ok=True)

today = pd.Timestamp.today()
latest_month = (today.replace(day=1) - pd.DateOffset(months=1)).strftime("%Y%m")

sold_pattern = re.compile(r"CRMLSSold(\d{6})\.csv$", re.IGNORECASE)
listing_pattern = re.compile(r"CRMLSListing(\d{6})\.csv$", re.IGNORECASE)

def get_files(folder, pattern, start_yyyymm="202401", end_yyyymm=latest_month):
    files = []
    for f in folder.glob("*.csv"):
        m = pattern.match(f.name)
        if m:
            yyyymm = m.group(1)
            if start_yyyymm <= yyyymm <= end_yyyymm:
                files.append((yyyymm, f))
    files.sort(key=lambda x: x[0])
    return [f for _, f in files]

def load_files(files):
    parts = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        parts.append(df)
        print(f"{f.name}: {len(df):,}")
    return pd.concat(parts, ignore_index=True)

def to_dt(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def to_num(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def get_mortgage():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
    m = pd.read_csv(url)
    m.columns = ["date", "rate30yrfixed"]
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m["rate30yrfixed"] = pd.to_numeric(m["rate30yrfixed"], errors="coerce")
    m = m.dropna(subset=["date", "rate30yrfixed"]).copy()
    m["yearmonth"] = m["date"].dt.to_period("M")
    m = m.groupby("yearmonth", as_index=False)["rate30yrfixed"].mean()
    m["yearmonth"] = m["yearmonth"].astype(str)
    return m

def add_geo_flags(df):
    lat = pd.to_numeric(df["Latitude"], errors="coerce") if "Latitude" in df.columns else pd.Series(np.nan, index=df.index)
    lon = pd.to_numeric(df["Longitude"], errors="coerce") if "Longitude" in df.columns else pd.Series(np.nan, index=df.index)

    df["missing_latitude_flag"] = lat.isna()
    df["missing_longitude_flag"] = lon.isna()
    df["latitude_zero_flag"] = lat.eq(0)
    df["longitude_zero_flag"] = lon.eq(0)
    df["longitude_positive_flag"] = lon.gt(0)
    df["implausible_latitude_flag"] = lat.lt(-90) | lat.gt(90)
    df["implausible_longitude_flag"] = lon.lt(-180) | lon.gt(180)
    df["out_of_ca_lat_flag"] = lat.notna() & ((lat < 32) | (lat > 43))
    df["out_of_ca_lon_flag"] = lon.notna() & ((lon < -125) | (lon > -114))
    return df

def add_iqr_flag(df, col, flag_name):
    if col not in df.columns:
        df[flag_name] = False
        return df

    x = pd.to_numeric(df[col], errors="coerce")
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df[flag_name] = x.notna() & ((x < lower) | (x > upper))
    return df

def clean_common(df):
    date_cols = ["CloseDate", "PurchaseContractDate", "ListingContractDate", "ContractStatusChangeDate"]
    num_cols = [
        "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres",
        "BedroomsTotal", "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt",
        "Latitude", "Longitude"
    ]

    df = to_dt(df, date_cols)
    df = to_num(df, num_cols)
    df = add_geo_flags(df)

    if "BedroomsTotal" in df.columns:
        df["bedrooms_negative_flag"] = df["BedroomsTotal"].lt(0)
    else:
        df["bedrooms_negative_flag"] = False

    if "BathroomsTotalInteger" in df.columns:
        df["bathrooms_negative_flag"] = df["BathroomsTotalInteger"].lt(0)
    else:
        df["bathrooms_negative_flag"] = False

    if "LivingArea" in df.columns:
        df["livingarea_le_zero_flag"] = df["LivingArea"].le(0)
    else:
        df["livingarea_le_zero_flag"] = False

    if "DaysOnMarket" in df.columns:
        df["daysonmarket_lt_zero_flag"] = df["DaysOnMarket"].lt(0)
    else:
        df["daysonmarket_lt_zero_flag"] = False

    if {"ListingContractDate", "CloseDate"}.issubset(df.columns):
        df["listingaftercloseflag"] = (
            df["ListingContractDate"].notna() &
            df["CloseDate"].notna() &
            (df["ListingContractDate"] > df["CloseDate"])
        )
    else:
        df["listingaftercloseflag"] = False

    if {"PurchaseContractDate", "CloseDate"}.issubset(df.columns):
        df["purchaseaftercloseflag"] = (
            df["PurchaseContractDate"].notna() &
            df["CloseDate"].notna() &
            (df["PurchaseContractDate"] > df["CloseDate"])
        )
    else:
        df["purchaseaftercloseflag"] = False

    if {"ListingContractDate", "PurchaseContractDate"}.issubset(df.columns):
        df["negativetimelineflag"] = (
            df["ListingContractDate"].notna() &
            df["PurchaseContractDate"].notna() &
            (df["PurchaseContractDate"] < df["ListingContractDate"])
        )
    else:
        df["negativetimelineflag"] = False

    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed:")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df

def build_sold(mortgage):
    files = get_files(raw_dir, sold_pattern)
    if not files:
        raise FileNotFoundError("No sold files found in raw")

    df = load_files(files)

    before_filter = len(df)
    df = df[df["PropertyType"].astype(str).str.strip().str.lower() == "residential"].copy()
    after_filter = len(df)

    print("sold before residential:", f"{before_filter:,}")
    print("sold after residential:", f"{after_filter:,}")

    df = clean_common(df)

    df["yearmonth"] = df["CloseDate"].dt.to_period("M").astype(str)
    df = df.merge(mortgage, on="yearmonth", how="left")

    if "ClosePrice" in df.columns:
        df["closeprice_le_zero_flag"] = df["ClosePrice"].le(0)
    else:
        df["closeprice_le_zero_flag"] = False

    df["price_ratio"] = np.where(
        df["OriginalListPrice"].notna() & (df["OriginalListPrice"] > 0),
        df["ClosePrice"] / df["OriginalListPrice"],
        np.nan
    )

    df["close_to_original_list_ratio"] = np.where(
        df["OriginalListPrice"].notna() & (df["OriginalListPrice"] > 0),
        df["ClosePrice"] / df["OriginalListPrice"],
        np.nan
    )

    df["price_per_sqft"] = np.where(
        df["LivingArea"].notna() & (df["LivingArea"] > 0),
        df["ClosePrice"] / df["LivingArea"],
        np.nan
    )

    df["YrMo"] = df["CloseDate"].dt.to_period("M").astype(str)

    df["listing_to_contract_days"] = np.where(
        df["ListingContractDate"].notna() & df["PurchaseContractDate"].notna(),
        (df["PurchaseContractDate"] - df["ListingContractDate"]).dt.days,
        np.nan
    )

    df["contract_to_close_days"] = np.where(
        df["PurchaseContractDate"].notna() & df["CloseDate"].notna(),
        (df["CloseDate"] - df["PurchaseContractDate"]).dt.days,
        np.nan
    )

    df = add_iqr_flag(df, "ClosePrice", "closeprice_outlier_flag")
    df = add_iqr_flag(df, "LivingArea", "livingarea_outlier_flag")
    df = add_iqr_flag(df, "DaysOnMarket", "daysonmarket_outlier_flag")

    df["analysis_filtered_flag"] = ~(
        df["closeprice_le_zero_flag"] |
        df["livingarea_le_zero_flag"] |
        df["daysonmarket_lt_zero_flag"] |
        df["bedrooms_negative_flag"] |
        df["bathrooms_negative_flag"] |
        df["listingaftercloseflag"] |
        df["purchaseaftercloseflag"] |
        df["negativetimelineflag"] |
        df["longitude_positive_flag"] |
        df["implausible_latitude_flag"] |
        df["implausible_longitude_flag"] |
        df["closeprice_outlier_flag"] |
        df["livingarea_outlier_flag"] |
        df["daysonmarket_outlier_flag"]
    )

    return df

def build_listings(mortgage):
    files = get_files(raw_dir, listing_pattern)
    if not files:
        raise FileNotFoundError("No listing files found in raw")

    df = load_files(files)

    before_filter = len(df)
    df = df[df["PropertyType"].astype(str).str.strip().str.lower() == "residential"].copy()
    after_filter = len(df)

    print("listings before residential:", f"{before_filter:,}")
    print("listings after residential:", f"{after_filter:,}")

    df = clean_common(df)

    df["yearmonth"] = df["ListingContractDate"].dt.to_period("M").astype(str)
    df = df.merge(mortgage, on="yearmonth", how="left")

    if "ListPrice" in df.columns:
        df["listprice_le_zero_flag"] = df["ListPrice"].le(0)
    else:
        df["listprice_le_zero_flag"] = False

    if "ClosePrice" in df.columns:
        df["closeprice_le_zero_flag"] = df["ClosePrice"].le(0)
    else:
        df["closeprice_le_zero_flag"] = False

    df["price_ratio"] = np.where(
        df["OriginalListPrice"].notna() & (df["OriginalListPrice"] > 0) & df["ClosePrice"].notna(),
        df["ClosePrice"] / df["OriginalListPrice"],
        np.nan
    )

    df["close_to_original_list_ratio"] = np.where(
        df["OriginalListPrice"].notna() & (df["OriginalListPrice"] > 0) & df["ClosePrice"].notna(),
        df["ClosePrice"] / df["OriginalListPrice"],
        np.nan
    )

    df["price_per_sqft"] = np.where(
        df["LivingArea"].notna() & (df["LivingArea"] > 0) & df["ListPrice"].notna(),
        df["ListPrice"] / df["LivingArea"],
        np.nan
    )

    df["YrMo"] = df["ListingContractDate"].dt.to_period("M").astype(str)

    df["listing_to_contract_days"] = np.where(
        df["ListingContractDate"].notna() & df["PurchaseContractDate"].notna(),
        (df["PurchaseContractDate"] - df["ListingContractDate"]).dt.days,
        np.nan
    )

    df["contract_to_close_days"] = np.where(
        df["PurchaseContractDate"].notna() & df["CloseDate"].notna(),
        (df["CloseDate"] - df["PurchaseContractDate"]).dt.days,
        np.nan
    )

    df = add_iqr_flag(df, "ListPrice", "listprice_outlier_flag")
    df = add_iqr_flag(df, "LivingArea", "livingarea_outlier_flag")
    df = add_iqr_flag(df, "DaysOnMarket", "daysonmarket_outlier_flag")

    df["analysis_filtered_flag"] = ~(
        df["listprice_le_zero_flag"] |
        df["livingarea_le_zero_flag"] |
        df["daysonmarket_lt_zero_flag"] |
        df["bedrooms_negative_flag"] |
        df["bathrooms_negative_flag"] |
        df["listingaftercloseflag"] |
        df["purchaseaftercloseflag"] |
        df["negativetimelineflag"] |
        df["longitude_positive_flag"] |
        df["implausible_latitude_flag"] |
        df["implausible_longitude_flag"] |
        df["listprice_outlier_flag"] |
        df["livingarea_outlier_flag"] |
        df["daysonmarket_outlier_flag"]
    )

    return df

mortgage = get_mortgage()

sold_flagged = build_sold(mortgage)
listings_flagged = build_listings(mortgage)

sold_clean = sold_flagged[sold_flagged["analysis_filtered_flag"]].copy()
listings_clean = listings_flagged[listings_flagged["analysis_filtered_flag"]].copy()

sold_flagged.to_csv(out_dir / "sold_flagged.csv", index=False)
sold_clean.to_csv(out_dir / "sold_clean.csv", index=False)

listings_flagged.to_csv(out_dir / "listings_flagged.csv", index=False)
listings_clean.to_csv(out_dir / "listings_clean.csv", index=False)

print("saved:", out_dir / "sold_flagged.csv")
print("saved:", out_dir / "sold_clean.csv")
print("saved:", out_dir / "listings_flagged.csv")
print("saved:", out_dir / "listings_clean.csv")