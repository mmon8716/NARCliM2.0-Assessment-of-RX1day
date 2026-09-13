
import os
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 1. FILES
# ============================================================

historical_file = (
    "/Users/maryammontazerolghaem/"
    "Documents/Climate_N2 /data_out2/"
    "RX1day_NARCliM2-0-WRF412R3_ACCESS-ESM1-5_historical_1985-2014.nc"
)

ssp126_2035_2064_file = (
    "/Users/maryammontazerolghaem/"
    "Documents/Climate_N2 /data_out_1/"
    "RX1day_NARCliM2-0-WRF412R3_ACCESS-ESM1-5_ssp126_2035-2064.nc"
)


# ============================================================
# 2. OPEN FILES
# ============================================================

hist_ds = xr.open_dataset(
    historical_file,
    engine="netcdf4"
)

future_ds = xr.open_dataset(
    ssp126_2035_2064_file,
    engine="netcdf4"
)

print("Historical:")
print(hist_ds)

print("\nFuture:")
print(future_ds)


# ============================================================
# 3. SELECT RX1DAY
# ============================================================

hist = hist_ds["pr"]
future = future_ds["pr"]

print("\nHistorical RX1day:")
print(hist)

print("\nFuture RX1day:")
print(future)


# ============================================================
# 4. ANNUAL MAXIMUM
# ============================================================

# If RX1day already contains one value per year,
# this simply preserves the annual values.

if "time" in hist.dims:

    hist_yearly = hist.groupby("time.year").max(
        dim="time",
        skipna=True
    )

else:

    hist_yearly = hist


if "time" in future.dims:

    future_yearly = future.groupby("time.year").max(
        dim="time",
        skipna=True
    )

else:

    future_yearly = future


print("\nHistorical yearly:")
print(hist_yearly)

print("\nFuture yearly:")
print(future_yearly)


# ============================================================
# 5. SPATIAL DIMENSIONS
# ============================================================

spatial_dims = [
    d for d in hist_yearly.dims
    if d not in ["year", "time"]
]

print("\nSpatial dimensions:")
print(spatial_dims)


# ============================================================
# 6. SPATIAL STATISTICS FOR EACH YEAR
# ============================================================

hist_mean_yearly = hist_yearly.mean(
    dim=spatial_dims,
    skipna=True
)

future_mean_yearly = future_yearly.mean(
    dim=spatial_dims,
    skipna=True
)


# Spatial 10th and 90th percentiles

hist_p10_yearly = hist_yearly.quantile(
    0.10,
    dim=spatial_dims,
    skipna=True
)

hist_p90_yearly = hist_yearly.quantile(
    0.90,
    dim=spatial_dims,
    skipna=True
)

future_p10_yearly = future_yearly.quantile(
    0.10,
    dim=spatial_dims,
    skipna=True
)

future_p90_yearly = future_yearly.quantile(
    0.90,
    dim=spatial_dims,
    skipna=True
)


# ============================================================
# 7. PERIOD MEANS
# ============================================================

# Baseline:
# Average annual maximum RX1day over 1985–2014

baseline_mean = hist_mean_yearly.mean(
    dim="year",
    skipna=True
).item()

baseline_p10 = hist_p10_yearly.mean(
    dim="year",
    skipna=True
).item()

baseline_p90 = hist_p90_yearly.mean(
    dim="year",
    skipna=True
)


# Future:
# Average annual maximum RX1day over 2035–2064

future_mean = future_mean_yearly.mean(
    dim="year",
    skipna=True
).item()

future_p10 = future_p10_yearly.mean(
    dim="year",
    skipna=True
).item()

future_p90 = future_p90_yearly.mean(
    dim="year",
    skipna=True
).item()


# ============================================================
# 8. PERCENTAGE CHANGE
# ============================================================

percentage_change = (
    (future_mean - baseline_mean)
    / baseline_mean
) * 100


# ============================================================
# 9. UNCERTAINTY / SPREAD
# ============================================================

baseline_uncertainty = (
    f"{baseline_p10:.1f}–{baseline_p90:.1f}"
)

future_uncertainty = (
    f"{future_p10:.1f}–{future_p90:.1f}"
)


# ============================================================
# 10. CREATE TABLE
# ============================================================

table = pd.DataFrame({

    "Climate scenario": [
        "Baseline",
        "SSP1-2.6 (low emissions)"
    ],

    "Period": [
        "Historical/reference 1985–2014",
        "2035–2064"
    ],

    "RX1day (mm)": [
        round(baseline_mean, 1),
        round(future_mean, 1)
    ],

    "Change from baseline (%)": [
        "—",
        round(percentage_change, 1)
    ],

    "Model ensemble / uncertainty": [
        f"{baseline_uncertainty} mm "
        "(10th–90th spatial percentile)",

        f"{future_uncertainty} mm "
        "(10th–90th spatial percentile)"
    ]
})


# ============================================================
# 11. DISPLAY TABLE
# ============================================================

print("\n")
print("=" * 100)
print("RX1DAY CLIMATE SUMMARY")
print("=" * 100)

print(
    table.to_string(index=False)
)


# ============================================================
# 12. SAVE CSV
# ============================================================

output_folder = (
    "/Users/maryammontazerolghaem/"
    "Documents/Climate_N2 /data_out2"
)

os.makedirs(
    output_folder,
    exist_ok=True
)


csv_file = os.path.join(
    output_folder,
    "RX1day_SSP126_summary_table.csv"
)

table.to_csv(
    csv_file,
    index=False
)


# ============================================================
# 13. SAVE EXCEL
# ============================================================

excel_file = os.path.join(
    output_folder,
    "RX1day_SSP126_summary_table.xlsx"
)

table.to_excel(
    excel_file,
    index=False
)


# ============================================================
# 14. PRINT RESULTS
# ============================================================

print("\nBaseline RX1day:")
print(f"{baseline_mean:.2f} mm")

print("\nSSP1-2.6 2035–2064 RX1day:")
print(f"{future_mean:.2f} mm")

print("\nChange:")
print(f"{percentage_change:.2f} %")

print("\nBaseline spatial range:")
print(
    f"{baseline_p10:.2f} – {baseline_p90:.2f} mm"
)

print("\nFuture spatial range:")
print(
    f"{future_p10:.2f} – {future_p90:.2f} mm"
)

print("\nCSV saved:")
print(csv_file)

hist_ds.close()
future_ds.close()