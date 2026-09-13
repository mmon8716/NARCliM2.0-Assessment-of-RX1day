


import os
import gc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# 1. FILE PATHS
# ============================================================

historical_file = (
    "/Users/maryammontazerolghaem/"
    "Documents/Climate_N2 /data_out2/"
    "RX1day_NARCliM2-0-WRF412R3_ACCESS-ESM1-5_historical_1985-2014.nc"
)
future_file = (
    "/Users/maryammontazerolghaem/"
    "Documents/Climate_N2 /data_out_1/"
    "RX1day_NARCliM2-0-WRF412R3_ACCESS-ESM1-5_ssp126_2035-2064.nc"
)
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# 1. FILES
# ============================================================

historical_file = (
    "/Users/maryammontazerolghaem/"
    "Documents/Climate_N2 /data_out2/"
    "RX1day_NARCliM2-0-WRF412R3_ACCESS-ESM1-5_historical_1985-2014.nc"
)

future_file = (
    "/Users/maryammontazerolghaem/"
    "Documents/Climate_N2 /data_out_1/"
    "RX1day_NARCliM2-0-WRF412R3_ACCESS-ESM1-5_ssp126_2035-2064.nc"
)


# ============================================================
# 2. OPEN
# ============================================================

hist_ds = xr.open_dataset(
    historical_file,
    engine="netcdf4"
)

future_ds = xr.open_dataset(
    future_file,
    engine="netcdf4"
)

print(hist_ds)
print(future_ds)


# ============================================================
# 3. RX1day
# ============================================================

hist = hist_ds["pr"]
future = future_ds["pr"]

print("\nHistorical:")
print(hist)

print("\nFuture:")
print(future)


# ============================================================
# 4. CHECK NANs
# ============================================================

hist_nan = hist.isnull().sum().item()
future_nan = future.isnull().sum().item()

hist_total = hist.size
future_total = future.size

print("\n================ NAN CHECK ================")

print(
    f"Historical NaN: {hist_nan:,} / {hist_total:,} "
    f"({100 * hist_nan / hist_total:.2f}%)"
)

print(
    f"Future NaN:     {future_nan:,} / {future_total:,} "
    f"({100 * future_nan / future_total:.2f}%)"
)


# ============================================================
# 5. MAXIMUM OVER YEARS
# ============================================================

hist_max = hist.max(
    dim="year",
    skipna=True
)

future_max = future.max(
    dim="year",
    skipna=True
)


# ============================================================
# 6. CHECK SPATIAL NaNs AFTER MAX
# ============================================================

print("\n=========== SPATIAL NAN CHECK ===========")

hist_max_nan = hist_max.isnull().sum().item()
future_max_nan = future_max.isnull().sum().item()

print("Historical spatial NaNs:", hist_max_nan)
print("Future spatial NaNs:", future_max_nan)


# ============================================================
# 7. COMMON VALID MASK
# ============================================================

valid_mask = (
    hist_max.notnull()
    & future_max.notnull()
)

print(
    "\nCommon valid grid cells:",
    valid_mask.sum().item()
)

print(
    "Total grid cells:",
    valid_mask.size
)

print(
    "Percentage valid:",
    100 * valid_mask.sum().item() / valid_mask.size
)


# ============================================================
# 8. APPLY COMMON MASK
# ============================================================

hist_max = hist_max.where(valid_mask)
future_max = future_max.where(valid_mask)


# ============================================================
# 9. RELATIVE CHANGE
# ============================================================

relative_change = (
    (future_max - hist_max)
    / hist_max
) * 100


# Remove unrealistic percentage changes caused by
# extremely small historical RX1day values

relative_change = relative_change.where(
    hist_max > 1e-6
)


# ============================================================
# 10. GET LAT/LON
# ============================================================

if "lat" in hist_max.coords:
    lat = hist_max["lat"]
else:
    lat = hist_max["latitude"]

if "lon" in hist_max.coords:
    lon = hist_max["lon"]
else:
    lon = hist_max["longitude"]


# ============================================================
# 11. COMMON COLOUR SCALE FOR RX1DAY
# ============================================================

all_rx1day = np.concatenate([
    hist_max.values[np.isfinite(hist_max.values)],
    future_max.values[np.isfinite(future_max.values)]
])

vmin = np.nanpercentile(all_rx1day, 2)
vmax = np.nanpercentile(all_rx1day, 98)


# ============================================================
# 12. CHANGE SCALE
# ============================================================

change_values = relative_change.values[
    np.isfinite(relative_change.values)
]

change_limit = np.nanpercentile(
    np.abs(change_values),
    98
)


# ============================================================
# 13. PLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(20, 6)
)


# ------------------------------------------------------------
# Historical
# ------------------------------------------------------------

im1 = axes[0].pcolormesh(
    lon,
    lat,
    hist_max,
    shading="auto",
    cmap="Blues",
    vmin=vmin,
    vmax=vmax
)

axes[0].set_title(
    "Historical Maximum RX1day\n1985–2014"
)

axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")

plt.colorbar(
    im1,
    ax=axes[0],
    label="RX1day"
)


# ------------------------------------------------------------
# Future
# ------------------------------------------------------------

im2 = axes[1].pcolormesh(
    lon,
    lat,
    future_max,
    shading="auto",
    cmap="Blues",
    vmin=vmin,
    vmax=vmax
)

axes[1].set_title(
    "SSP126 Maximum RX1day\n2035–2064"
)

axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")

plt.colorbar(
    im2,
    ax=axes[1],
    label="RX1day"
)


# ------------------------------------------------------------
# Relative change
# ------------------------------------------------------------

im3 = axes[2].pcolormesh(
    lon,
    lat,
    relative_change,
    shading="auto",
    cmap="RdBu_r",
    vmin=-change_limit,
    vmax=change_limit
)

axes[2].set_title(
    "Relative Change\nSSP126 2035–2064 vs 1985–2014"
)

axes[2].set_xlabel("Longitude")
axes[2].set_ylabel("Latitude")

plt.colorbar(
    im3,
    ax=axes[2],
    label="Change (%)"
)


# ============================================================
# 14. SAVE
# ============================================================

plt.suptitle(
    "NARCliM2.0 RX1day Extreme Rainfall",
    fontsize=16
)

plt.tight_layout()

output_file = (
    "/Users/maryammontazerolghaem/"
    "Documents/Climate_N2 /data_out2/"
    "RX1day_SSP126_2035-2064_relative_change.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print(output_file)


# ============================================================
# 15. CLOSE
# ============================================================

hist_ds.close() 
future_ds.close()


# ============================================================
# 7. SPATIAL DIMENSIONS
# ============================================================

spatial_dims = [
    d for d in hist.dims
    if d not in ["year", "time"]
]

print("\nSpatial dimensions:")
print(spatial_dims)


# ============================================================
# 8. SPATIAL MEAN
# ============================================================

hist_mean = hist.mean(
    dim=spatial_dims,
    skipna=True
)

future_mean = future.mean(
    dim=spatial_dims,
    skipna=True
)


# ============================================================
# 9. SPATIAL CONFIDENCE / VARIABILITY BAND
# ============================================================

# 10th and 90th percentiles across spatial grid cells

hist_lower = hist.quantile(
    0.10,
    dim=spatial_dims,
    skipna=True
)

hist_upper = hist.quantile(
    0.90,
    dim=spatial_dims,
    skipna=True
)


future_lower = future.quantile(
    0.10,
    dim=spatial_dims,
    skipna=True
)

future_upper = future.quantile(
    0.90,
    dim=spatial_dims,
    skipna=True
)
#8. Plot
# ============================================================
# 10. PLOT TIME SERIES
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)


# ------------------------------------------------------------
# Historical confidence band
# ------------------------------------------------------------

ax.fill_between(
    hist["year"].values,
    hist_lower.values,
    hist_upper.values,
    alpha=0.20,
    color="grey",
    label="Historical spatial 10–90th percentile"
)


# ------------------------------------------------------------
# Historical mean
# ------------------------------------------------------------

ax.plot(
    hist["year"].values,
    hist_mean.values,
    color="grey",
    linewidth=2.5,
    label="Historical spatial mean"
)


# ------------------------------------------------------------
# Future confidence band
# ------------------------------------------------------------

ax.fill_between(
    future["year"].values,
    future_lower.values,
    future_upper.values,
    alpha=0.20,
    color="blue",
    label="SSP126 spatial 10–90th percentile"
)


# ------------------------------------------------------------
# Future mean
# ------------------------------------------------------------

ax.plot(
    future["year"].values,
    future_mean.values,
    color="blue",
    linewidth=2.5,
    label="SSP126 spatial mean"
)


# ============================================================
# 11. FORMATTING
# ============================================================

ax.set_xlabel(
    "Year",
    fontsize=12
)

ax.set_ylabel(
    "RX1day (mm)",
    fontsize=12
)

ax.set_title(
    "Annual Maximum RX1day — NSW",
    fontsize=15,
    fontweight="bold"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend(
    frameon=False,
    loc="best"
)