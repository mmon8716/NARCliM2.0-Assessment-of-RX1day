
# ============================================================
# calculate_narclim_rx1day_nsw.py
#
# Calculate annual RX1day extreme rainfall for NSW using
# NARCliM2.0 hourly precipitation data.
#
# Workflow:
#   1. Open one NARCliM2.0 file at a time
#   2. Identify the NSW grid cells using 2-D lat/lon
#   3. Calculate daily precipitation
#   4. Calculate annual RX1day
#   5. Add the year as a coordinate
#   6. Save processing status/errors to CSV
#   7. Save annual RX1day data to NetCDF
#
# ============================================================

import os
import gc
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 1. CONFIGURATION
# ============================================================

OPENDAP_BASE = (
    "https://thredds.nci.org.au/thredds/dodsC/"
    "zz63/NARCliM2-0/output-CMIP6"
)

DOMAIN = "NARCliM2-0-SEAus-04"
PUBLISHER = "NSW-Government"

VARIABLE = "pr"
FREQUENCY = "1hr"
VERSION = "v1-r1"

RCMS = [
    "NARCliM2-0-WRF412R3",
   "NARCliM2-0-WRF412R5",
]

GCM_REALIZATION = {
    "ACCESS-ESM1-5": "r6i1p1f1",
    "EC-Earth3-Veg": "r1i1p1f1",
    "MPI-ESM1-2-HR": "r1i1p1f1",
    "NorESM2-MM": "r1i1p1f1",
    "UKESM1-0-LL": "r1i1p1f2",
}

MODELS = list(GCM_REALIZATION.keys())

SCENARIOS = [
    "historical",
    "ssp126",
    "ssp245",
     "ssp370",
]
# ============================================================
# 2. NSW BOUNDING BOX
# ============================================================

NSW_BBOX = {
    "lat_min": -37.5,
    "lat_max": -28.0,
    "lon_min": 140.9,
    "lon_max": 153.7,
}
# ============================================================
# 3. PERIODS
# ============================================================

PERIODS = {
    "historical": [
        (1985,2014),       # Change to (1985, 2014) for full baseline
    ],

    "ssp126": [
        (2035, 2064),
        (2070, 2099),
    ],

    "ssp245": [
        (2035, 2064),
        (2070, 2099),
    ],

    "ssp370": [
        (2035, 2064),
        (2070, 2099),
    ],
}
# ============================================================
# 4. OUTPUT DIRECTORY
# ============================================================
print("set OUTPUT_DIR")
OUTPUT_DIR = (
    "Set it to yours""
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ============================================================
# 5. BUILD NARCliM DIRECTORY
# ============================================================

def build_narclim_directory(
    gcm,
    scenario,
    rcm,
    variable=VARIABLE,
    frequency=FREQUENCY,
    version=VERSION,
):

    realization = GCM_REALIZATION[gcm]

    directory = (
        f"{OPENDAP_BASE}/"
        f"DD/{DOMAIN}/"
        f"{PUBLISHER}/"
        f"{gcm}/"
        f"{scenario}/"
        f"{realization}/"
        f"{rcm}/"
        f"{version}/"
        f"{frequency}/"
        f"{variable}/"
        f"latest/"
    )

    return directory


# ============================================================
# 6. BUILD NARCliM FILE NAME
# ============================================================

def build_narclim_filename(
    gcm,
    scenario,
    rcm,
    start_year,
    end_year,
    start_suffix="010100",
    end_suffix="123123",
):

    realization = GCM_REALIZATION[gcm]

    start_date = f"{start_year}{start_suffix}"
    end_date = f"{end_year}{end_suffix}"

    filename = (
        f"{VARIABLE}_"
        f"{DOMAIN}_"
        f"{gcm}_"
        f"{scenario}_"
        f"{realization}_"
        f"{PUBLISHER}_"
        f"{rcm}_"
        f"{VERSION}_"
        f"{FREQUENCY}_"
        f"{start_date}-{end_date}.nc"
    )

    return filename


# ============================================================
# 7. BUILD COMPLETE OPENDAP URL
# ============================================================

def build_narclim_url(
    gcm,
    scenario,
    rcm,
    start_year,
    end_year,
):

    directory = build_narclim_directory(
        gcm=gcm,
        scenario=scenario,
        rcm=rcm,
    )

    filename = build_narclim_filename(
        gcm=gcm,
        scenario=scenario,
        rcm=rcm,
        start_year=start_year,
        end_year=end_year,
    )

    return directory + filename


# ============================================================
# 8. PROCESS ONE YEAR / ONE FILE
# ============================================================

def process_rx1day_file(
    gcm,
    scenario,
    rcm,
    year,
    bbox,
):

    url = build_narclim_url(
        gcm=gcm,
        scenario=scenario,
        rcm=rcm,
        start_year=year,
        end_year=year,
    )

    print("\n" + "=" * 70)
    print(f"GCM      : {gcm}")
    print(f"Scenario : {scenario}")
    print(f"RCM      : {rcm}")
    print(f"Year     : {year}")
    print("=" * 70)

    print("Opening:")
    print(url)

    ds = None

    try:

        # ----------------------------------------------------
        # Open dataset
        # ----------------------------------------------------

        ds = xr.open_dataset(
            url,
            engine="netcdf4",
        )

        # ----------------------------------------------------
        # Identify precipitation variable
        # ----------------------------------------------------

        pr = ds[VARIABLE]

        print("\nPrecipitation attributes:")
        print(pr.attrs)

        # ----------------------------------------------------
        # 2-D geographic coordinates
        # ----------------------------------------------------

        lat_2d = ds["lat"]
        lon_2d = ds["lon"]

        # ----------------------------------------------------
        # Create NSW geographic mask
        # ----------------------------------------------------

        mask = (
            (lat_2d >= bbox["lat_min"])
            & (lat_2d <= bbox["lat_max"])
            & (lon_2d >= bbox["lon_min"])
            & (lon_2d <= bbox["lon_max"])
        )

        # ----------------------------------------------------
        # Find rotated-grid indices covering NSW
        # ----------------------------------------------------

        rlat_has_data = mask.any(dim="rlon").compute()
        rlon_has_data = mask.any(dim="rlat").compute()

        rlat_indices = np.where(rlat_has_data.values)[0]
        rlon_indices = np.where(rlon_has_data.values)[0]

        if len(rlat_indices) == 0 or len(rlon_indices) == 0:
            raise ValueError(
                "No grid cells found inside the NSW bounding box."
            )

        rlat_min = int(rlat_indices.min())
        rlat_max = int(rlat_indices.max())

        rlon_min = int(rlon_indices.min())
        rlon_max = int(rlon_indices.max())

        # ----------------------------------------------------
        # Subset spatial domain
        # ----------------------------------------------------

        subset = ds.isel(
            rlat=slice(rlat_min, rlat_max + 1),
            rlon=slice(rlon_min, rlon_max + 1),
        )

        # ----------------------------------------------------
        # Calculate daily precipitation
        #
        # IMPORTANT:
        # If pr is kg m-2 s-1, multiply by 3600 before
        # daily aggregation.
        # ----------------------------------------------------

        pr_daily = (
            subset[VARIABLE]
            .resample(time="1D")
            .sum(dim="time")
        )

        # ----------------------------------------------------
        # Convert to mm if hourly pr is kg m-2 s-1
        # ----------------------------------------------------

        units = str(
            subset[VARIABLE].attrs.get("units", "")
        ).lower()

        if (
            "kg" in units
            and "s-1" in units
        ):
            pr_daily_mm = pr_daily * 3600

        elif "mm" in units:
            pr_daily_mm = pr_daily

        else:
            print(
                f"WARNING: Unknown precipitation units: {units}"
            )
            print(
                "Assuming precipitation is already in mm."
            )
            pr_daily_mm = pr_daily

        # ----------------------------------------------------
        # RX1day
        #
        # Maximum daily precipitation for the year
        # ----------------------------------------------------

        rx1day = pr_daily_mm.max(
            dim="time",
            skipna=True,
        )

        # ----------------------------------------------------
        # Apply exact NSW mask
        # ----------------------------------------------------

        lat_subset = subset["lat"]
        lon_subset = subset["lon"]

        mask_subset = (
            (lat_subset >= bbox["lat_min"])
            & (lat_subset <= bbox["lat_max"])
            & (lon_subset >= bbox["lon_min"])
            & (lon_subset <= bbox["lon_max"])
        )

        rx1day_nsw = rx1day.where(
            mask_subset
        )

        # ----------------------------------------------------
        # Load only the final annual RX1day result
        # ----------------------------------------------------

        rx1day_nsw = rx1day_nsw.compute()

        # ----------------------------------------------------
        # NSW maximum
        # ----------------------------------------------------

        max_value = float(
            rx1day_nsw.max(skipna=True).values
        )

        print(
            f"Maximum NSW RX1day: "
            f"{max_value:.2f} mm"
        )

        return max_value, rx1day_nsw

    finally:

        # ----------------------------------------------------
        # Always close dataset
        # ----------------------------------------------------

        if ds is not None:
            ds.close()


# ============================================================
# 9. PROCESS MULTIPLE YEARS
# ============================================================

def process_period(
    gcm,
    scenario,
    rcm,
    start_year,
    end_year,
    bbox,
):

    annual_results = []
    max_values = []
    log_records = []

    for year in range(
        start_year,
        end_year + 1,
    ):

        try:

            max_value, rx1day_nsw = (
                process_rx1day_file(
                    gcm=gcm,
                    scenario=scenario,
                    rcm=rcm,
                    year=year,
                    bbox=bbox,
                )
            )

            # ------------------------------------------------
            # Add actual year as coordinate
            # ------------------------------------------------

            rx1day_nsw = rx1day_nsw.expand_dims(
                year=[year]
            )

            annual_results.append(
                rx1day_nsw
            )

            max_values.append(
                max_value
            )

            # ------------------------------------------------
            # Success log
            # ------------------------------------------------

            log_records.append({
                "year": year,
                "gcm": gcm,
                "scenario": scenario,
                "rcm": rcm,
                "status": "SUCCESS",
                "max_rx1day_nsw_mm": max_value,
                "error": "",
            })

            print(
                f"SUCCESS: {year}"
            )

        except Exception as e:

            error_message = str(e)

            print(
                f"FAILED: {year}"
            )

            print(
                f"ERROR: {error_message}"
            )

            # ------------------------------------------------
            # Error log
            # ------------------------------------------------

            log_records.append({
                "year": year,
                "gcm": gcm,
                "scenario": scenario,
                "rcm": rcm,
                "status": "FAILED",
                "max_rx1day_nsw_mm": np.nan,
                "error": error_message,
            })

        finally:

            # Release memory after every year
            gc.collect()

    # --------------------------------------------------------
    # Check successful years
    # --------------------------------------------------------

    if not annual_results:

        raise RuntimeError(
            "No files were successfully processed."
        )

    # --------------------------------------------------------
    # Combine all years
    # --------------------------------------------------------

    result = xr.concat(
        annual_results,
        dim="year",
    )

    return (
        result,
        max_values,
        log_records,
    )


# ============================================================
# 10. MAIN PROCESSING
# ============================================================

def main():

    all_log_records = []

    # --------------------------------------------------------
    # Example: historical only
    # --------------------------------------------------------

    #scenario = "historical"

    for rcm in RCMS:

        for gcm in MODELS:
            for scenario in SCENARIOS:

                for start_year, end_year in PERIODS[scenario]:

                    print("\n")
                    print("#" * 80)
                    print(
                        f"PROCESSING: "
                        f"{gcm} | {scenario} | {rcm} | "
                        f"{start_year}-{end_year}"
                    )
                    print("#" * 80)

                    # ------------------------------------------------
                    # Process period
                    # ------------------------------------------------

                    (
                        rx1day_result,
                        max_values,
                        log_records,
                    ) = process_period(
                        gcm=gcm,
                        scenario=scenario,
                        rcm=rcm,
                        start_year=start_year,
                        end_year=end_year,
                        bbox=NSW_BBOX,
                    )

                    # ------------------------------------------------
                    # Add logs to master list
                    # ------------------------------------------------

                    all_log_records.extend(
                        log_records
                    )

                    # ------------------------------------------------
                    # Output filename
                    # ------------------------------------------------

                    output_filename = (
                        f"RX1day_"
                        f"{rcm}_"
                        f"{gcm}_"
                        f"{scenario}_"
                        f"{start_year}-{end_year}.nc"
                    )

                    output_file = os.path.join(
                        OUTPUT_DIR,
                        output_filename,
                    )

                    # ------------------------------------------------
                    # Save RX1day NetCDF
                    # ------------------------------------------------

                    print("\nSaving:")
                    print(output_file)

                    rx1day_result.to_netcdf(
                        output_file,
                        mode="w",
                    )

                    print(
                        "Saved successfully."
                    )

                    # ------------------------------------------------
                    # Clean memory
                    # ------------------------------------------------

                    del rx1day_result
                    del max_values

                    gc.collect()

    # ========================================================
    # SAVE MASTER PROCESSING LOG
    # ========================================================

    if all_log_records:

        log_df = pd.DataFrame(
            all_log_records
        )

        log_file = os.path.join(
            OUTPUT_DIR,
            "RX1day_processing_log.csv",
        )

        log_df.to_csv(
            log_file,
            index=False,
        )

        print("\n")
        print("=" * 80)
        print("PROCESSING LOG SAVED")
        print("=" * 80)
        print(log_file)

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print("\nProcessing summary:")
        print(
            log_df["status"]
            .value_counts()
        )


# ============================================================
# 11. RUN
# ============================================================

if __name__ == "__main__":
    main()
