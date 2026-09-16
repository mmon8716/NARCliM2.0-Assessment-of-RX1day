# NARCliM2.0-Assessment-of-RX1day
RX1-day extreme rainfall trends across NSW using NARClim2.0 projections
NARCliM2.0 RX1day Extreme Rainfall Analysis – NSW

This project calculates and analyses RX1day extreme rainfall across New South Wales (NSW) using hourly precipitation data from NARCliM2.0 climate simulations.

Workflow

The analysis is divided into three main Python scripts:


1. extract_narclim_rx1day_nsw.py

Extracts and processes NARCliM2.0 hourly precipitation data for NSW.

Main tasks:

Reads NARCliM2.0 NetCDF files.
Identifies grid cells covering NSW using latitude/longitude coordinates.
Calculates daily rainfall from hourly precipitation.
Calculates annual RX1day — the maximum 1-day precipitation for each year.
Processes historical and future climate simulations.
Saves the processed RX1day results for further analysis.

2. Fig1_2.py

Generates Figure 1 and Figure 2 for the written assessment.

Main tasks:

Reads the extracted RX1day datasets.
Calculates model/ensemble statistics.
Produces spatial maps of extreme rainfall.
Compares historical and future periods/scenarios.
Generates figures showing changes in RX1day across NSW.

3. Table1.py

Generates Table 1 for the written assessment.

Main tasks:

Reads the processed RX1day data.
Summarises the available climate models, scenarios and periods.
Calculates relevant ensemble statistics.
Produces a summary table for reporting.

Data

The analysis uses NARCliM2.0 hourly precipitation data, including multiple:

Global Climate Models (GCMs)
Regional Climate Models (RCMs)
Historical simulations
Future SSP scenarios
Future climate periods
Outputs

The workflow produces:


Processed RX1day NetCDF files
        ↓
    Figure 1
    Figure 2
        ↓
     Table 1
     

The outputs can be used to assess spatial patterns, future changes, model agreement and uncertainty in extreme rainfall across NSW.

Requirements

Python 3.x with:

xarray
numpy
pandas
netCDF4
matplotlib
cartopy
scipy
Running the analysis

Run the scripts in the following order:

python extract_narclim_rx1day_nsw.py
python Fig1_2.py
python Table1.py

The extraction script should be completed first because Fig1_2.py and Table1.py use its processed outputs.

Notes

The workflow is designed to process multiple NARCliM2.0 model simulations consistently and provide reproducible inputs for the written climate assessment. Future improvements include faster processing, parallel computing, missing-data checks, automated validation, and improved assessment of model agreement and uncertainty.
