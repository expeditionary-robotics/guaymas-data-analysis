# Setup
1. Clone the repository locally: `git clone git@github.com:expeditionary-robotics/guaymas-cruise-analysis.git`
2. Download the relevent cruise data from [Dropbox](https://www.dropbox.com/sh/aeqfwvkzprejffe/AAAEDHlltJNRAnlv2FT3iiW_a?dl=0). Reach out to Victoria or Genevieve if you need to be added to the Dropbox data share. 
3. Place the downloaded `data` and `ouptut` directories in the root directory of the repostiory, i.e., in `guaymas-cruise-analysis/data` and `guaymas-cruise-analysis/output`.
4. Install the `pipenv` virtual Python environment: `pipenv install`. 
5. Active the Python virtual environment: `pipenv shell`. You should always activate the virtual environment when running code in this respository. This will ensure that you have the right Python packages and versions installed and that your PYTHONPATH is correctly set. 
6. Run the data analysis scripts in the `data_processing` folder. These scripts will generate data products in the folder `data` and graphs/visualization products in the folder `output`. 

# Package Details 
This package implements the analysis pipeline for the RR2107 Guaymas cruise data. The `sentry_data_analysis` folder is organized as a python package and has two main folders:
* `data_processing`: scripts for data processing
* `utils`: utility files to support scripts for data processing

The analysis pipeline consists of the following steps:
1. `sentry_parse_dive.py`: convert the raw sentry science data into a csv format and save in SENTRY_DATA
2. `sentry_correct_background.py`: read in the sentry science data and fit a linear trend to account for background correlation. Save as a new csv with original science data an appended `<science_variable>_anom` anomaly columns.
3. `sentry_detect_plume.py`: read in the anomaly data (or science data) and perform binary plume detection. Save as a new csv with science data and an appended `detections` column for binary detections.

These steps can be run for all available sentry dives
`SENTRY_DIVES=("sentry607" "sentry608" "sentry609" "sentry610" "sentry611" "sentry612" "sentry613")`
using the script file `run_for_all_sentry_dives.sh`, e.g., `./run_for_all_sentry_dives.sh sentry_parse_dive.py`. There are additional dives `sentry613_full`, `sentry613_ascent`, and `sentry613_descent` that correspond respectively to the full, ascent, and descent portions of the final transect dive.

The 3 step pipeline can be run for all available sentry dives using the script `run_analysis.sh`.

There are a few other analysis scripts:
1. `tiltmeter_parse_current.py` takes as input a tiltmeter deployment, e.g., `A1`, `A2`, `A3`, or `B1` and produces a processed tiltmeter data file and training data file.
3. `ctd_convert_profiles.py`: takes a list of ctd profiles in an argument list at the top of the script and produces a processed ctd file.


Finally, the main piece: the script `sentry_visualize_dive.py` produces all of visualizations and videos, etc. and takes as input a sentry dive names and phase number (optional). The flags at the top of the file allow the visualization to be configured.
