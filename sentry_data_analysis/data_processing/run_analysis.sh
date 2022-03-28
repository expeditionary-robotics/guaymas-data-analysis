# Parse the raw data files into processed data files
./run_for_all_sentry_dives.sh sentry_parse_dive.py

#  Correct for background stratification using the trend method
./run_for_all_sentry_dives.sh sentry_correct_background.py trend

#  Detect binary plume obserations from plume anomolies
./run_for_all_sentry_dives.sh sentry_detect_plume.py