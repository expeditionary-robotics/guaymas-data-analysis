#!/bin/bash
# Runs a specifed python script with arguments for all Sentry dives
#
# Example uses:
#   run_for_all_sentry_dives.sh script.py (arg2) (arg3) (arg4)
#
# Additional args
#   script.py : name of Python script followed by any of its arguments

SENTRY_DIVES=("sentry607" "sentry608" "sentry609" "sentry610" "sentry611" "sentry612" "sentry613")

for dive in ${SENTRY_DIVES[@]};
do
  echo "Running: python ${@:0:1} $dive ${@:2}"
  python ${@:0:1} $dive ${@:2}
done
