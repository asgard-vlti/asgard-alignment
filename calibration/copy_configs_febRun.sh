#!/bin/bash

# Check if at least two arguments are provided: mode + at least one beam
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <file name to save to in /home/asg/ben_feb2026_bld_telem/filename"
    exit 1
fi

# First argument is beam
beam="$1"

file_label="$2"

# Get current full timestamp: YYYY-MM-DDTHH-MM-SS
full_timestamp=$(date +%F"T"%H-%M-%S)

# Extract just the date part: YYYY-MM-DD
today=${full_timestamp%%T*}

# Base destination directory
base_dst_dir="/home/asg/ben_feb2026_bld_telem/${today}"

# Create base destination folder if it doesn't exist
mkdir -p "${base_dst_dir}"

# Loop over all provided beam numbers

src="/usr/local/etc/baldr/baldr_config_${beam}.toml"
dst="${base_dst_dir}/baldr_config_${beam}_${full_timestamp}_${file_label}.toml"

if [ -f "$src" ]; then
    cp "$src" "$dst"
    echo "Copied $src → $dst"
else
    echo "Warning: $src does not exist, skipping."
fi

# #!/bin/bash

# # Check if at least one beam number is provided
# if [ "$#" -lt 1 ]; then
#     echo "Usage: $0 <beam_number1> [beam_number2 ...]"
#     exit 1
# fi

# # Get current full timestamp: YYYY-MM-DDTHH-MM-SS
# full_timestamp=$(date +%F"T"%H-%M-%S)

# # Extract just the date part: YYYY-MM-DD
# today=${full_timestamp%%T*}

# # Base destination directory
# base_dst_dir="/usr/local/etc/baldr/rtc_config/${today}"

# # Create base destination folder if it doesn't exist
# mkdir -p "${base_dst_dir}"

# # Loop over all provided beam numbers
# for beam in "$@"; do
#     src="/usr/local/etc/baldr/baldr_config_${beam}.toml"
#     dst="${base_dst_dir}/baldr_config_${beam}_${full_timestamp}.toml"

#     if [ -f "$src" ]; then
#         cp "$src" "$dst"
#         echo "Copied $src → $dst"
#     else
#         echo "Warning: $src does not exist, skipping."
#     fi
# done
