#!/bin/bash


#!/bin/bash

BASE_URL="https://spdf.gsfc.nasa.gov/pub/data/gold/level1c/2025/"  # URL to download from
OUTPUT_DIR="D:/gold_level1c_2025"  # Output directory
mkdir -p "$OUTPUT_DIR"  # Create directory if it doesn't exist

# Get folder list from NASA's website
curl -s "$BASE_URL" | grep -oP '(?<=href=")[^"]*/' > folder_list.txt


# Path to Julian days file
days_to_download="days.txt"

# Check if the file exists
if [[ ! -f "$days_to_download" ]]; then
    echo "Error: Julian days file not found at $days_to_download"
    exit 1
fi

# Filter folder list based on `days_to_download`
grep -Ff "$days_to_download" folder_list.txt > selected_folders.txt

# Download only matching folders
while read -r folder; do
    echo "Processing folder: $folder"
    
    LOCAL_FOLDER="${OUTPUT_DIR}/${folder}"
    mkdir -p "$LOCAL_FOLDER"
    
    # Get file list inside the folder
    curl -s "${BASE_URL}${folder}" | grep -oP '(?<=href=")[^"]*' | grep "DAY" > files_in_folder.txt 
    
    # Download each file if not already present
    while read -r file; do
        LOCAL_FILE="${LOCAL_FOLDER}/${file}"
        if [ -f "$LOCAL_FILE" ]; then
            echo "File ${file} already exists. Skipping."
        else
            echo "Downloading ${file} from ${folder}..."
            aria2c -x 16 -s 16 -d "$LOCAL_FOLDER" -o "$file" "${BASE_URL}${folder}${file}" || \
            curl -s -o "$LOCAL_FILE" "${BASE_URL}${folder}${file}"
        fi
    done < files_in_folder.txt
    
done < selected_folders.txt

echo "Download complete. Files stored in $OUTPUT_DIR."

#to run file ./file_name.sh