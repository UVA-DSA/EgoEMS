#!/bin/bash

# Variables
directory="/standard/storage/CognitiveEMS_Datasets/North_Garden/May_2024/May24_updated_structure/ego_camera/"
       # Directory to start the search from (e.g., ".")
extension="MP4"  # File extension to search for (e.g., "txt")
output_file="./file_paths.txt"   # Output file to store the paths (e.g., "file_paths.txt")


# Find files with the specified extension and write their full paths to the output file
find "$directory" -type f -name "*.$extension" > "$output_file"

# Print a message when done
echo "File paths with extension .$extension written to $output_file"
