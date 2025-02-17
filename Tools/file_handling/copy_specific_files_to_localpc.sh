# Define source base path and destination base path
SOURCE_BASE="/mnt/standard/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final"
DESTINATION_BASE="/mnt/d"

# Read each line in file_list.txt, clean up any extra characters, and copy using rsync
while IFS= read -r file; do
    # Remove potential carriage returns or other trailing whitespace
    clean_file=$(echo "$file" | tr -d '\r')

    # Use rsync with -R to preserve directory structure
    rsync -R "$SOURCE_BASE$clean_file" "$DESTINATION_BASE"
done < specific_file_list.txt
