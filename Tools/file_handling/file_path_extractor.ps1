# Variables
$directory = "E:\EgoExoEMS\TestData\20-09-2024\chas\cardiac_arrest"  # Directory to search
$extension = "MP4"  # File extension to search for
$output_file = "file_paths.txt"  # Output file to store the paths

# Find files with the specified extension and write their full paths to the output file
Get-ChildItem -Path $directory -Recurse -Filter "*.$extension" | ForEach-Object { $_.FullName } > $output_file

# Print a message when done
Write-Host "File paths with extension .$extension written to $output_file"
