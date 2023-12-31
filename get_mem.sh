#!/bin/bash

# This script writes the "Mem" line from the free -h command and the current time in one line every second to a file, and also prints it to the console

output_file="memory_usage_log.txt"

# Header line
header="                                    total        used        free      shared  buff/cache   available"

# Write header to both console and file
echo "$header" | tee $output_file

while true; do
    # Get the "Mem" line from free -h and store it
    mem_usage=$(free -h | grep "Mem:")

    # Format the output string
    output="$(date +"%a %b %d %H:%M:%S") | $mem_usage"

    # Write to both console and file
    echo -e "$output" | tee -a $output_file

    # Wait for one second
    sleep 1
done
