#!/bin/bash

# Define the directory where the logs are stored
LOG_DIR=$HOME/mydata/projects/TabulaSapiens/logs

# Define how many lines from the end of the log files you want to display
TAIL_LINES=10

# Function to display job status and log file content
monitor_jobs() {
    while true; do
        echo "===================================="
        echo "Current Running Jobs:"
        # List all running jobs
        squeue --user $USER --format="%.18i %.9P %.25j %.8u %.2t %.10M %.6D %R"

        echo "------------------------------------"
        echo "Latest Log Files (Last $TAIL_LINES lines):"

        # Find the most recent log file
        LATEST_LOG=$(ls -t $LOG_DIR/*.log | head -n 1)

        if [ -z "$LATEST_LOG" ]; then
            echo "No log files found."
        else
            # Display the last few lines of the most recent log file
            tail -n $TAIL_LINES $LATEST_LOG
        fi

        # Wait for 10 seconds before updating
        sleep 10
    done
}

# Run the monitoring function
monitor_jobs
