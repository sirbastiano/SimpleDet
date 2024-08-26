#!/bin/bash
clear

# Log file path
LOG_FILE="/Data_large/marine/PythonProjects/MMDET/runscripts/Logs/supplement_S2_single.log"

# Function to handle keyboard interrupt (Ctrl+C)
interrupt_handler() {
    echo "Keyboard interrupt received. Deleting $OUTPUT_DIR..."
    rm -rf ${OUTPUT_DIR}
    echo "Folder $OUTPUT_DIR deleted."
    echo "Script interrupted by the user at $(date)" >> $LOG_FILE
    exit 1
}

# This script installs the MMDET package and its dependencies
# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda and try again."
    echo "See https://docs.anaconda.com/anaconda/install/ for more information."
    exit 1
fi

source $(conda info --base)/etc/profile.d/conda.sh
if conda activate openmmlab; then
    echo "openmmlab environment activated"
    echo "$(date): openmmlab environment activated" >> $LOG_FILE
else
    echo "openmmlab environment not found"
    echo "$(date): openmmlab environment not found" >> $LOG_FILE
    exit 1
fi

# Default values for the arguments
BANDS=("2" "3" "4" "8")
SEEDS=(42 71 18) # 53 89
BATCH_SIZES=(2)
LEARNING_RATE=(0.01)
RESIZE=2048

# Parsing command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seed) IFS=',' read -r -a SEEDS <<< "$2"; shift ;;
        --batch_size) IFS=',' read -r -a BATCH_SIZES <<< "$2"; shift ;;
        --learning_rate) LEARNING_RATE="$2"; shift ;;
        --resize) RESIZE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run the Python script for each combination of BAND, SEED, and BATCH_SIZE
for BAND in "${BANDS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
            for LEARNING_RATE in "${LEARNING_RATE[@]}"; do
                CONFIG="BAND=$BAND, SEED=$SEED, BATCH_SIZE=$BATCH_SIZE, LEARNING_RATE=$LEARNING_RATE, RESIZE=$RESIZE"
                echo "Running with $CONFIG"
                echo "$(date): Running with $CONFIG" >> $LOG_FILE
                python /Data_large/marine/PythonProjects/MMDET/MyConfigs/ExecutorSen.py --band "$BAND" --seed "$SEED" --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --resize "$RESIZE"
                
                if [ $? -eq 0 ]; then
                    echo "$(date): Completed successfully with $CONFIG" >> $LOG_FILE
                else
                    echo "$(date): Failed with $CONFIG" >> $LOG_FILE
                fi
            done
        done
    done
done

# Trap keyboard interrupt and call the interrupt_handler function
trap interrupt_handler SIGINT