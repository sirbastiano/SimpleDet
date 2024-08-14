
#!/bin/bash
clear
# Function to handle keyboard interrupt (Ctrl+C)
interrupt_handler() {
    echo "Keyboard interrupt received. Deleting $OUTPUT_DIR..."
    rm -rf ${OUTPUT_DIR}
    echo "Folder $OUTPUT_DIR deleted."
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
else
    echo "openmmlab environment not found"
    exit 1
fi

# Caller:
# Default values for the arguments
BANDS=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "5,10" "5,12" "3,4,7" "5,10,11" "3,4,7,11" "3,5,7,11")
SEEDS=(53 89)
BATCH_SIZES=(2 3)
LEARNING_RATE=(0.001)
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
                echo "Running with BAND=$BAND, SEED=$SEED, BATCH_SIZE=$BATCH_SIZE, LEARNING_RATE=$LEARNING_RATE, RESIZE=$RESIZE"
                python /Data_large/marine/PythonProjects/MMDET/MyConfigs/ExecutorM.py --band "$BAND" --seed "$SEED" --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --resize "$RESIZE"
            done
        done
    done
done
# Trap keyboard interrupt and call the interrupt_handler function
trap interrupt_handler SIGINT