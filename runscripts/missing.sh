
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
BANDS=("3,4,7" "5,10,11" "3,4,7,11" "3,5,7,11")
SEEDS=(42 71 18 53 89)
BATCH_SIZES=(2 3)
LEARNING_RATE=(0.0008 0.0009 0.001 0.002 0.003 0.004 0.005)
RESIZE=2048

python /Data_large/marine/PythonProjects/MMDET/MyConfigs/ExecutorM.py --band "5,12" --seed "53" --batch_size "2" --learning_rate "0.0008" --resize "$RESIZE"
python /Data_large/marine/PythonProjects/MMDET/MyConfigs/ExecutorM.py --band "5,12" --seed "89" --batch_size "2" --learning_rate "0.0008" --resize "$RESIZE"
python /Data_large/marine/PythonProjects/MMDET/MyConfigs/ExecutorM.py --band "5,12" --seed "18" --batch_size "2" --learning_rate "0.0008" --resize "$RESIZE"

python /Data_large/marine/PythonProjects/MMDET/MyConfigs/ExecutorM.py --band "5,10,11" --seed "42" --batch_size "3" --learning_rate "0.003" --resize "$RESIZE"

python /Data_large/marine/PythonProjects/MMDET/MyConfigs/ExecutorM.py --band "5,10" --seed "42" --batch_size "2" --learning_rate "0.001" --resize "$RESIZE"
python /Data_large/marine/PythonProjects/MMDET/MyConfigs/ExecutorM.py --band "5,10" --seed "71" --batch_size "2" --learning_rate "0.001" --resize "$RESIZE"
python /Data_large/marine/PythonProjects/MMDET/MyConfigs/ExecutorM.py --band "5,10" --seed "53" --batch_size "2" --learning_rate "0.001" --resize "$RESIZE"


# Trap keyboard interrupt and call the interrupt_handler function
trap interrupt_handler SIGINT