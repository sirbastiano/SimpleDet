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

# RUNNING:
python /Data_large/marine/PythonProjects/MMDET/runscripts/SenTrainer.py --seed 5  --band '[2,8]'
python /Data_large/marine/PythonProjects/MMDET/runscripts/SenTrainer.py --seed 99 --band '[2,8]'
python /Data_large/marine/PythonProjects/MMDET/runscripts/SenTrainer.py --seed 16 --band '[2,8]'
python /Data_large/marine/PythonProjects/MMDET/runscripts/SenTrainer.py --seed 40 --band '[2,8]'
python /Data_large/marine/PythonProjects/MMDET/runscripts/SenTrainer.py --seed 23 --band '[2,8]'
python /Data_large/marine/PythonProjects/MMDET/runscripts/SenTrainer.py --seed 71 --band '[2,8]'
python /Data_large/marine/PythonProjects/MMDET/runscripts/SenTrainer.py --seed 18 --band '[2,8]'



# Trap keyboard interrupt and call the interrupt_handler function
trap interrupt_handler SIGINT