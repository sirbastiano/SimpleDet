#!/bin/sh

export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export PYTHONPATH=/Data_large/marine/PythonProjects/MMDET/MyConfigs/:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH

BASE='/Data_large/marine/PythonProjects/MMDET/'
DEPLOY_CFG_PATH='/Data_large/marine/PythonProjects/MMDET/runscripts/Deploy/mmdeploy/configs/mmdet/detection/detection_onnxruntime_static.py'
MODEL_CFG_PATH='/Data_large/marine/PythonProjects/MMDET/notebooks/OpenVINO/config.py'
MODEL_CHECKPOINT_PATH='/Data_large/marine/PythonProjects/MMDET/notebooks/OpenVINO/weights.pth'
INPUT_IMG="/Data_large/marine/PythonProjects/MMDET/resampled_10th_band.tif"
WORK_DIR='/Data_large/marine/PythonProjects/MMDET/notebooks/OpenVINO/export'


DEVICE="cuda:0" # The device used for model conversion. If not specified, it will be set to cpu. For trt, use cuda:0 format.
DEVICE="cpu" # The device used for model conversion. If not specified, it will be set to cpu. For trt, use cuda:0 format.
CALIB_DATA_CFG="" # Only valid in int8 mode. The config used for calibration. If not specified, it will be set to None and use the “val” dataset in the model config for calibration.


python3 ${BASE}runscripts/Deploy/mmdeploy/tools/deploy.py \
    ${DEPLOY_CFG_PATH} \
    ${MODEL_CFG_PATH} \
    ${MODEL_CHECKPOINT_PATH} \
    ${INPUT_IMG} \
    --work-dir ${WORK_DIR} \
    --device ${DEVICE} \
    --log-level INFO \
    --show \
    --dump-info
    # --test-img ${TEST_IMG} \
    # --calib-dataset-cfg ${CALIB_DATA_CFG} \


# RUN:
# FROM WORKSPACE: cd /root/workspace
# source mmdet/deploy/deploy.sh