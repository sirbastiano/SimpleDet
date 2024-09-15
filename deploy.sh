#!/bin/sh

export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export PYTHONPATH=/Data_large/marine/PythonProjects/MMDET/MyConfigs/:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH

BASE="/Data_large/marine/PythonProjects/MMDET/"

DEPLOY_CFG_PATH="${BASE}runscripts/Deploy/mmdeploy/configs/mmdet/_base_/base_openvino_static-800x1344.py" # The deployment configuration of mmdeploy for the model, including the type of inference framework, whether quantize, whether the input shape is dynamic, etc. There may be a reference relationship between configuration files,
MODEL_CFG_PATH="${BASE}checkpoints/VENuS/Single/perfect_b10/18_BS_3_LR_0.0008_ME_30_OPT_SGD/vfnet_r18.py" # Model configuration for algorithm library
MODEL_CHECKPOINT_PATH="${BASE}checkpoints/VENuS/Single/perfect_b10/18_BS_3_LR_0.0008_ME_30_OPT_SGD/epoch_30.pth" # Model checkpoint file path


INPUT_IMG="/Data_large/marine/PythonProjects/MMDET/resampled_10th_band.tif" # Input image path or point cloud file used for testing during the model conversion.


# TEST_IMG ="/root/workspace/mmdet/checkpoints/ASH_L0_02686_20180203_CoReg_mask_OK.tif" # The path of the image file that is used to test the model. If not specified, it will be set to None.
WORK_DIR="${BASE}Deploy_out" # The path of the work directory that is used to save logs and models.



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
    --dump-info
    # --show \
    # --test-img ${TEST_IMG} \
    # --calib-dataset-cfg ${CALIB_DATA_CFG} \


# RUN:
# FROM WORKSPACE: cd /root/workspace
# source mmdet/deploy/deploy.sh