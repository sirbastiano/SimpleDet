#!/bin/sh

DEPLOY_CFG_PATH="/root/workspace/mmdeploy/configs/mmdet/detection/detection_openvino_dynamic-800x1344.py" # The deployment configuration of mmdeploy for the model, including the type of inference framework, whether quantize, whether the input shape is dynamic, etc. There may be a reference relationship between configuration files,
MODEL_CFG_PATH="/root/workspace/mmdet/checkpoints/VENuS/Single/perfect_b10/18_BS_3_LR_0.0008_ME_30_OPT_SGD/vfnet_r18.py" # Model configuration for algorithm library
MODEL_CHECKPOINT_PATH="/root/workspace/mmdet/checkpoints/VENuS/Single/perfect_b10/18_BS_3_LR_0.0008_ME_30_OPT_SGD/epoch_30.pth" # Model checkpoint file path

INPUT_IMG="/root/workspace/mmdet/tgt_imgs/ASH_L0_09936_20190618_MultiLayer_mask_OK.tif" # Input image path or point cloud file used for testing during the model conversion.
# INPUT_IMG="/root/workspace/mmdet/tgt_imgs/ASH_L0_09936_20190618_MultiLayer_mask_OK.tif" # Input image path or point cloud file used for testing during the model conversion.
# TEST_IMG ="/root/workspace/mmdet/checkpoints/ASH_L0_02686_20180203_CoReg_mask_OK.tif" # The path of the image file that is used to test the model. If not specified, it will be set to None.
WORK_DIR="/root/workspace/mmdet/deploy/deployed_model" # The path of the work directory that is used to save logs and models.
DEVICE="cuda:0" # The device used for model conversion. If not specified, it will be set to cpu. For trt, use cuda:0 format.
DEVICE="cpu" # The device used for model conversion. If not specified, it will be set to cpu. For trt, use cuda:0 format.
CALIB_DATA_CFG="" # Only valid in int8 mode. The config used for calibration. If not specified, it will be set to None and use the “val” dataset in the model config for calibration.



clear
cd /root/workspace/mmdeploy/

python3 ./tools/deploy.py \
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