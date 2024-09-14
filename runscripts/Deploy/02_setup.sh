git clone --recursive -b main https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
python3 tools/scripts/build_ubuntu_x64_ort.py $(nproc)
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export PYTHONPATH=/root/workspace/mmdet/MyConfigs/:$PYTHONPATH
export PYTHONPATH=/root/workspace/mmdet/MyConfigs/custom_components/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH

mim install mmdet
pip install rasterio scikit-image