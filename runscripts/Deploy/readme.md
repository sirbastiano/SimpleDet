First install mmdeploy using 02_setup.sh



if model conversion fails after onnx conversion try to run:

mo --input_model="/Data_large/marine/PythonProjects/MMDET/Deploy_out/end2end.onnx" --output_dir="/Data_large/marine/PythonProjects/MMDET/Deploy_out" --output="dets,labels" --input="input" --input_shape="[1, 1, 800, 1344]"


Output:

mo --input_model="/Data_large/marine/PythonProjects/MMDET/Deploy_out/end2end.onnx" --output_dir="/Data_large/marine/PythonProjects/MMDET/Deploy_out" --output="dets,labels" --input="input" --input_shape="[1, 1, 800, 1344]"
Check for a new version of Intel(R) Distribution of OpenVINO(TM) toolkit here https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?cid=other&source=prod&campid=ww_2023_bu_IOTG_OpenVINO-2022-3&content=upg_all&medium=organic or on https://github.com/openvinotoolkit/openvino
[ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
[ SUCCESS ] Generated IR version 11 model.
[ SUCCESS ] XML file: /Data_large/marine/PythonProjects/MMDET/Deploy_out/end2end.xml
[ SUCCESS ] BIN file: /Data_large/marine/PythonProjects/MMDET/Deploy_out/end2end.bin