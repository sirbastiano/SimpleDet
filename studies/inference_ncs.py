"""
DOCKER (CLI):
docker run -it --rm --platform linux/arm64     --net=host     --privileged     -v /dev:/dev     -v /home/pi/ir_output:/home/mount/     -v /tmp/.X11-unix:/tmp/.X11-unix     -e DISPLAY=$DISPLAY     --device-cgroup-rule='c 189:* rmw'     -v /dev/bus/usb:/dev/bus/usb     -v $HOME/.Xauthority:/root/.Xauthority     -u root     -w /home/     ghcr.io/uaws/openvino-on-aarch64:latest     /bin/bash

MODEL COMPILATION (CLI):
mo --input_model resnet18.onnx --output_dir ./ --scale 1 --mean_values [0] --data_type FP16 --input_shape "(1,1,64,64)" --model_name VENUSIR
mo --input_model model_apisonnx.onnx --output_dir ./ --scale 1 --mean_values [0] --data_type FP16 --input_shape "(1,1,64,64)" --model_name VENUSIR
"""


import sys
import logging
import numpy as np
from openvino.runtime import Core  # Updated API for OpenVINO 2022.1
import tifffile as tiff
import time
from pathlib import Path
import argparse
import os

# Initialize logging
def setup_logger(log_filename):
    """
    Set up a logger to write log messages to a file and print to console.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger('inferencer.log')

# Helper function to check if file exists
def check_file_exists(file_path):
    if not isinstance(file_path, str):
        logger.error("file_path must be a string")
        sys.exit(1)
    try:
        return Path(file_path).exists()
    except Exception as e:
        logger.error(f"An error occurred reading the file {Path(file_path).name}: {e}")
        sys.exit(1)

# Save function
def save(obj, output_dir, class_dict, total_time, avg_time, throughput, metadata=None, saver='numpy'):
    """
    Saves inference results along with class dictionary, timing info, and throughput in a .npz file.

    Args:
    obj (numpy.ndarray): The NumPy array or object to save.
    output_dir (str): Directory where the output file will be saved.
    class_dict (dict): Dictionary mapping class indices to class names.
    total_time (float): Total inference time.
    avg_time (float): Average inference time per iteration.
    throughput (float): Throughput inferences per second.
    metadata (dict, optional): Additional metadata to save.
    saver (str, optional): Method for saving the object. Defaults to 'numpy'.
    """
    try:
        # Validate inputs
        if not isinstance(obj, np.ndarray):
            raise ValueError("obj must be a numpy.ndarray.")
        if not isinstance(output_dir, str):
            raise ValueError("output_dir must be a string.")
        if saver not in ['npy', 'numpy']:
            raise ValueError("saver must be 'npy', or 'numpy'.")

        if saver == 'numpy':
            saver = 'npy'

        # Prepare file path
        file_path = output_dir + f".npz"

        # Save multiple objects in a .npz file
        np.savez(file_path, results=obj, class_dict=class_dict, 
                 total_time=total_time, avg_time=avg_time, throughput=throughput, metadata=metadata)

        print(f"Results saved successfully to {file_path}")

    except ValueError as e:
        logger.error(f"Input validation error: {e}")
        raise
    except OSError as e:
        logger.error(f"File saving error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in save function: {e}")
        raise

# Preprocessing function
def preprocess_input(image_path, band_select=1, input_shape=None):
    img = tiff.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from path: {image_path}.")
    
    img = img[:, :, band_select]
    if input_shape is not None:
        img = np.resize(img, input_shape)
        
    img = np.expand_dims(img, axis=0)  # Add channel dimension C
    img = np.expand_dims(img, axis=0)  # Add batch size dimension B
    return img

# Load all .tif images from a directory
def load_images_from_directory(directory):
    image_files = sorted([str(f) for f in Path(directory).rglob('*.tif')])
    if len(image_files) == 0:
        logger.error(f"No .tif images found in directory: {directory}")
        sys.exit(1)
    return image_files

# Batch the images
def batch_images(image_paths, batch_size, input_shape):
    batches = []
    for i in range(0, len(image_paths), batch_size):
        batch = []
        for image_path in image_paths[i:i+batch_size]:
            img = preprocess_input(image_path, input_shape=input_shape)
            batch.append(img)
        batch_input = np.vstack(batch)  # Stack the images into a single batch
        batches.append(batch_input)
    return batches

# Main inference function
def run_inference(args, logger):
    logger.info('Setting parameters:')
    device_name = args.device_name
    logger.info(f'DEVICE: {device_name}')
    
    if device_name != 'MYRIAD':
        logger.error(f'Incorrect Device Name: {device_name}')
        sys.exit(1)

    # Check model files
    model_xml = args.model_xml
    model_bin = args.model_bin
    check_file_exists(model_xml)
    check_file_exists(model_bin)
    
    logger.info(f'MODEL_XML: {model_xml}')
    logger.info(f'MODEL_BIN: {model_bin}')
    
    logger.info('Creating OpenVINO Runtime Core')
    core = Core()
    
    logger.info('Loading the model to MYRIAD device')
    model = core.read_model(model=model_xml)
    compiled_model = core.compile_model(model=model, device_name="MYRIAD")

    logger.info(f'Fetching input images from directory: {args.input_dir}')
    image_files = load_images_from_directory(args.input_dir)

    # Select the class dictionary
    if args.class_dict == "venus":
        class_dict = {0: 'Bulk Carrier', 1: 'ContainerShip', 2: 'General Cargo', 3: 'Other', 4: 'Tanker'}
    elif args.class_dict == "s2":
        class_dict = {0: 'Cargo', 1: 'Fishing', 2: 'Pleasure', 3: 'Sailing', 4: 'Tanker'}
    else:
        logger.error(f"Unknown class dictionary selected: {args.class_dict}")
        sys.exit(1)

    logger.info(f'Using class dictionary: {args.class_dict}')

    # Process all images in batches
    input_shape = tuple(map(int, args.input_size.split(',')))  # Convert input_size to tuple
    batches = batch_images(image_files, args.batch_size, input_shape)
    
    # Warmup Phase
    logger.info(f'Starting warmup phase with {args.warmup_iterations} iterations...')
    for _ in range(args.warmup_iterations):
        compiled_model([batches[0]])  # Use the first batch for warmup
    logger.info('Warmup phase completed.')

    # Actual Inference
    logger.info('Starting actual inference...')
    start_time = time.time()

    predictions = []
    for batch in batches:
        results = compiled_model([batch])
        
        # Process results
        output_layer = compiled_model.output(0)
        results_np = np.array(results[output_layer])  # Extract and convert to numpy array
        max_indices = np.argmax(results_np, axis=1)
        predictions.extend(max_indices)

    end_time = time.time()

    elapsed_time = end_time - start_time
    avg_inference_time = elapsed_time / len(image_files)
    throughput = len(image_files) / elapsed_time  # Throughput: images processed per second

    logger.info(f'Total inference time: {elapsed_time} seconds')
    logger.info(f'Average inference time per image: {avg_inference_time} seconds')
    logger.info(f'Throughput: {throughput} images per second')

    # Log results for each image
    logger.info('Inference results:')
    for i, img_path in enumerate(image_files):
        predicted_class = class_dict[predictions[i]]
        logger.info(f"Image: {img_path} | Predicted: {predicted_class}")

    # Save results
    savepath = args.output + '/inference_results'
    metadata = {
        'model_xml': model_xml,
        'model_bin': model_bin,
        'input_directory': args.input_dir,
        'inference_iterations': len(image_files),
        'warmup_iterations': args.warmup_iterations,
        'batch_size': args.batch_size,
        'input_size': args.input_size
    }
    
    save(obj=np.array(predictions), output_dir=savepath, class_dict=class_dict,
         total_time=elapsed_time, avg_time=avg_inference_time, throughput=throughput, metadata=metadata)
    logger.info('Results saved successfully.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVINO Inference with MYRIAD Device")

    parser.add_argument("--input_dir", type=str, required=True, help="Path to directory containing input images.")
    parser.add_argument("--device_name", type=str, default="MYRIAD", help="Inference device: MYRIAD")
    parser.add_argument("--model_xml", type=str, required=True, help="Path to the model's XML file.")
    parser.add_argument("--model_bin", type=str, required=True, help="Path to the model's BIN file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--warmup_iterations", type=int, default=0, help="Number of warmup iterations before actual inference.")
    parser.add_argument("--inference_iterations", type=int, default=1, help="Number of inference iterations to measure performance.")
    parser.add_argument("--class_dict", type=str, required=True, choices=["venus", "s2"], help="Select the class dictionary to use: 'venus' or 's2'.")
    parser.add_argument("--input_size", type=str, default="64,64", help="Input image size, default is '64,64'.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")

    args = parser.parse_args()
    run_inference(args, logger)



