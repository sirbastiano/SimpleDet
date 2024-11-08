# SimpleDet

> **Warning**: The full code and models will be released upon the pubblication of the final paper. Please refer to the arxive: https://arxiv.org/pdf/2411.03403.



![GitHub stars](https://img.shields.io/github/stars/sirbastiano/SimpleDet.svg)
![GitHub forks](https://img.shields.io/github/forks/sirbastiano/SimpleDet.svg)
![GitHub issues](https://img.shields.io/github/issues/sirbastiano/SimpleDet.svg)
![GitHub license](https://img.shields.io/github/license/sirbastiano/SimpleDet.svg)
![GitHub pull requests](https://img.shields.io/github/issues-pr/sirbastiano/SimpleDet.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/sirbastiano/SimpleDet.svg)
![GitHub code size](https://img.shields.io/github/languages/code-size/sirbastiano/SimpleDet.svg)
![GitHub language count](https://img.shields.io/github/languages/count/sirbastiano/SimpleDet.svg)
![GitHub top language](https://img.shields.io/github/languages/top/sirbastiano/SimpleDet.svg)
![GitHub repo size](https://img.shields.io/github/repo-size/sirbastiano/SimpleDet.svg)
![GitHub contributors](https://img.shields.io/github/contributors/sirbastiano/SimpleDet.svg)
![Open issues percentage](https://img.shields.io/github/issues/detail/state/sirbastiano/SimpleDet/1.svg)

<p align="center">
    <img src="SimpleDet.jpg?raw=true" alt="Logo" width="100%">
</p>


## Introduction

This project is part of a collaboration with ESA Phi-Lab, focusing on Object Detection in satellite imagery. The primary goal is to detect objects in satellite images and provide bounding boxes around them for precise localization.


## Data

• The VDS2Raw (v2) dataset, which is integral to our research, is publicly accessible for further exploration and utilization. Researchers and practitioners can download the dataset from the following link: 10.5281/zenodo.13889073.

• The VDVRaw dataset is publicly accessible for further exploration and utilization. The dataset is accessible at the following link: 10.5281/zenodo.13897485.

• Both classification datasets are publicly accessible for further exploration and utilization at the following link: 10.5281/zenodo.14007820.


### Object Detection

Object detection is a computer vision technique that identifies and locates objects within images or videos. It combines two key tasks: object classification and object localization. The goal is not only to classify the object but also to determine its position in the image.

## Custom MMDetection

MMDetection is an open-source object detection toolbox based on PyTorch. It is part of the OpenMMLab project developed by the Multimedia Laboratory at CUHK. The toolbox is widely used in both research and industry for object detection tasks, providing a wide variaety of pre-trained models and enabling users to train custom models on their own datasets.

This repository contains a simplified API to MMDetection, including custom modules such as an optical modeling system for SNR (Signal-to-Noise Ratio) and MTF (Modulation Transfer Function), with support for custom datasets and models. It extends the original MMDetection toolbox, allowing rapid testing and prototyping of new models on new datasets with ease.

## Installation

> **Warning**: To install the project on Linux, you will need Conda installed. If Conda is not already installed, please follow the official Conda installation guide for your operating system. Once Conda is installed, proceed with the steps below.

1. Clone the repository:

    ```bash
    git clone https://github.com/sirbastiano/SimpleDet.git
    ```

2. Install the required dependencies:

    ```bash
    source setup.sh
    ```

Once installed, you can start using the custom MMDetection toolbox for object detection tasks:

- **Train a model:** Follow the instructions in the documentation to train a custom model on your dataset.
- **Test a model:** Use pre-trained models or your own trained models to detect objects in new satellite images.
- **Visualize results:** The toolbox includes utilities to visualize bounding boxes and classifications directly on the images.

For detailed usage instructions, refer to the [documentation](docs/README.md) included in this repository.

## Contributing

Contributions to this project are highly encouraged! If you have suggestions for improvements or new features, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Implement your changes.
4. Commit your changes (`git commit -m "Description of changes"`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

Please ensure that your code follows the coding standards outlined in the [Contributing Guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE). Please review the license file for details.

## Notes

### Important Note

- This project is still under active development. Features and APIs are subject to change.
- Ensure you are using a compatible version of Python (3.8 or later) and PyTorch.
- The project is designed to run on Linux, and compatibility with other operating systems has not been extensively tested.

