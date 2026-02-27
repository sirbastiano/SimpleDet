# DetectionProject


## Introduction

This project is part of the collaboration with ESA Phi-Lab, and deals with Object Detection.
The main goal is to detect objects in satellite images, and to provide a bounding box around them.

### Object Detection

Object detection is a computer vision technique that allows to detect objects in images or videos. It is a combination of two tasks: object classification and object localization. The goal is to not only classify the object, but also to locate it in the image.

## Custom MMDetection

MMDetection is an open-source object detection toolbox based on PyTorch. It is a part of the OpenMMLab project developed by Multimedia Laboratory, CUHK. The toolbox is widely used in research and industry for object detection tasks. It provides a large number of pre-trained models, and allows to train custom models on custom datasets.

This repository contains a custom implementation of MMDetection, with custom datasets and a custom models support. It extends the original MMDetection toolbox, and allows to rapidly test and propotype new models on new datasets.

## Installation

> [!WARNING]  
> To install the project on Linux, you will need to have Conda installed. If you don't have Conda installed, please follow the official Conda installation guide for your operating system. Once you have Conda installed, you can proceed with the installation steps mentioned above.

1. Clone the repository:

```bash
git clone https://github.com/sirbastiano/MDet.git
```

2. Install the required dependencies:

```bash
python -m pip install .
```

For an installation with optional OpenMMLab runtime packages:

```bash
python -m pip install ".[openmmlab]"
```



1. Clone the repository:

```bash
git clone https://github.com/sirbastiano/MDet.git
```

2. Install the requirements:

```bash
python -m pip install ".[openmmlab]"
```

## Packaging and publishing

You can also install this package from PyPI after it is published:

```bash
python -m pip install simpledet
```

Release workflow with Makefile:

```bash
make install           # install dependencies from project metadata
make build             # build dist/*.whl and dist/*.tar.gz
make check             # validate distributions
make publish           # upload to PyPI
make publish-test      # upload to TestPyPI
```

> `make publish` and `make publish-test` require Twine credentials (or token in environment variables).



Once installed, you can start using the custom MMDetection toolbox for object detection tasks:

- **Train a model:** Follow the instructions provided in the documentation to train a custom model on your dataset.
- **Test a model:** Use pre-trained models or your own trained models to detect objects in new satellite images.
- **Visualize results:** The toolbox includes utilities to visualize the bounding boxes and classifications directly on the images.

For detailed usage instructions, refer to the [documentation](docs/README.md) included in this repository.

## Contributing

Contributions to this project are welcome! If you have ideas for improvements or new features, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m "Description of changes"`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

Please ensure that your code adheres to the coding standards outlined in the [Contributing Guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE). Please review the license file for details.

## Notes

### Important Note

- This project is still under active development. Features and APIs are subject to change.
- Ensure you are using a compatible version of Python (3.8 or later) and PyTorch.
- The project is designed to run on Linux; compatibility with other operating systems has not been extensively tested.
