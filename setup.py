from setuptools import setup, find_packages
import os
import subprocess
import platform

def is_linux():
    print("Platform system: ",platform.system())
    return platform.system() == 'Linux'

if is_linux():
    print("The system is running on Linux.")
    system = 0
    
else:
    print("The system is not running on Linux.")
    system = 1

def install_dependencies():
    torch_dependencies = "https://download.pytorch.org/whl/torch_stable.html"

    if system == 0:
        subprocess.check_call(["pip", "install", "torch==2.0.0+cu118", "torchvision==0.15.1+cu118", "-f", torch_dependencies])
    else:
        subprocess.check_call(["pip", "install", "torch==2.0.0", "torchvision==0.15.1", "-f", torch_dependencies])
        
    subprocess.check_call(["pip", "install", "SciencePlots"])
    subprocess.check_call(["pip", "install", "-U", "openmim"])
    subprocess.check_call(["mim", "install", "mmengine"])
    subprocess.check_call(["mim", "install", "mmcv==2.1.0"])
    lista_packgs = [
              'h5py',
              'tifffile',
              'netcdf4',
              'h5netcdf',
              'rasterio',
              'numpy',
              'scikit-image',
              'scipy',
              'scikit-learn',
              'geopandas',
              'pandas>=1.4, <2',
              'matplotlib',
              'wandb',
              'seaborn',
              'tqdm',
              'tzlocal',
              'regex',]
    subprocess.check_call(["mim", "install"] + lista_packgs)
    subprocess.check_call(["git", "clone", "https://github.com/open-mmlab/mmdetection.git"])
    os.chdir("./mmdetection")
    subprocess.check_call(["pip", "install", "-v", "-e", "."])

install_dependencies()

setup(
    name='openmmlab',
    version='0.1',
    description='MMDET',
    author='Roberto Del Prete',
    author_email='roberto.delprete@ext.esa.int',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)