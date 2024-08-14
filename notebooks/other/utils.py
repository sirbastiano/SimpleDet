import pandas as pd 
from pathlib import Path
import numpy as np 
import os, shutil


def list_folder(folder_dir):
    return [x for x in Path(folder_dir).iterdir() if x.is_dir()]

def remove_folder(folder_dir):
    # Remove the directory and all its contents
    try:
        shutil.rmtree(folder_dir)
        print(f"Directory '{folder_dir}' and all its contents removed successfully")
    except OSError as e:
        print(f"Error: {e.strerror}")
        


if __name__ == '__main__':
    print('Not run by main...')