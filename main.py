from zipfile import ZipFile
import pathlib
from urllib.request import urlretrieve
from os import remove 
import numpy as np


FID_DIRECTORY = pathlib.Path("FID-300")
FID_LABELS = FID_DIRECTORY / "label_table.csv"
FID_SOURCE_URL = "https://fid.dmi.unibas.ch/FID-300.zip"

def fetch_FID_300_data():
    """Downloads and extracts FID-300 data to a local folder"""
    if FID_DIRECTORY.exists():
        print("FID-300 Database already exists")
        return
    print("Downloading FID_300")
    local_file, _ = urlretrieve(FID_SOURCE_URL)
    with ZipFile(local_file) as archive:
        print("Extracting FID-300")
        archive.extractall()
    remove(local_file)

def load_labels():
    return np.loadtxt(FID_LABELS, delimiter=",")


if __name__ == '__main__':
    fetch_FID_300_data()
    labels = load_labels()
    print(labels)
