from zipfile import ZipFile
import pathlib
from urllib.request import urlretrieve
import os
from os import remove
import os.path
import numpy as np
import Augmentor as aug
import pandas as pd
import shutil as sh
import torch
import glob
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

FID_DIRECTORY = pathlib.Path("FID-300")
FID_LABELS = FID_DIRECTORY / "label_table.csv"
FID_SOURCE_URL = "https://fid.dmi.unibas.ch/FID-300.zip"
TRAIN_DIR = "FID-300/tracks_cropped/"
NUM_EPOCHS = 10


def process_images(valid_size=0.2):
    transformations = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(TRAIN_DIR, transform=transformations)
    test_data = datasets.ImageFolder(TRAIN_DIR, transform=transformations)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler, batch_size=64)
    return trainloader, testloader


def organize_files(label_df):
    """ Moves all pngs in tracked_cropped into subfolders by label (for PyTorch Image Folder """
    train_dir = pathlib.Path(TRAIN_DIR)

    files = glob.glob(str(train_dir / "*.jpg"))
    for i in range(len(files)):
        f = pathlib.Path(files[i])
        fname = f.name
        id = int(f.stem)
        label = label_df["label"].iloc[id-1]
        new_dir = train_dir / str(label)
        new_dir.mkdir(exist_ok=True)
        new_file = new_dir / fname

        sh.move(f, new_file)


def load_labels():
    labels = pd.read_csv(FID_LABELS, delimiter=",", header=None,
                         dtype=np.dtype(int), names=['id', 'label'])
    return labels


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


def run_nn(train_load, test_load, device, model):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)

    steps = 0
    running_loss = 0
    print_every = 10
    train_losses = []
    test_losses = []
    for epoch in range(NUM_EPOCHS):
        for inputs, labels in train_load:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_load:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss / len(train_load))
                test_losses.append(test_loss / len(test_load))
                print(f"Epoch {epoch + 1}/{NUM_EPOCHS}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(test_load):.3f}.. "
                      f"Test accuracy: {accuracy / len(test_load):.3f}")
                running_loss = 0
                model.train()
    torch.save(model, 'aerialmodel.pth')


def preprocess_data():
    fetch_FID_300_data()
    labels = load_labels()
    organize_files(labels)
    train_load, test_load = process_images()
    return train_load, test_load


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = models.resnet50(pretrained=True)
    return device, model


if __name__ == '__main__':
    train_load, test_load = preprocess_data()
    device, model = load_model()
    run_nn(train_load, test_load, device, model)
