import csv
import argparse
import json
import numpy as np
import datetime
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.io import read_image

from datasets.casia_webface import CasiaWebFaceDataset
from custom_models.resnet18 import resnet18_base
from custom_models.classifiers import ArcFaceClassifier, SoftMaxClassifier
from losses.losses import arcface_loss

def main(config_file):
    # Get dictionary of configuration parameters
    if isinstance(config_file, str):
        with open(config_file) as json_file:
            config_dict = json.load(json_file)

    # Device to use (cuda or CPU)
    device = torch.device(config_dict["device"])

    # Define data transformations applied to dataset
    transform = transforms.Compose([
        transforms.Resize(config_dict['input_size']),
    ])

    # Define dataset & dataloader
    train_dataset = CasiaWebFaceDataset(config_dict["data_path"], transform=transform)
    num_classes = train_dataset.__classes__()

    # Load model
    base_model = resnet18_base(embedding_size=config_dict["embedding_size"]).to(device)
    classifier = ArcFaceClassifier(base_model, config_dict["embedding_size"], num_classes).to(device)
    classifier.load_state_dict(torch.load(config_dict["final_model_dir"]))
    classifier.eval()

    # Keep track of labels already stored
    # labels_seen = np.zeros((num_classes))

    # Create .csv database, image path in the 1st row,
    with open('database.csv', 'wb') as f:
        writer = csv.writer(f)

        # Get 1 image per label and add to database
        for root, dirs, files in enumerate(os.walk(config_dict["data_path"])):
            image_path = os.path.join(root, files[0])

            # Read image, normalize, convert to rgb if necessary, and resize
            image = read_image(image_path)
            image = image / 255.0
            if image.shape[0] == 1:
                image = image.expand(3, -1, -1)
            image = transforms.Resize(config_dict['input_size'])(image)
            image = image.to(device)

            embedding = classifier.base_model(image)

            # Save to csv
            writer.writerow([image_path, embedding])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('config_file', help='path to the configuration file (.json)')

    args = parser.parse_args()
    main(args.config_file)