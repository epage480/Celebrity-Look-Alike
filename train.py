import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
print(torchvision.__version__)

from datasets.casia_webface import CasiaWebFaceDataset
from custom_models.resnet18 import resnet18




def train_epoch(model, device, train_loader, optimizer, epoch, verbose=False):
    log_interval = 10

    # Set model to train mode (enables dropout etc.)
    model.train()

    # For each batch,
    # 1. Move data to GPU
    # 2. Zero gradients
    # 3. Generate predictions (forward pass)
    # 4. Calculate loss
    # 5. Propogate loss backwards
    # 6. Run Optimizer (step)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if verbose:
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

# Calculate test set accuracy & loss
def test(model, device, test_loader):
    # Set model to eval mode (disables dropout etc.)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main(config_file):

    # Get dictionary of configuration parameters
    if isinstance(config_file, str):
        with open(config_file) as json_file:
            config_dict = json.load(json_file)

    # Device to use (cuda or CPU)
    device = torch.device(config_dict["device"])

    # Set keyword arguments for data loaders
    train_kwargs = {'batch_size': config_dict["batch_size"]}
    if device == "cuda":
        cuda_kwargs = {'num_workers': config_dict["num_workers"],
                       'pin_memory': config_dict["pin_memory"],
                       'shuffle': config_dict["shuffle"]}
        train_kwargs.update(cuda_kwargs)

    # Define data transformations applied to dataset
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config_dict['input_size']),
        transforms.ToTensor()
    ])

    train_dataset = CasiaWebFaceDataset(config_dict['data_path'])
    num_classes = train_dataset.__classes__()

    # Define model & adjust final layer to
    model = resnet18(pretrained=False)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('config_file', help='path to the configuration file (.json)')

    args = parser.parse_args()
    main(args.config_file)