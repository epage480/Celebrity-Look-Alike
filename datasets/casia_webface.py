import os
import numpy
from skimage.io import imread

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


"""
Dataset object for the cassia webface dataset
"""

class CasiaWebFaceDataset(Dataset):
    """Class representing the CASIA WebFace dataset"""

    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: str The path to the data
        :param transform: `torchvision.transforms` The transform(s) to apply to the face images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_files = []
        self.id_labels = []

        for label, (root, dirs, files) in enumerate(os.walk(self.root_dir)):
            for name in files:
                self.data_files.append(os.path.join(root, name))
                self.id_labels.append(label)

    def __len__(self):
        return len(self.data_files)

    def __classes__(self):
        """Returns # of labels in dataset"""
        return len(self.id_labels)

    def __getitem__(self, idx):
        """Returns a single image & label pair"""
        image = read_image(self.data_files[idx])
        image = image / 255.0
        # image = torch.from_numpy(image)

        label = self.id_labels[idx]

        # Converts 1 channel b&w to rgb
        if image.shape[0] == 1:
            image = image.expand(3, -1, -1)

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import numpy as np

    print("initializing dataset")
    train_data = CasiaWebFaceDataset('/home/eric/Datasets/CASIA-WebFace')
    print("done initializing")

    # Create subplots, remove ticks, add titles
    N = 10000

    # Checking for unuaually shaped images
    for i in range(N):
        # print("i:", i)
        rint = np.random.randint(train_data.__len__())
        images, labels = train_data.__getitem__(rint)
        if images.shape[0] != 3:
            print("rint:", rint, images.shape)
            fig, ax = plt.subplots(nrows=1, ncols=1)
            images = images.expand(3, -1, -1)
            print(images.shape)
            images = np.moveaxis(np.array(images), 0, -1)
            ax.imshow(images, interpolation='nearest')
            plt.show()

        # print(images.shape)

    # # Display N random images
    # fig, ax = plt.subplots(nrows=N, ncols=1)
    # for i in range(N):
    #     rint = np.random.randint(train_data.__len__())
    #     images, labels = train_data.__getitem__(rint)
    #
    #     images = np.moveaxis(np.array(images), 0, -1)
    #
    #     print(type(images), images.shape)
    #     print(labels)
    #     ax[i].imshow(images, interpolation='nearest')
    #
    # fig.tight_layout()
    # plt.show()
