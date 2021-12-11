import os
import numpy
from skimage.io import imread

import torch
from torch.utils.data import Dataset


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
        image = imread(self.data_files[idx])
        image /= 255
        image = torch.from_numpy(image)

        label = self.id_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import numpy as np

    train_data = CasiaWebFaceDataset('/home/eric/Datasets/CASIA-WebFace')

    # Create subplots, remove ticks, add titles
    N = 2
    fig, ax = plt.subplots(nrows=N, ncols=1)

    for i in range(N):
        rint = np.random.randint(train_data.__len__())
        images, labels = train_data.__getitem__(rint)

        print(type(images), images.shape)
        print(labels)
        ax[i].imshow(images, interpolation='nearest')

    fig.tight_layout()
    plt.show()