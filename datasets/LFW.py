import PIL.Image
import os

import numpy as np
from torch.utils.data import Dataset



class LFW_pairs(Dataset):
    """
    Dataset class for Labeled Faces in the Wild Dataset
    Found here: https://www.kaggle.com/atulanandjha/lfwpeople
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        if train:
            self.pairs_path = os.path.join(root, 'pairsDevTrain.csv')
        else:
            self.pairs_path = os.path.join(root, 'pairsDevTest.csv')
        self.pairs = self._read_pairs()
        self.train = train
        self.transform = transform

    def _read_pairs(self):
        """Get pairs"""
        pairs = []
        with open(self.pairs_path, 'r') as f:
            for line in f.readlines()[1:]:  # skip header
                pair = line.strip().split()
                pairs.append(pair)

        path_list = []

        for pair in pairs:
            if len(pair) == 3:
                path0 = os.path.join(self.root, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.jpg')
                id0 = pair[0]
                path1 = os.path.join(self.root, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.jpg')
                id1 = pair[0]
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(self.root, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.jpg')
                id0 = pair[0]
                path1 = os.path.join(self.root, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.jpg')
                id1 = pair[0]
                issame = False

            path_list.append((path0, path1, issame, id0, id1))

        return path_list

    def _load_img(self, img_path):
        """Loads an image from dist, then performs face alignment and applies transform"""
        img = PIL.Image.open(img_path)

        if self.transform is None:
            return img

        return self.transform(img)

    def __getitem__(self, index):
        """Returns a pair of images and similarity flag by index"""
        (path_1, path_2, is_same, id0, id1) = self.pairs[index]
        img1, img2 = self._load_img(path_1), self._load_img(path_2)

        return {'img1': img1, 'img2': img2, 'is_same': is_same, 'id0': id0, 'id1': id1}

    def __len__(self):
        """Returns total number of pairs"""
        return len(self.pairs)

if __name__ == "__main__":
    train_data = LFW_pairs(train=True)

    # Create subplots, remove ticks, add titles
    N = 2
    fig, ax = plt.subplots(nrows=N, ncols=1)

    for i in range(N):
        rint = np.random.randint(10000)
        images, labels = train_data.__getitem__(rint)

        print(type(images), images.shape)
        ax[i].imshow(images, cmap='gray', interpolation='nearest')

    fig.tight_layout()
    plt.show()

# class LFWDataset(data.Dataset):
#     """
#         Dataset class for Labeled Faces in the Wild Dataset
#     """
#
#     def __init__(self, path_list, issame_list, transforms, train=True):
#         '''
#             Parameters
#             ----------
#             path_list    -   List of full path-names to LFW images
#         '''
#         self.files = collections.defaultdict(list)
#         self.split = split
#         if train:
#             self.files['train'] =  path_list
#         else:
#             self.files['test'] = path_list
#         self.pair_label = issame_list
#         self.transforms = transforms
#
#     def __len__(self):
#         return len(self.files[self.split])
#
#     def __getitem__(self, index):
#         img_file = self.files[self.split][index]
#         img = PIL.Image.open(img_file)
#         im_out = self.transforms(img)
#         return im_out