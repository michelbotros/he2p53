import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
import torch
from skimage.color import rgb2hed
from torch.utils.data import DataLoader, random_split


class VirtualStainDataset(Dataset):
    """Dataset containing HE and p53 image pairs"""

    def __init__(self, dataset_dir):
        self.he_files = [os.path.join(dataset_dir, 'HE', f) for f in os.listdir(os.path.join(dataset_dir, 'HE'))]
        self.p53_files = [os.path.join(dataset_dir, 'p53', f) for f in os.listdir(os.path.join(dataset_dir, 'p53'))]

    def __len__(self):
        return len(self.he_files)

    def __getitem__(self, idx):

        # read in HE and p53
        he = read_image(self.he_files[idx]).to(torch.float)
        p53 = read_image(self.p53_files[idx]).to(torch.float)

        # compute staining percentage from p53
        p53_rgb = np.transpose(p53.numpy(), (1, 2, 0)).astype(np.uint8)
        dab_density = torch.tensor(DAB_density_norm(p53_rgb), dtype=torch.float32)

        return he, p53, dab_density


def DAB_density_norm(ihc_rgb):
    """ Convert RGB colour space to Haematoxylin-Eosin-DAB (HED) colour space and calculate the ratio of DAB to HE staining.

    Args:
        ihc_rgb: p53 image patch (torch.Tensor) [C, H, W]

    Returns
        dab_density: the ratio of  DAB to Haematoxyl staining (float)

    """
    # convert to hed-space and get the dab channel
    ihc_hed = rgb2hed(ihc_rgb)
    h_channel = ihc_hed[:, :, 0]
    d_channel = ihc_hed[:, :, 2]

    # compute ratio d-to-h staining ratio
    dab_density = np.sum(d_channel) / np.sum(h_channel)

    return dab_density


def get_dataloaders(dataset_dir, batch_size, num_workers=16):
    """ Returns the dataloaders for the experiment
    """
    # ToDo: properly split on patient-level

    # load dataset
    dataset = VirtualStainDataset(dataset_dir)
    print('Dataset contains: {} pairs.'.format(dataset.__len__()))

    # split into train, val, test: 0.8, 0.1, 0.1
    proportions = [.8, .1, .1]
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])

    train_set, val_set, test_set = random_split(dataset, lengths)
    print('Train pairs: {}'.format(train_set.__len__()))
    print('Val pairs: {}'.format(val_set.__len__()))
    print('Test pairs: {}'.format(test_set.__len__()))

    # initialize data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return train_loader, val_loader, test_loader