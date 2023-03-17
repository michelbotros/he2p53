import os
from torch.utils.data import Dataset
from torchvision.io import read_image


class VirtualStainDataset(Dataset):
    """Dataset containing HE and p53 image pairs"""

    def __init__(self, dataset_dir):
        self.he_files = [os.path.join(dataset_dir, 'HE', f) for f in os.listdir(os.path.join(dataset_dir, 'HE'))]
        self.p53_files = [os.path.join(dataset_dir, 'p53', f) for f in os.listdir(os.path.join(dataset_dir, 'p53'))]

    def __len__(self):
        return len(self.he_files)

    def __getitem__(self, idx):
        he = read_image(self.he_files[idx])
        p53 = read_image(self.p53_files[idx])
        return he, p53
