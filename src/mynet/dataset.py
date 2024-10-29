# Resources that were used for documentation include pytorch.org, matplotlib.org and LLMs

from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import pandas as pd

class CustomMNISTDataset(Dataset):
    def __init__(self, root: str, subset: str, transformation=None):
        self.root = root
        self.subset = subset
        self.transformation = transformation
        self.dataframe = pd.read_csv(os.path.join(root, f"{subset}.csv"))

    def __getitem__(self, idx):
        img_file, label = self.dataframe.iloc[idx]['filename'], int(self.dataframe.iloc[idx]['label'])

        img = Image.open(os.path.join(self.root, self.subset, img_file))

        img = self.transformation(img)

        return img, label

    def __len__(self):
        return len(self.dataframe)
