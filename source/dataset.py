import os
import torch
import numpy as np
from PIL.Image import Image
from tqdm import tqdm
from torch.utils.data import Dataset


class Traubenwelke(Dataset):
    def __init__(self, samples: list[Image], labels: list[int], transform=None):
        super(Traubenwelke, self).__init__()
        self.samples: list[Image] = samples
        self.labels: list[int] = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index: int):
        image: Image = self.samples[index]
        label: int = self.labels[index]

        image = image.convert("RGB")

        if self.transform:
            image_transform = self.transform(image)
        else:
            image_transform = None

        torch_image: torch.Tensor = torch.tensor(np.asarray(image))
        torch_label: torch.Tensor = torch.tensor(label).type(dtype=torch.long)
        torch_image_transform = torch.tensor(np.asarray(image_transform))

        return torch_image_transform, torch_label#, torch_image