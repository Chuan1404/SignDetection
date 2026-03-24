import os
import cv2
import torch
from torch.utils.data import Dataset

# MAX_PER_CLASS = 300

class ASLDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = []
        self.labels = []
        self.label_map = {}
        self.transform = transform

        for idx, label in enumerate(sorted(os.listdir(data_path))):
            self.label_map[idx] = label
            folder = os.path.join(data_path, label)

            for file in os.listdir(folder):
                self.data.append(os.path.join(folder, file))
                self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        # else:
        #     img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        label = self.labels[idx]
        return img, label

