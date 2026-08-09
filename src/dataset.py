import os
import torch
import json

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation):
        self.root = root
        self.annotation = annotation

        with open(os.path.join(self.annotation, "gloss2idx.json"), "r") as f:
            self.gloss2idx = json.load(f)
