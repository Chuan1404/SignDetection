import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion import MultiStreamFusion
from models.LLM import LLMDecoder
from models.feature_extractor import CNNFeatureExtractor
from models.finger_recognition import FingerspellingEncoder
from models.lip_recognition import LipreadingEncoder
from models.sign_recognition import SignEncoder


class SignModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(63, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)  # no activation here, use CrossEntropyLoss
        return x

class MultiStreamModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.sign_enc = SignEncoder()
        self.finger_enc = FingerspellingEncoder()
        self.lip_enc = LipreadingEncoder()

        self.fusion = MultiStreamFusion()
        self.decoder = LLMDecoder()

    def forward(self, sign_x, finger_x, lip_x, tgt_seq):
        # sign_x: [B, T, 512]

        sign_feat = self.sign_enc(sign_x)
        finger_feat = self.finger_enc(finger_x)
        lip_feat = self.lip_enc(lip_x)

        fused = self.fusion(sign_feat, finger_feat, lip_feat)

        out = self.decoder(fused, tgt_seq)
        return out

