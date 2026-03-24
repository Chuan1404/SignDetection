import os

import numpy as np
import torchvision.transforms as transforms

from datasets_loader.asl_dataset import ASLDataset
from hand_detection import HandDetection

# Create label dictionary
data_path = 'datasets/ASL_Alphabet_Dataset/train'

# Create transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

hand_detection = HandDetection()

dataset = ASLDataset("datasets/ASL_Alphabet_Dataset/train")
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
features = []
labels = []

for i in range(len(dataset)):
    img, label = dataset[i]
    # print(dataset.label_map[label])

    result = hand_detection.detect_image(img)

    if result.hand_landmarks:
        base = result.hand_landmarks[0][0]
        for landmarks in result.hand_landmarks:
            feature = []
            for landmark in landmarks:
                feature.extend(
                    [landmark.x - base.x,
                     landmark.y - base.y,
                     landmark.z - base.z])

            features.append(feature)
            labels.append(label)

hand_detection.close()

features = np.array(features)
labels = np.array(labels)

np.save('features.npy', features)
np.save('labels.npy', labels)

