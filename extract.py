import os
import cv2
import numpy as np

from hand_detection import HandDetection

MAX_PER_CLASS = 200
hand_detection = HandDetection()
data_path = 'datasets/ASL_Alphabet_Dataset/train'

labels = os.listdir(data_path)
X = []  # sample
y = []  # labels

for label in labels:
    folder_path = os.path.join(data_path, label)
    for f in os.listdir(folder_path)[:MAX_PER_CLASS]:
        image_path = os.path.join(folder_path, f)
        image = cv2.imread(image_path)

        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hand_detection.detect_image(image_rgb)

        if result.hand_landmarks:
            base = result.hand_landmarks[0][0]
            for landmarks in result.hand_landmarks:
                feature = []
                for landmark in landmarks:
                    feature.extend(
                        [landmark.x - base.x,
                         landmark.y - base.y,
                         landmark.z - base.z])

                X.append(feature)
                y.append(label)

X = np.array(X)
y = np.array(y)

np.save('X.npy', X)
np.save('y.npy', y)

print("Saved X.npy and y.npy")
