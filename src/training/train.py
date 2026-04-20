# import numpy as np
# from sklearn.model_selection import train_test_split
# import torch
# from torch import nn
#
# from src.data.asl_dataset import ASLDataset
# from models.model import SignModel
#
# features = np.load("features.npy")
# labels = np.load("../../labels.npy")
#
# features_train, features_test, labels_train, labels_test = train_test_split(
#     features, labels, test_size=0.2, random_state=42
# )
# features_train = torch.tensor(features_train, dtype=torch.float32)
# features_test = torch.tensor(features_test, dtype=torch.float32)
#
# labels_train = torch.tensor(labels_train, dtype=torch.long)
# labels_test = torch.tensor(labels_test, dtype=torch.long)
#
# dataset = ASLDataset("C:\\Users\\ADMIN\\OneDrive\\Desktop\\SignDetection\\datasets\\ASL_Alphabet_Dataset\\asl_alphabet_train")
# model = SignModel(num_classes=len(dataset.label_map))
#
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# for epoch in range(51):
#     model.train()
#
#     outputs = model(features_train)
#     loss = criterion(outputs, labels_train)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if epoch % 5 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item()}")
#
# model.eval()
#
# with torch.no_grad():
#     outputs = model(features_test)
#     _, predicted = torch.max(outputs, 1)
#
#
#     correct = (predicted == labels_test).sum().item()
#     total = labels_test.size(0)
#
#     accuracy = correct / total
#     print(f"Correct: {correct}/{total}")
#     print(f"Test Accuracy: {100 * correct / total:.2f}%")
#
# torch.save({
#     "model_state": model.state_dict(),
#     "num_classes": len(dataset.label_map)
# }, "model.pth")
# #
#
# def normalize_landmarks(landmarks):
#     """
#     landmarks: numpy array of shape (21, 3) [x, y, z]
#     returns: flattened, normalized landmarks relative to wrist
#     """
#     wrist = landmarks[0]
#     landmarks_rel = landmarks - wrist  # relative to wrist
#
#     # Optionally normalize to unit length
#     max_val = np.max(np.abs(landmarks_rel))
#     if max_val > 0:
#         landmarks_rel /= max_val
#
#     return landmarks_rel.flatten()
#
# def compute_angles(landmarks):
#     """
#     landmarks: numpy array of shape (21, 3) [x, y, z]
#     returns: array of angles (in radians) between consecutive bones
#     """
#     angles = []
#     # Define bones as pairs (start, end)
#     bones = [
#         (0,1),(1,2),(2,3),(3,4),    # Thumb
#         (0,5),(5,6),(6,7),(7,8),    # Index
#         (0,9),(9,10),(10,11),(11,12), # Middle
#         (0,13),(13,14),(14,15),(15,16), # Ring
#         (0,17),(17,18),(18,19),(19,20)  # Pinky
#     ]
#
#     for i in range(len(bones)-1):
#         start1, end1 = bones[i]
#         start2, end2 = bones[i+1]
#
#         vec1 = landmarks[end1] - landmarks[start1]
#         vec2 = landmarks[end2] - landmarks[start2]
#
#         # Normalize vectors
#         vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
#         vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
#
#         # Angle using dot product
#         angle = np.arccos(np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0))
#         angles.append(angle)
#
#     return np.array(angles)
#
# def compute_distances(landmarks, pairs=None):
#     """
#     landmarks: numpy array of shape (21, 3) [x, y, z]
#     pairs: list of tuples specifying which landmarks to compute distance between
#     returns: array of distances
#     """
#     if pairs is None:
#         # Default: distance from wrist (0) to all other landmarks
#         pairs = [(0, i) for i in range(1, 21)]
#
#     distances = []
#     for i,j in pairs:
#         dist = np.linalg.norm(landmarks[i] - landmarks[j])
#         distances.append(dist)
#
#     return np.array(distances)