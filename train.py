import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from datasets_loader.asl_dataset import ASLDataset
from model import SignModel

features = np.load("features.npy")
labels = np.load("labels.npy")

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)
features_train = torch.tensor(features_train, dtype=torch.float32)
features_test = torch.tensor(features_test, dtype=torch.float32)

labels_train = torch.tensor(labels_train, dtype=torch.long)
labels_test = torch.tensor(labels_test, dtype=torch.long)

dataset = ASLDataset("datasets/ASL_Alphabet_Dataset/train")
model = SignModel(num_classes=len(dataset.label_map))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(51):
    model.train()

    outputs = model(features_train)
    loss = criterion(outputs, labels_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

model.eval()

with torch.no_grad():
    outputs = model(features_test)
    _, predicted = torch.max(outputs, 1)


    correct = (predicted == labels_test).sum().item()
    total = labels_test.size(0)

    accuracy = correct / total
    print(f"Correct: {correct}/{total}")
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

torch.save({
    "model_state": model.state_dict(),
    "num_classes": len(dataset.label_map)
}, "model.pth")
#