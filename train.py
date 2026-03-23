import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import pickle

from model import SignModel

X = np.load("X.npy")
y = np.load("y.npy")

le = LabelEncoder()
y = le.fit_transform(y)
pickle.dump(le, open("label_encoder.pkl", "wb"))
y_unique = np.unique(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

model = SignModel(num_classes=len(set(y_unique)))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(51):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

model.eval()

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)


    correct = (predicted == y_test).sum().item()
    total = y_test.size(0)

    accuracy = correct / total
    print(f"Correct: {correct}/{total}")
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

torch.save({
    "model_state": model.state_dict(),
    "num_classes": len(y_unique)
}, "model.pth")
