import importlib.util

import pandas as pd
import numpy as np

from model import EmotionCNN



torch = None
TensorDataset = None
DataLoader = None
classification_report = None
confusion_matrix = None

if importlib.util.find_spec("torch") is not None:
    import torch
    from torch.utils.data import TensorDataset, DataLoader

if importlib.util.find_spec("sklearn") is not None:
    from sklearn.metrics import classification_report, confusion_matrix


if torch is None:
    raise ImportError("PyTorch is required to run evaluation.")

if classification_report is None or confusion_matrix is None:
    raise ImportError("scikit-learn is required to run evaluation.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)




df = pd.read_csv("data/fer2013.csv")

print("Dataset loaded!")




X = np.array([
    np.fromstring(pixels, dtype=np.uint8, sep=" ")
    for pixels in df["pixels"]
])

X = X.reshape(-1, 1, 48, 48)

X = X.astype(np.float32) / 255.0

y = df["emotion"].values




test_mask = df["Usage"] == "PrivateTest"

X_test = X[test_mask]
y_test = y[test_mask]


print("Test images:", X_test.shape[0])



X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test, dtype=torch.long)


test_dataset = TensorDataset(X_test, y_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)




model = EmotionCNN().to(device)

model.load_state_dict(
    torch.load("emotion_cnn.pth", map_location=device)
)

model.eval()

print("Trained model loaded!")




correct = 0
total = 0

all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


accuracy = 100 * correct / total

print("\n================================")
print(f"Test Accuracy: {accuracy:.2f}%")
print("================================")



emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=emotion_names
    )
)



cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:\n")
print(cm)