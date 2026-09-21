import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader  # type: ignore[import-not-found]
from sklearn.metrics import confusion_matrix  # type: ignore[import-not-found]

from model import EmotionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

df = pd.read_csv("data/fer2013.csv")

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
    torch.load(
        "emotion_cnn.pth",
        map_location=device
    )
)

model.eval()

print("Model loaded successfully!")

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

cm = confusion_matrix(
    all_labels,
    all_predictions
)

emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# cspell:ignore figsize imshow xlabel ylabel xticks yticks colorbar
plt.figure(figsize=(9, 7))

plt.imshow(cm)

plt.title("Facial Expression Recognition - Confusion Matrix")

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

plt.xticks(
    range(7),
    emotion_names,
    rotation=45
)

plt.yticks(
    range(7),
    emotion_names
)

for i in range(7):

    for j in range(7):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

print("Confusion matrix saved!")

plt.show()