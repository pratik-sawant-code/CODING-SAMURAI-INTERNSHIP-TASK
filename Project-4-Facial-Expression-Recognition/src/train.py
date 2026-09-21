import pandas as pd
import numpy as np
import importlib

torch = importlib.import_module("torch")
nn = importlib.import_module("torch.nn")
TensorDataset = importlib.import_module("torch.utils.data").TensorDataset
DataLoader = importlib.import_module("torch.utils.data").DataLoader

from model import EmotionCNN


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load dataset
df = pd.read_csv("data/fer2013.csv")

print("\nDataset loaded!")
print("Dataset shape:", df.shape)


# Convert pixels to arrays
X = np.array([
    np.fromstring(pixels, dtype=np.uint8, sep=" ")
    for pixels in df["pixels"]
])

X = X.reshape(-1, 1, 48, 48)
X = X.astype(np.float32) / 255.0

y = df["emotion"].values


# Dataset split
train_mask = df["Usage"] == "Training"
val_mask = df["Usage"] == "PublicTest"
test_mask = df["Usage"] == "PrivateTest"

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]


print("\nTraining:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# Convert to tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val, dtype=torch.long)

X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test, dtype=torch.long)


# DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# Create model
model = EmotionCNN().to(device)

print("\nCNN model created!")


# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training settings
epochs = 20

best_val_accuracy = 0.0


print("\n========== TRAINING STARTED ==========\n")


for epoch in range(epochs):

    # -------------------------
    # Training
    # -------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total


    average_loss = running_loss / len(train_loader)


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.4f} "
        f"Training Accuracy: {train_accuracy:.2f}% "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "emotion_cnn_best.pth"
        )

        print(
            f"Best model saved! "
            f"Validation Accuracy: {best_val_accuracy:.2f}%"
        )


# -------------------------
# Test Accuracy
# -------------------------

print("\n========== TESTING MODEL ==========\n")


model.load_state_dict(
    torch.load(
        "emotion_cnn_best.pth",
        map_location=device
    )
)

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        test_total += labels.size(0)

        test_correct += (predicted == labels).sum().item()


test_accuracy = 100 * test_correct / test_total


print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
print(f"Final Test Accuracy: {test_accuracy:.2f}%")


# Save final model
torch.save(
    model.state_dict(),
    "emotion_cnn.pth"
)


print("\n====================================")
print("TRAINING COMPLETED!")
print("====================================")
print("Model saved as: emotion_cnn.pth")
print(f"Test Accuracy: {test_accuracy:.2f}%")