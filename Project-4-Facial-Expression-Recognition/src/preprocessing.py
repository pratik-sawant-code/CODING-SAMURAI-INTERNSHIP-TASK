import pandas as pd
import numpy as np



df = pd.read_csv("data/fer2013.csv")

print("Dataset loaded!")
print("Original shape:", df.shape)




X = np.array(
    [np.fromstring(pixels, dtype=np.uint8, sep=" ")
     for pixels in df["pixels"]]
)

X = X.reshape(-1, 48, 48, 1)

y = df["emotion"].values




X = X.astype("float32") / 255.0




X_train = X[df["Usage"] == "Training"]
y_train = y[df["Usage"] == "Training"]

X_validation = X[df["Usage"] == "PublicTest"]
y_validation = y[df["Usage"] == "PublicTest"]

X_test = X[df["Usage"] == "PrivateTest"]
y_test = y[df["Usage"] == "PrivateTest"]




print("\nPreprocessing completed!")

print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)

print("Validation images:", X_validation.shape)
print("Validation labels:", y_validation.shape)

print("Test images:", X_test.shape)
print("Test labels:", y_test.shape)




print("\nPixel minimum:", X_train.min())
print("Pixel maximum:", X_train.max())