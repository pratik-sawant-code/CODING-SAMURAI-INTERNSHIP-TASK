import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/fer2013.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

print("\nEmotion Distribution:")
print(df["emotion"].value_counts())

emotion_names = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

emotion_counts = df["emotion"].value_counts().sort_index()

emotion_counts.index = [
    emotion_names[i] for i in emotion_counts.index
]

plt.figure(figsize=(10, 5))
emotion_counts.plot(kind="bar")

plt.title("Facial Expression Distribution")
plt.xlabel("Emotion")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()




pixels = df.iloc[0]["pixels"]

image = np.array(pixels.split(), dtype="uint8")

image = image.reshape(48, 48)

plt.figure(figsize=(4, 4))
plt.imshow(image, cmap="gray")
plt.axis("off")

emotion = emotion_names[df.iloc[0]["emotion"]]

plt.title(f"Emotion: {emotion}")
plt.show()