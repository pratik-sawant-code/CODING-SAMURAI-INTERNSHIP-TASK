import torch
from PIL import Image

from model import EmotionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

model = EmotionCNN().to(device)

model.load_state_dict(
    torch.load(
        "emotion_cnn.pth",
        map_location=device
    )
)

model.eval()

image_path = "images/test_face.jpg"

image = Image.open(image_path).convert("L").resize((48, 48))

image_tensor = torch.frombuffer(
    bytearray(image.tobytes()),
    dtype=torch.uint8
).to(torch.float32).reshape(1, 48, 48) / 255.0
image_tensor = image_tensor.unsqueeze(0).to(device)

with torch.no_grad():

    output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )

emotion = emotion_names[prediction.item()]
confidence = confidence.item() * 100

print("\n==============================")
print("FACIAL EXPRESSION RESULT")
print("==============================")
print("Predicted Emotion:", emotion)
print(f"Confidence: {confidence:.2f}%")
print("==============================")