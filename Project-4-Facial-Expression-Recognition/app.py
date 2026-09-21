import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
import sys
import numpy as np

sys.path.append("src")

from model import EmotionCNN


st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="🧠",
    layout="centered"
)

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

MODEL_PATH = "emotion_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    model = EmotionCNN()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model


def detect_face(image):
    image_array = np.array(image)

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        return None

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    padding = int(0.15 * max(w, h))

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_array.shape[1], x + w + padding)
    y2 = min(image_array.shape[0], y + h + padding)

    face = image_array[y1:y2, x1:x2]

    return Image.fromarray(face)


def predict_emotion(face_image, model):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image_tensor = transform(face_image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)[0]

    predicted_index = torch.argmax(probabilities).item()

    return (
        EMOTIONS[predicted_index],
        probabilities.cpu().numpy()
    )


st.title("🧠 Facial Expression Recognition")

st.write(
    "Upload a face image and the AI model will predict the facial expression."
)


try:
    model = load_model()
    st.success("Model loaded successfully.")

except Exception as e:
    st.error("Could not load the trained model.")
    st.code(str(e))
    st.stop()


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Detect Emotion"):

        face = detect_face(image)

        if face is None:

            st.warning(
                "No face detected. Please upload a clear front-facing face image."
            )

        else:

            emotion, probabilities = predict_emotion(
                face,
                model
            )

            confidence = probabilities[
                EMOTIONS.index(emotion)
            ] * 100

            st.subheader("Prediction Result")

            st.success(
                f"Predicted Emotion: {emotion}"
            )

            st.info(
                f"Prediction Confidence: {confidence:.2f}%"
            )

            st.subheader("Emotion Probabilities")

            for i, emotion_name in enumerate(EMOTIONS):

                probability = probabilities[i] * 100

                st.write(
                    f"{emotion_name}: {probability:.2f}%"
                )

                st.progress(
                    float(probabilities[i])
                )

            st.subheader("Detected Face")

            st.image(
                face,
                caption="Face used for prediction",
                width=250
            )


st.divider()

st.caption(
    "Facial Expression Recognition | PyTorch + OpenCV + Streamlit"
)

st.caption(
    "FER2013 Test Accuracy: 55.25%"
)

