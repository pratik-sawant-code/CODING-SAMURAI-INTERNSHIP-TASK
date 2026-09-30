import torch
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="wide"
)

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model.eval()

    return processor, model

processor, image_captioner = load_model()

def generate_caption(image):
    inputs = processor(
        images=image,
        text="a photo of"
        , return_tensors="pt"
    )

    with torch.no_grad():
        output = image_captioner.generate(
            **inputs,
            max_new_tokens=40,
            num_beams=5,
            repetition_penalty=1.2
        )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    ).strip()

    if caption.lower().startswith("a photo of"):
        caption = caption[9:].strip()

    if caption:
        caption = caption[0].upper() + caption[1:]

    if not caption.endswith("."):
        caption += "."

    return caption

def enhance_caption(caption, style):
    if style == "Simple":
        return caption

    if style == "Professional":
        return (
            f"{caption}. "
            "The image presents the subject in a clear, "
            "professional, and visually engaging manner."
        )

    if style == "Creative":
        return (
            f"{caption}. "
            "A beautiful moment captured through creative "
            "visual storytelling."
        )

    if style == "Social Media":
        return (
            f"{caption} ✨ "
            "A moment worth sharing and remembering. "
            "#Photography #VisualStory #Moments"
        )

    return caption

st.title("🖼️ AI Image Caption Generator")

st.write(
    "Upload an image and let AI generate a natural-language caption."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Project", "6")

with col2:
    st.metric("AI Model", "BLIP")

with col3:
    st.metric("Mode", "AI")

st.divider()

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")

        st.image(
            image,
            width="stretch"
        )

    with col2:
        st.subheader("🤖 AI Caption Generator")

        if st.button(
            "✨ Generate Caption",
            width="stretch"
        ):

            with st.spinner("AI is analyzing the image..."):

                try:
                    caption = generate_caption(image)

                    st.session_state["caption"] = caption

                    if "enhanced_caption" in st.session_state:
                        del st.session_state["enhanced_caption"]

                    st.success("Caption generated successfully!")

                except Exception as e:
                    st.error("Could not generate caption.")
                    st.exception(e)

    if "caption" in st.session_state:

        st.divider()

        st.subheader("📝 Generated Caption")

        st.info(
            st.session_state["caption"]
        )

        st.subheader("✨ Enhance Caption")

        style = st.selectbox(
            "Choose caption style",
            [
                "Simple",
                "Professional",
                "Creative",
                "Social Media"
            ]
        )

        if st.button(
            "🚀 Enhance Caption",
            width="stretch"
        ):

            enhanced = enhance_caption(
                st.session_state["caption"],
                style
            )

            st.session_state["enhanced_caption"] = enhanced

        if "enhanced_caption" in st.session_state:

            st.success("Enhanced caption ready!")

            st.info(
                st.session_state["enhanced_caption"]
            )

            st.download_button(
                "📥 Download Caption",
                data=st.session_state["enhanced_caption"],
                file_name="enhanced_caption.txt",
                mime="text/plain",
                width="stretch"
            )

st.divider()

st.caption(
    "AI Image Caption Generator | "
    "Python + PyTorch + Transformers + Streamlit | "
    "Coding Samurai Internship Project 6"
)