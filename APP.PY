import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import io
import os
import json
from gtts import gTTS
import tempfile
import numpy as np
import cv2

# --- Configure Gemini API ---
API_KEY = "AIzaSyAllFwTyIKZ79s1-iBkSakzpUdWiRWASgw"  # Replace with your actual key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# --- Cartoonify function ---
def cartoonify_image(pil_img):
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for _ in range(7):
        img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(img, edges_colored)
    cartoon_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cartoon_rgb)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="AI Image Caption & Cartoon Generator", layout="wide")
st.title("📸 AI Image Caption & Cartoon Generator using Gemini 1.5")

st.markdown("Upload images and generate captions or cartoon style images using Gemini AI and OpenCV filters.")

# --- Sidebar Options ---
st.sidebar.header("🧠 Feature Controls")

caption_style = st.sidebar.selectbox("🎯 Caption Style", ["Professional", "Funny", "Poetic", "Romantic", "For Instagram"])
story_mode = st.sidebar.checkbox("📖 Story Mode (Short Story from Image)")
blind_assist = st.sidebar.checkbox("🧑‍🦯 Blind Assist Mode (Text-to-Speech Description)")
object_detect = st.sidebar.checkbox("🧠 Object & Activity Detection")
multilingual = st.sidebar.selectbox("🌍 Translate Caption to", ["None", "Hindi", "Telugu", "Spanish"])
instagram_tags = st.sidebar.checkbox("📷 Instagram Hashtags")

generate_caption_checkbox = st.sidebar.checkbox("📝 Enable Caption Generation")
generate_cartoon_checkbox = st.sidebar.checkbox("🖌️ Enable Cartoon Image Generation")

generate_caption_button = None
generate_cartoon_button = None

# --- File Uploader ---
uploaded_files = st.file_uploader("📁 Upload up to 10 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.markdown("### 🖼️ Uploaded Images")
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown("---")
        col1, col2 = st.columns([1, 2])

        with col1:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.image(image, caption=f"Image {idx + 1}", use_column_width=True)

        with col2:
            # Caption generation
            if generate_caption_checkbox:
                if st.button(f"Generate Caption for Image {idx + 1}"):
                    # Build prompt parts for caption generation
                    prompt_parts = [f"Give a {caption_style.lower()} caption for this image."]
                    if story_mode:
                        prompt_parts.append("Also generate a short 3-5 line story based on the image.")
                    if object_detect:
                        prompt_parts.append("List the objects, actions, and type of scene in the image.")
                    if instagram_tags:
                        prompt_parts.append("Generate 5-10 trending hashtags suitable for Instagram based on this image.")

                    try:
                        with st.spinner("🚀 Generating caption using Gemini..."):
                            response = model.generate_content([
                                "\n".join(prompt_parts),
                                {"mime_type": "image/jpeg", "data": image_bytes}
                            ])
                            result = response.text.strip()

                        # Translation
                        if multilingual != "None":
                            result_translated = model.generate_content(
                                f"Translate the following to {multilingual}:\n{result}"
                            ).text.strip()
                            st.markdown(f"**🌐 Translated ({multilingual}) Output:**")
                            st.markdown(result_translated)

                        st.markdown("**📝 Generated Caption Output:**")
                        st.markdown(result)

                        # TTS
                        if blind_assist:
                            tts = gTTS(text=result, lang='en')
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                                tts.save(fp.name)
                                st.audio(fp.name, format="audio/mp3")

                        # Downloads
                        st.markdown("**💾 Download Options:**")
                        filename_base = os.path.splitext(uploaded_file.name)[0]
                        st.download_button("📄 Caption (.txt)", result.encode(), file_name=f"{filename_base}_caption.txt")
                        st.download_button("🧾 Metadata (.json)", json.dumps({"caption": result}, indent=2).encode(),
                                           file_name=f"{filename_base}_meta.json")

                        # Overlay Caption on Image
                        caption_overlay = result.split("\n")[0][:100] + "..."
                        img_draw = image.copy()
                        draw = ImageDraw.Draw(img_draw)
                        font = ImageFont.load_default()
                        draw.rectangle([0, 0, img_draw.width, 50], fill=(0, 0, 0))
                        draw.text((10, 10), caption_overlay, fill=(255, 255, 255), font=font)

                        buffered = io.BytesIO()
                        img_draw.save(buffered, format="JPEG")
                        st.download_button("🖼️ Image with Caption", buffered.getvalue(),
                                           file_name=f"{filename_base}_with_caption.jpg", mime="image/jpeg")

                    except Exception as e:
                        st.error(f"❌ Error generating caption: {e}")

            # Cartoon image generation
            if generate_cartoon_checkbox:
                if st.button(f"Generate Cartoon Image for Image {idx + 1}"):
                    with st.spinner("🎨 Generating cartoon image..."):
                        cartoon_img = cartoonify_image(image)
                        st.image(cartoon_img, caption="Cartoonified Image", use_column_width=True)

                        # Download cartoon image
                        buf = io.BytesIO()
                        cartoon_img.save(buf, format="JPEG")
                        byte_im = buf.getvalue()
                        st.download_button("📥 Download Cartoon Image", byte_im, file_name=f"{os.path.splitext(uploaded_file.name)[0]}_cartoon.jpg", mime="image/jpeg")

else:
    st.info("📥 Upload images from the uploader above to get started.")
