import io
import os

from PIL import Image
import streamlit as st
import google.generativeai as genai


SYSTEM_PROMPT = """You are an expert, board-certified medical specialist and diagnostic radiologist AI. Your task is to analyze clinical images (such as X-rays, MRIs, CT scans, ultrasounds, or clinical pathology photographs) along with any provided patient history, and deliver a comprehensive, highly accurate medical evaluation.

Always adhere to the following strict clinical guidelines:
1. IMAGING ANALYSIS: Systematically describe the visual findings. Note anatomical landmarks, abnormalities, lesions, structural changes, and artifacts. State what is visible and what is explicitly absent (pertinent negatives).
2. DIFFERENTIAL DIAGNOSIS: Provide a prioritized list of potential diagnoses based on the image findings and clinical context. Rank them from most likely to least likely, and highlight any life-threatening "must-not-miss" conditions.
3. RECOMMENDATIONS: Suggest appropriate next steps, which may include further specific imaging modalities, laboratory tests, or immediate specialist consultations.
4. CLINICAL TONE & SAFETY: Maintain an objective, precise, and professional medical tone. Use standard medical terminology. Include a clear, mandatory clinical disclaimer stating that this AI analysis is an educational support tool and must be verified by a human attending physician before making any treatment decisions.
"""


def configure_gemini() -> None:
    # Try environment variable first
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Fall back to secrets.toml if it exists
    if not api_key:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
    
    if not api_key:
        st.error(
            "GOOGLE_API_KEY not found!\n\n"
            "Set it using one of:\n"
            "1. Environment variable: GOOGLE_API_KEY\n"
            "2. Create .streamlit/secrets.toml with GOOGLE_API_KEY=\"your-key\""
        )
        st.stop()
    
    genai.configure(api_key=api_key)


def build_model():
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024,
        "response_mime_type": "text/plain",
    }

    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=generation_config,
        system_instruction=SYSTEM_PROMPT,
    )


st.set_page_config(page_title="MedScan", page_icon="🩺", layout="centered")
st.title("MedScan")
st.write("Upload a medical image and optional context to get a structured analysis.")

configure_gemini()
model = build_model()

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
patient_history = st.text_area("Optional patient history", placeholder="Add brief clinical context here...")

if uploaded_file:
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    st.image(image, caption="Uploaded image", use_container_width=True)

if st.button("Analyze"):
    if not uploaded_file:
        st.warning("Please upload an image first.")
    else:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        prompt = (
            f"Patient history:\n{patient_history.strip() or 'Not provided.'}\n\n"
            "Analyze the uploaded image and provide a structured clinical response."
        )
        response = model.generate_content([prompt, image])
        st.subheader("Analysis")
        st.write(response.text)




