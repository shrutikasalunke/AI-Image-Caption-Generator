import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="AI Visual Intelligence Assistant",
    page_icon="",
    layout="wide"
)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("⚡ Features")

    feature = st.radio(
        "Select Analysis",
        [
            "Image Captioning",
            "Object Detection (Coming Soon)",
            "OCR Text Extraction (Coming Soon)",
            "Image Tags (Coming Soon)",
            "Full AI Report (Coming Soon)"
        ]
    )

# -----------------------------
# HEADER
# -----------------------------
st.title("AI Visual Intelligence Assistant")

st.markdown("""
Upload any image and let AI analyze it using Computer Vision.

### Features
- AI Image Captioning
- Object Detection (Upcoming)
- OCR Text Extraction (Upcoming)
- Image Tag Generation (Upcoming)
- AI Report Generation (Upcoming)
""")

# -----------------------------
# DASHBOARD METRICS
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("AI Models", "1")

with col2:
    st.metric("Current Feature", "Captioning")

with col3:
    st.metric("Status", "Active")

st.divider()

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    return processor, model


processor, model = load_model()

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# IMAGE PROCESSING
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:
        st.success("Image Uploaded")
        st.info("Ready for AI Analysis")

    st.divider()

    if st.button("Generate Caption"):

        with st.spinner("Generating caption..."):

            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            output = model.generate(**inputs)

            caption = processor.decode(
                output[0],
                skip_special_tokens=True
            )

        st.success("Caption Generated Successfully!")

        # -----------------------------
        # TABS
        # -----------------------------
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Caption",
                "Objects",
                "OCR",
                "Report"
            ]
        )

        # CAPTION TAB
        with tab1:
            st.subheader("Generated Caption")
            st.write(caption)

            st.download_button(
                label="Download Caption",
                data=caption,
                file_name="caption.txt",
                mime="text/plain"
            )

        # OBJECTS TAB
        with tab2:
            st.subheader("Object Detection")
            st.warning(
                "Object Detection feature will be added using YOLOv8."
            )

        # OCR TAB
        with tab3:
            st.subheader("OCR Text Extraction")
            st.warning(
                "OCR feature will be added using Tesseract OCR."
            )

        # REPORT TAB
        with tab4:
            st.subheader("AI Analysis Report")

            report = f"""
AI VISUAL INTELLIGENCE REPORT

Generated Caption:
{caption}

Analysis Type:
Image Captioning

Status:
Completed Successfully
"""

            st.text(report)

            st.download_button(
                label="Download Report",
                data=report,
                file_name="AI_Report.txt",
                mime="text/plain"
            )

        st.divider()

        st.subheader("AI Insights")

        st.info(
            f"""
The AI analyzed the uploaded image and generated a descriptive caption.

Generated Caption:
"{caption}"

Possible Use Cases:
• Social Media Posts
• Blogging
• Image Indexing
• Content Creation
• Accessibility Support
"""
        )

# -----------------------------
# FOOTER
# -----------------------------
st.divider()

st.markdown(
    """
    <center>
    Developed by Shrutika Salunke | AI Visual Intelligence Assistant
    </center>
    """,
    unsafe_allow_html=True
)