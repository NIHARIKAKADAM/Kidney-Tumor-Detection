
import streamlit as st
from PIL import Image

from prediction_grouped import KidneyTumorPredictor
from preprocessing import enhance_kidney_image


st.set_page_config(
    page_title="Kidney Tumor Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


@st.cache_resource
def get_predictor():
    return KidneyTumorPredictor()


# Dashboard styling
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 {
        font-size: 2.2rem !important;
    }

    h2, h3 {
        letter-spacing: -0.02em;
    }

    div.stButton > button {
        width: 100%;
        min-height: 3rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 10px;
    }

    div[data-testid="stMetric"] {
        padding: 18px;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🔬 Kidney Tumor Detection System")

st.subheader("Fuzzy-Enhanced CT Image Analysis")

st.caption(
    "DenseNet121 + ResNet101 Feature Fusion | "
    "SVM + Random Forest Weighted Ensemble"
)

st.info(
    "Academic research prototype. Predictions are experimental "
    "and must not be used for medical diagnosis."
)

st.divider()

st.header("1. Upload Kidney CT Image")

uploaded_file = st.file_uploader(
    "Choose a kidney CT scan",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")
        enhanced_image = enhance_kidney_image(image)

        st.header("2. Image Preprocessing")

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.subheader("Original CT Image")
            st.image(
                image,
                width=380
            )

        with col2:
            st.subheader("Fuzzy-Enhanced CT Image")
            st.image(
                enhanced_image,
                width=380
            )

        st.caption(
            "Experimental fuzzy-style enhancement used "
            "in the classification pipeline."
        )

        st.divider()

        st.header("3. AI Classification Result")

        if st.button(
            "🔍 Analyze CT Image",
            type="primary"
        ):

            with st.spinner("Analyzing CT image..."):
                predictor = get_predictor()
                result = predictor.predict(image)

            st.success(
                f"Predicted Class: {result['prediction']}"
            )

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Normal Model Score",
                    f"{result['normal_probability'] * 100:.2f}%"
                )

            with result_col2:
                st.metric(
                    "Tumor Model Score",
                    f"{result['tumor_probability'] * 100:.2f}%"
                )

            st.caption(
                "Experimental model scores; not clinically "
                "validated disease probabilities."
            )

    except Exception as error:
        st.error(
            f"Image processing or prediction failed: {error}"
        )

else:
    st.info("Upload a CT image to begin classification.")


st.divider()

st.caption(
    "Kidney Tumor Detection | Academic Research Project"
)

st.caption(
    "Not intended for clinical use or medical diagnosis."
)

