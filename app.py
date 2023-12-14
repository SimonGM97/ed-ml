from ed_ml.streamlit.inference_page import build_inference_page
from ed_ml.streamlit.model_registry_page import build_model_registry_page
import streamlit as st
from PIL import Image
import os


# streamlit run app.py
if __name__ == '__main__':
    # Set Page Config
    st.set_page_config(layout="wide")

    # Define first row
    row00, row01, row02 = st.columns([3, 3, 3])

    # Show Ed Machina Logo
    row01.image(Image.open(os.path.join("docs", "images", "logo_no_background.png")), use_column_width=True)
    
    # Blank space
    st.write("#")
    st.write("#")
    st.write("#")

    # Sidebar
    sidebar = st.sidebar.selectbox('Choose Page:', ['New Inferences', 'Model Registry'])

    if sidebar == 'New Inferences':
        # Build Inference Page
        build_inference_page()

    if sidebar == 'Model Registry':
        # Build Model Registry Page
        build_model_registry_page()

    