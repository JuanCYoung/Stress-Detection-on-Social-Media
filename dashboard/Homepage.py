import seaborn as sns
import streamlit as st
import base64
def add_logo():
    with open('../dashboard/logo/logo_filkom_resized.png', 'rb') as f:
        image_data = f.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url('data:image/png;base64,{encoded_image}');
                background-repeat: no-repeat;
                padding-top: 20px;
                background-position: 20px 20px;
            }}
            [data-testid="stSidebarNav"]::before {{
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

sns.set(style='dark')

st.set_page_config(
    page_title="Stress Detection on Social Media",
    page_icon=":sparkles:",
)

add_logo()
st.title("Welcome To Stress Sentiment Analyzer:sparkles:")
st.header("Discover Your Stress Sentiment with Advanced AI")
st.image("../dashboard/image/istockphoto-1281210009-612x612.jpg")
st.write(
    "Life can be hectic, and stress is a natural part of the human experience. Welcome to the Stress Sentiment Analyzer, "
    "powered by cutting-edge BERT and ROBERTA algorithms."
)

st.subheader("How to Get Started :question:")
st.write(
    "- **Go to Stress Analyzer**: Uncover insights into your stress levels.\n"
    "- **Share Your Concerns**: Let us know what's on your mind.\n"
    "- **AI Assessment**: Our advanced AI will analyze and determine your stress status."
)


# Why Stress Sentiment Analyzer section
st.subheader("Why Stress Sentiment Analyzer?")
st.write(
    " - **Cutting-Edge Technology:** Benefit from the precision of BERT and ROBERTA models, ensuring accurate and nuanced analysis.\n"
    " - **Instant Feedback:** Receive immediate insights into your stress sentiment to make informed decisions about your well-being.\n"
    " - **Confidential and Secure:** Your privacy is our priority. The Stress Sentiment Analyzer does not store your data, ensuring a confidential and secure experience."
)

