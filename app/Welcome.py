import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
st.sidebar.success("Select a page above.")

st.write("# Welcome to AutoML! 👋")
st.write("## How to use this app?")
st.write(
    """
    Here's how you can use this app:
    - **Upload Dataset**: Upload a dataset to the app.
    - **Select Dataset**: Select a dataset from the app.
    - **Train Model**: Train a model on the selected dataset.
    - **Deploy Model**: Deploy the trained model for prediction.

    **Note**: There are datasets provided in the repository which are
    available to use. You can also upload your own dataset and use
    that instead!
    """
)

st.write("## Authors")
st.write("This app was created by Group 77 for OOP 2024 at RUG.")
