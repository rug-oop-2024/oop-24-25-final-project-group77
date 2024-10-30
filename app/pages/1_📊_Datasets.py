import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

# Initialize AutoML system instance
automl = AutoMLSystem.get_instance()

# List of datasets from the AutoML system
datasets = automl.registry.list(type="dataset")

st.title("Dataset Manager")
st.write("Manage datasets available in the AutoML system.")

st.subheader("Available Datasets")
if datasets:
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_names)
    selected_dataset = next(ds for ds in datasets if
                            ds.name == selected_dataset_name)

    if st.button("Show Preview"):
        data = selected_dataset.read()
        st.write(f"Preview of **{selected_dataset_name}**:")
        st.dataframe(data.head())

    if st.button("Delete Dataset"):
        if st.button("Confirm Deletion"):
            automl.registry.delete(selected_dataset)
            st.warning(f"Deleted {selected_dataset_name}")
            st.rerun()  # Change here from experimental_rerun to rerun

else:
    st.info("No datasets available in the system. Please upload one.")

# Upload a new dataset
st.subheader("Upload New Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    try:
        # Load the uploaded file into a DataFrame
        data = pd.read_csv(uploaded_file)

        asset_path = f"datasets/{uploaded_file.name}"
        version = "ohio"  # Set a default version

        st.write(f"Dataset name: {uploaded_file.name}")

        # Create a new Dataset instance
        new_dataset = Dataset.from_dataframe(
            name=uploaded_file.name,
            data=data,
            asset_path=asset_path,
            version=version
        )

        st.write(f"Creating dataset with name: {new_dataset.name}")

        # Register the new dataset
        automl.registry.register(new_dataset)
        st.success(f"Uploaded and registered {uploaded_file.name} successfully!")
        st.rerun()  # Change here from experimental_rerun to rerun
    except Exception as e:
        st.error(f"Failed to upload dataset: {e}")
