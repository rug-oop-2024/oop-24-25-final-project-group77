import streamlit as st
import pandas as pd
import io

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
        data_io = io.BytesIO(data)
        try:
            df = pd.read_csv(data_io)
            st.write(df.head())
        except Exception as e:
            st.error(f"Error reading data: {e}")

    if 'confirm_delete' not in st.session_state:
        st.session_state.confirm_delete = False

    if st.button("Delete Dataset"):
        st.session_state.confirm_delete = True

    if st.session_state.confirm_delete:
        st.error("Once you delete a dataset, "
                 "it cannot be recovered. Are you sure?")
        if st.button("Confirm Deletion"):
            st.write(f"Deleting {selected_dataset_name}")
            artifact_id = selected_dataset.id
            automl.registry.delete(artifact_id)
            st.warning(f"Deleted {selected_dataset_name}")
            st.session_state.confirm_delete = False
            st.rerun()

        if st.button("Cancel"):
            st.session_state.confirm_delete = False

else:
    st.info("No datasets available in the system. Please upload one.")

# Initialize upload success flag
if "upload_success" not in st.session_state:
    st.session_state.upload_success = False

st.title("Upload New Dataset")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)

        asset_path = f"datasets/{uploaded_file.name}"
        version = "ohio"

        # Create a new Dataset instance
        new_dataset = Dataset.from_dataframe(
            name=uploaded_file.name,
            data=data,
            asset_path=asset_path,
            version=version
        )

        if st.button("Confirm Upload"):
            automl.registry.register(new_dataset)
            st.session_state.upload_success = True

            st.session_state.uploaded_file = None
            st.rerun()

    except Exception as e:
        st.error(f"Failed to upload dataset: {e}")

# Display success message iff upload was successful and confirmed
if st.session_state.upload_success:
    st.success("Dataset uploaded successfully!")
    st.session_state.upload_success = False
