import streamlit as st

from app.core.system import AutoMLSystem
from app.core.utils import (
    select_pipeline,
    load_pipeline,
    predict_pipeline
                            )

# Initialize AutoML system instance
automl = AutoMLSystem.get_instance()

st.title("Pipeline Manager")
st.write("Manage Pipelines available in the AutoML system.")

selected_pipeline = select_pipeline(automl)
loaded_pipeline = load_pipeline(selected_pipeline)

if loaded_pipeline:
    st.success("Pipeline loaded successfully!")
else:
    st.error("Failed to load pipeline.")

st.subheader("Pipeline Summary")
st.write(loaded_pipeline)  # not sure if right model used

st.subheader("Upload a Dataset")
predict_pipeline(loaded_pipeline)

# st.write(pickle.loads(selected_pipeline.data))
