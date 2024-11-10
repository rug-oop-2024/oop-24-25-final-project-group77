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
results = predict_pipeline(loaded_pipeline)

if results is not None:
    for metric_result in results["metrics"]:
        metric_name = metric_result[1].__class__.__name__
        st.write(f"**Metric**: {metric_name}")
        st.write(f"- {metric_result[0]} {metric_result[2]:.5f}")
        st.write(f"- {metric_result[3]} {metric_result[5]:.5f}")
        st.write("\n")
    st.write("Predictions:")
    predictions = results["predictions"]
    st.dataframe(predictions)

# st.write(pickle.loads(selected_pipeline.data))
