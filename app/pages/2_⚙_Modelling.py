import streamlit as st
import pandas as pd
import io

from autoop.core.ml.pipeline import Pipeline
from app.core.system import AutoMLSystem
from app.core.utils import (
    select_dataset,
    handle_nan_values,
    select_features_and_target,
    choose_model,
    set_train_test_split,
    select_metrics,
    display_pipeline_summary,
    train_pipeline,
    reset_pipeline,
    save_pipeline
)

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_header():
    st.title("⚙ Modeling Pipeline")
    st.write("Set up and train a model pipeline with your dataset.")
    st.write("If you ever want to reset the pipeline, "
             "click the '**Reset Pipeline**' button below!")


write_header()

if st.button("Reset Pipeline"):
    reset_pipeline()

automl = AutoMLSystem.get_instance()
selected_dataset = select_dataset(automl)

if "modified_df" not in st.session_state:
    # Load the data initially
    data = selected_dataset.read()
    data_io = io.BytesIO(data)
    original_df = pd.read_csv(data_io).replace(
        r'^\s*$', float('NaN'), regex=True)
    st.session_state["original_df"] = original_df

# Retrieve the original or modified DataFrame from session state
df = st.session_state.get("modified_df", st.session_state["original_df"])

# Display a preview of the DataFrame
st.write("Dataset Preview:")
st.write(df.head())

# Session states tracking whether NaN handling has been performed
if "nan_handling_confirmed" not in st.session_state:
    st.session_state.nan_handling_confirmed = False

if "nan_summary" not in st.session_state:
    st.session_state.nan_summary = ""

# Check for NaN values and handle them if any are found
if df.isna().values.any() and not st.session_state.nan_handling_confirmed:
    st.warning("Your dataset contains missing values (NaN). "
               "Please handle them before proceeding.")

    # Show NaN count and give user option to handle them
    df, summary_nan = handle_nan_values(df)
    if summary_nan:
        st.warning(summary_nan)
        st.session_state.nan_summary = summary_nan
        st.session_state.nan_handling_confirmed = True
        st.session_state["modified_df"] = df
        st.rerun()
else:
    # If NaN handling was performed, display summary
    if st.session_state.nan_handling_confirmed:
        st.subheader("NaN Summary")
        for key, value in st.session_state.nan_summary.items():
            st.write(f"**- {key}:** {value}")

    input_features, target_feature = select_features_and_target(df)
    task_type = "Classification" if target_feature.type == \
        "categorical" else "Regression"

    model = choose_model(df, task_type)
    split = set_train_test_split()
    metrics = select_metrics(task_type)

    pipeline = Pipeline(dataset=selected_dataset, model=model,
                        input_features=input_features,
                        target_feature=target_feature,
                        metrics=metrics, split=split)

    display_pipeline_summary(pipeline)
    train_pipeline(pipeline)
    save_pipeline(automl, pipeline)
