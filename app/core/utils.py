"""
Module for storing multiple utilities necessary for clean code in the pages
"""

import pickle
import streamlit as st
import pandas as pd

from autoop.core.ml.model.classification import MultipleLogisticRegressor
from autoop.core.ml.model.classification import SVMClassifier
from autoop.core.ml.model.classification import KNearestNeighbors
from autoop.core.ml.model.regression import MultipleLinearRegression, Lasso
from autoop.core.ml.model.regression import XGBRegressor
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.feature import Feature
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.metric import get_metric
from app.core.system import AutoMLSystem


def get_model_parameter_mapping():
    model_mapping = {
        "Multiple Linear Regression": (MultipleLinearRegression, {}),
        "Multiple Logistic Regression": (MultipleLogisticRegressor, {
            "C": (st.slider, "C", 0.01, 10.0, 1.0, 0.1),
            "penalty": (st.selectbox, "Penalty", ["l2", "None"]),
        }),
        "K Nearest Neighbours": (KNearestNeighbors, {
            "k": (st.slider, "K", 1, 10, 3),
        }),
        "Support Vector Machine": (SVMClassifier, {
            "C": (st.slider, "C", 0.01, 10.0, 1.0, 0.1),
            "kernel": (st.selectbox, "Kernel", ["linear",
                                                "poly", "rbf", "sigmoid"]),
            "degree": (st.slider, "Degree", 1, 5, 3, 1)
        }),
        "Lasso": (Lasso, {
            "alpha": (st.slider, "Alpha", 0.0, 10.0, 1.0, 0.5),
        }),
        "XGBoost Regressor": (XGBRegressor, {
            "max_depth": (st.slider, "Max Depth", 1, 15, 6),
            "learning_rate": (st.slider, "Learning Rate", 0.00, 1.00, 0.1,
                              0.01),
            "n_estimators": (st.slider, "Number of Estimators", 0, 500, 100),
            "gamma": (st.slider, "Gamma", 0.0, 5.0, 0.0, 0.1),
        }),
    }

    return model_mapping


def reset_pipeline():
    if "original_df" in st.session_state:
        del st.session_state["original_df"]
    if "modified_df" in st.session_state:
        del st.session_state["modified_df"]
    if "nan_handling_confirmed" in st.session_state:
        del st.session_state["nan_handling_confirmed"]
    if "nan_summary" in st.session_state:
        del st.session_state["nan_summary"]
    st.rerun()


def select_dataset(automl):
    st.subheader("1. Select a Dataset")
    datasets = automl.registry.list(type="dataset")
    if datasets:
        dataset_names = [dataset.name for dataset in datasets]
        selected_dataset_name = st.selectbox("Select a dataset", dataset_names)
        selected_dataset = next(ds for ds in datasets if ds.name
                                == selected_dataset_name)
        return selected_dataset
    else:
        st.warning("No datasets available. Please upload one.")
        st.stop()


def try_convert(value: str) -> float | str:
    """ Converts values to numeric if possible, otherwise returns as is """
    try:
        return pd.to_numeric(value)
    except ValueError:
        return value


def handle_nan_values(df):
    """Handle NaN values based on user input."""
    initial_nan_count = df.isna().sum().sum()  # Initial count of NaN values
    st.write(f"There are {initial_nan_count} NaN values in the dataset.")

    for col in df.columns:
        df[col] = df[col].apply(try_convert)

    option = st.selectbox("Choose how to handle NaN values:",
                          ["Keep", "Remove", "Interpolate", "Fill with 0"])

    if st.button("Confirm"):
        if option == "Remove":
            df = df.dropna().reset_index(drop=True)
            final_nan_count = df.isna().sum().sum()
            st.success("NaN values removed.")
            nan_method = "Remove NaN Values"
        elif option == "Keep":
            final_nan_count = initial_nan_count
            st.info("NaN values kept.")
            nan_method = "Keep NaN Values"
        elif option == "Interpolate":
            df = df.interpolate(method="linear").dropna(
            ).reset_index(drop=True)
            final_nan_count = df.isna().sum().sum()
            st.success("NaN values interpolated.")
            nan_method = "Interpolate NaN Values"
        elif option == "Fill with 0":
            df = df.fillna(0)
            final_nan_count = df.isna().sum().sum()
            st.success("NaN values filled with 0.")
            nan_method = "NaN values filled with 0"

        nan_summary = {
            "Method Used": nan_method,
            "Initial NaN Count": initial_nan_count,
            "Remaining NaN Count": final_nan_count,
            "Change in NaN Count": initial_nan_count - final_nan_count
        }

        return df, nan_summary

    return df, {}


def select_features_and_target(df):
    st.subheader("2. Select Features and Target")
    features = list(df.columns)
    input_features = st.multiselect("Select input features", features)
    target_feature = st.selectbox("Select target feature",
                                  [f for f in features if f not in
                                   input_features])

    if not input_features or not target_feature:
        st.warning("Please select at least one input feature "
                   "and one target feature.")
        st.stop()

    return [Feature(name=feat, type='categorical' if df[feat].dtype ==
                    'object' else 'numerical') for feat in input_features], \
        Feature(name=target_feature, type='categorical' if
                df[target_feature].dtype == 'object' else 'numerical')


def choose_model(df, task_type):
    st.subheader("3. Choose Model Type")
    model_mapping = get_model_parameter_mapping()
    if df.isna().sum().sum() > 0:
        st.warning("As there are still NA values, model "
                   "selection is restricted")
        model_choices = ["Support Vector Machine",] \
            if task_type == "Classification" else \
                        ["XGBoost Regressor"]
    else:
        model_choices = ["K Nearest Neighbours", "Support Vector Machine",
                         "Multiple Logistic Regression"
                         ] if task_type == "Classification" else \
                        ["Multiple Linear Regression", "Lasso",
                         "XGBoost Regressor"]
    model_choice = st.selectbox("Select a model", model_choices)
    selected_model_class, hyperparams = model_mapping[model_choice]

    chosen_params = {param_name: input_fn(label, *args) for param_name,
                     (
                         input_fn, label, *args
                         ) in hyperparams.items()} if hyperparams else {}
    return selected_model_class(**chosen_params)


def set_train_test_split():
    st.subheader("4. Set Train-Test Split")
    return st.slider("Training set percentage", min_value=0.1, max_value=0.9,
                     value=0.8)


def select_metrics(task_type):
    st.subheader("5. Choose Evaluation Metrics")
    metrics = ["Accuracy", "Recall", "Precision", "F1"
               ] if task_type == "Classification" else [
                   "MSE", "RMSE", "R2"]
    selected_metrics = st.multiselect("Select metrics", metrics)

    if not selected_metrics:
        st.warning("Please select at least one metric.")
        st.stop()

    return [get_metric(name) for name in selected_metrics]


def display_pipeline_summary(pipeline):
    st.subheader("Pipeline Summary")
    st.write(str(pipeline))


def train_pipeline(pipeline):
    # ! removed buttom since it caused too many issues with rerunning !
    results = pipeline.execute()
    if results:
        st.subheader("Evaluation Results")
        st.success("Training complete!")
        for metric_result in results["metrics"]:
            metric_name = metric_result[1].__class__.__name__
            st.write(f"**Metric**: {metric_name}")
            st.write(f"- {metric_result[0]} {metric_result[2]:.5f}")
            st.write(f"- {metric_result[3]} {metric_result[5]:.5f}")
            st.write("\n")
        st.write(pd.DataFrame(results["predictions"]))


def serialize_pipeline_data(pipeline):
    """Serialize pipeline artifacts, model, and metrics."""

    artifacts_data = {
        'model': pipeline.model,
        'artifacts': pipeline.artifacts,
        'metrics': pipeline.metrics
    }

    return pickle.dumps(artifacts_data)


def save_pipeline(automl: AutoMLSystem, pipeline: Pipeline) -> None:
    with st.form("save_pipeline"):
        pl_name = st.text_input("Enter a name for the pipeline:")
        pl_version = st.text_input("Enter a version for the pipeline:")

        if st.form_submit_button("Save Pipeline"):
            if pl_name and pl_version:
                artifact_pipeline = Artifact(
                    name=pl_name,
                    version=pl_version,
                    asset_path=f"pipelines/{pl_name}",
                    data=serialize_pipeline_data(pipeline),
                    type="pipeline"
                )

                automl.registry.register(artifact_pipeline)
            else:
                st.warning("Please enter a name and version for the pipeline.")
