import streamlit as st
import pandas as pd
import numpy as np
import io

from autoop.core.ml.model.classification import MultipleLogisticRegressor
from autoop.core.ml.model.classification import SVMClassifier
from autoop.core.ml.model.classification import KNearestNeighbors
from autoop.core.ml.model.regression import MultipleLinearRegression, Lasso
from autoop.core.ml.model.regression import XGBRegressor
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.metric import get_metric
from autoop.core.ml.pipeline import Pipeline
from app.core.system import AutoMLSystem



st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline"
                  " to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

st.title("Modeling Pipeline")
st.write("Set up and train a model pipeline with your dataset.")

# Step 1: Select a Dataset
st.subheader("1. Select a Dataset")
datasets = automl.registry.list(type="dataset")

if datasets:
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_names)
    selected_dataset = next(ds for ds in datasets if
                            ds.name == selected_dataset_name)

    data = selected_dataset.read()
    st.write("Dataset preview (first five rows):")
    data_io = io.BytesIO(data)
    try:
        df = pd.read_csv(data_io)
        # EXTRA FEATURE FOR DEALING WITH NAN VALUES
        df = df.replace(r'^\s*$', float('NaN'), regex=True)
        st.write(df.head())  # generate preview for user
    except Exception as e:
        st.error(f"Error reading data: {e}")
else:
    st.warning("No datasets available. Please upload one.")
    st.stop()

# Step 2: Select Features and Target
st.subheader("2. Select Features and Target")


def try_convert(value: str) -> float | str:
    """ Converts values to numeric if possible, otherwise returns as is """
    try:
        return pd.to_numeric(value)
    except ValueError:
        return value


st.session_state.nans_left = False
# EXTRA FEATURE FOR DEALING WITH NAN VALUES
if df.isna().values.any():
    st.session_state.nans_left = True
    for col in df.columns:  # convert to numeric (NaNs hinder type checking)
        df[col] = df[col].apply(try_convert)

    st.warning(f"There are {df.isna().sum().sum()} NaN values in the dataset. "
               "What do you want to do with them?")

    option = st.selectbox("Select an option", ["keep", "remove",
                                               "interpolate", "fill with 0"])

    if option == "remove":
        confirmation = st.button("Confirm removal of NaN values")
        if confirmation:
            df = df.dropna().reset_index(drop=True)
            st.write("Succesfully removed NaN values!")
        else:
            st.stop()
    elif option == "keep":
        confirmation = st.button("Confirm keeping NaN values")
        if confirmation:
            st.write("Keeping NaN values.")
        else:
            st.stop()
    elif option == "interpolate":
        confirmation = st.button("Confirm linear interpolation")
        if confirmation:
            st.write("Interpolating NaN values.")
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
        else:
            st.stop()
    elif option == "fill with 0":
        confirmation = st.button("Confirm filling NaN values with 0")
        if confirmation:
            st.write("Filling NaN values with 0.")
            df = df.fillna(0)
        else:
            st.stop()
    nans_left = df.isna().sum().sum()
    st.write(f"Applying this operation left you with {nans_left} NaN values.")

    if nans_left == 0:
        st.session_state.nans_left = False

features = list(df.columns)

input_feature_names = st.multiselect("Select input features", features)

# exclude input features from target feature options
remaining_features = [feature for feature in features if feature
                      not in input_feature_names]
target_feature_name = st.selectbox("Select target feature", remaining_features)

if not input_feature_names or not target_feature_name:
    st.warning("Please select at least one input feature and one "
               "target feature.")
    st.stop()

# convert the selected features into instances of the features
input_features = [Feature(name=name, type='categorical' if df[name].dtype
                          == 'object' else 'numerical'
                          ) for name in input_feature_names]
target_feature = Feature(name=target_feature_name, type='categorical'
                         if df[target_feature_name].dtype == 'object'
                         else 'numerical')

# determine task type for step 3
if target_feature.type == "categorical":
    task_type = "Classification"
else:
    task_type = "Regression"

# Step 3: Choose Model Type and Set Hyperparameters
st.subheader("3. Choose Model Type")

# mapping of models with corresponding hyperparameters to be chosen
model_mapping = {
    "Multiple Linear Regression": (MultipleLinearRegression, {}),
    "Multiple Logistic Regression": (MultipleLogisticRegressor, {
        "C": (st.slider, "C", 0.01, 10.0, 1.0, 0.1),
        "penalty": (st.selectbox, "Penalty", ["l1", "l2",
                                              "elasticnet", "none"]),
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
        "learning_rate": (st.slider, "Learning Rate", 0.00, 1.00, 0.1, 0.01),
        "n_estimators": (st.slider, "Number of Estimators", 0, 500, 100),
        "gamma": (st.slider, "Gamma", 0.0, 5.0, 0.0, 0.1),
    }),
}

if task_type == "Classification":
    model_choices = ["K Nearest Neighbours", "Support Vector Machine",
                     "Multiple Logistic Regression"]
else:
    model_choices = ["Multiple Linear Regression", "Lasso",
                     "XGBoost Regressor"]

# prompt user to select a model
model_choice = st.selectbox("Select a model", model_choices)

selected_model_class, hyperparams = model_mapping[model_choice]

# process hyperparameters
if hyperparams:
    st.subheader("Hyperparameter Configuration")
    chosen_params = {}
    for param_name, (input_fn, label, *args) in hyperparams.items():
        chosen_params[param_name] = input_fn(label, *args)

    # intiaize model model with chosen hyperparameters
    model = selected_model_class(**chosen_params)
else:
    # no hyperparameters
    model = selected_model_class()


# Step 4: Configure Split
st.subheader("4. Set Train-Test Split")
split = st.slider("Training set percentage", min_value=0.1,
                  max_value=0.9, value=0.8)

# Step 5: Select Metrics
st.subheader("5. Choose Evaluation Metrics")

classification_metrics = ["Accuracy", "Recall", "Precision", "F1"]
regression_metrics = ["MSE", "RMSE", "R2"]

if task_type == "Classification":
    selected_metrics = st.multiselect("Select metrics", classification_metrics)
else:
    selected_metrics = st.multiselect("Select metrics", regression_metrics)

if not selected_metrics:
    st.warning("Please select at least one metric.")
    st.stop()

# convert selected metrics into respective instances
metric_to_use = [get_metric(name) for name in selected_metrics]


# Step 6: Pipeline Summary

pipeline = Pipeline(
    dataset=selected_dataset,
    model=model,
    input_features=input_features,
    target_feature=target_feature,
    metrics=metric_to_use,
    split=split
)

# print the beautifully formatted summary
st.subheader("Pipeline Summary​")
st.write(pipeline.__str__())

# Step 7: Train Pipeline
if st.button("Train Pipeline"):

    # add error exc?
    results = pipeline.execute()
    # not working atm

    st.success("Training complete!")
    st.write("Evaluation Results:")
    for metric_result in results["metrics"]:
        st.write(metric_result)
