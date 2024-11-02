import streamlit as st
import pandas as pd
import io

from autoop.core.ml.model.classification import MultipleLogisticRegressor
from autoop.core.ml.model.classification import SVMClassifier
from autoop.core.ml.model.classification import KNearestNeighbors
from autoop.core.ml.model.regression import MultipleLinearRegression, Lasso
from autoop.core.ml.model.regression import XGBRegressor
from autoop.core.ml.feature import Feature
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
        st.write(df.head())  # generate preview for user
    except Exception as e:
        st.error(f"Error reading data: {e}")
else:
    st.warning("No datasets available. Please upload one.")
    st.stop()

# Step 2: Select Features and Target
st.subheader("2. Select Features and Target")
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

# Step 3: Choose Model Type
st.subheader("3. Choose Model Type")

model_mapping = {
    "Multiple Linear Regression": MultipleLinearRegression,
    "Multiple Logistic Regression": MultipleLogisticRegressor,
    "K Nearest Neighbours": KNearestNeighbors,
    "Support Vector Machine": SVMClassifier,
    "Lasso": Lasso,
    "XGBoost Regressor": XGBRegressor,
}

# show avaliable models for task type
if task_type == "Classification":
    model_choices = ["K Nearest Neighbours", "Support Vector Machine",
                     "Multiple Logistic Regression"]
else:
    model_choices = ["Multiple Linear Regression", "Lasso",
                     "XGBoost Regressor"]

model_choice = st.selectbox("Select a model", model_choices)

# institiate selected model
selected_model_class = model_mapping[model_choice]
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
