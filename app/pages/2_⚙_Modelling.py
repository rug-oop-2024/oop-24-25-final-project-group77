import streamlit as st
import pandas as pd
import io
import re

from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import get_metric
from autoop.core.ml.pipeline import Pipeline
from app.core.system import AutoMLSystem
from app.core.utils import get_model_parameter_mapping


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

    # check if a modified dataframe is already in session state
    if "modified_df" not in st.session_state:
        data = selected_dataset.read()
        data_io = io.BytesIO(data)
        try:
            original_df = pd.read_csv(data_io)
            original_df = original_df.replace(r'^\s*$', float(
                'NaN'), regex=True)
            st.session_state["original_df"] = original_df
        except Exception as e:
            st.error(f"Error reading data: {e}")
            st.stop()

    # use the modified df if it exists, otherwise fall back to the original
    df = st.session_state.get("modified_df", st.session_state["original_df"])
    st.write("Dataset preview (first five rows):")
    st.write(df.head())
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


# EXTRA FEATURE FOR DEALING WITH NAN VALUES
if "nan_handling_confirmed" not in st.session_state:
    st.session_state.nan_handling_confirmed = False

if "nan_summary" not in st.session_state:
    st.session_state.nan_summary = ""

if df.isna().values.any() and not st.session_state.nan_handling_confirmed:
    # make it nice!!!
    for col in df.columns:
        df[col] = df[col].apply(try_convert)

    # we use placeholders throughout so that when the user has successfully
    # dealt with the nan values. the placeholders can be cleared so the section
    # is cleared so that we can prompt them to redo this step again and start
    # from scratch with the refreshed dataset (unaltered)
    # ! its messy but it prevents errors with rerun and users backtracking !

    warning_placeholder = st.empty()
    warning_placeholder.warning(f"There are {df.isna().sum().sum()}"
                                " NaN values in the dataset. "
                                "What do you want to do with them?")

    option_placeholder = st.empty()  # placeholder to clear
    option = option_placeholder.selectbox("Select an option",
                                          ["keep", "remove", "interpolate",
                                           "fill with 0"])

    confirmation_placeholder = st.empty()  # placeholder to clear

    if option == "remove":
        confirmation = confirmation_placeholder.button("Confirm removal of "
                                                       "NaN values")
        if confirmation:
            df = df.dropna().reset_index(drop=True)
            confirmation_placeholder.success("Successfully removed NaN"
                                             " values!")
            st.session_state["modified_df"] = df  # store the modified df
            st.session_state.nan_handling_confirmed = True
            nans_left = df.isna().sum().sum()
            st.session_state.nan_summary = ("\n- Removed all NaN values. "
                                            "\n - Leaving you with "
                                            f"{nans_left}"
                                            " NaN values.")
            warning_placeholder.empty()
            option_placeholder.empty()
            confirmation_placeholder.empty()
        else:
            st.stop()

    elif option == "keep":
        confirmation = confirmation_placeholder.button("Confirm keeping "
                                                       "NaN values")
        if confirmation:
            confirmation_placeholder.success("Keeping NaN values.")
            st.session_state["modified_df"] = df  # store the modified df
            st.session_state.nan_handling_confirmed = True
            nans_left = df.isna().sum().sum()
            st.session_state.nan_summary = ("\n- Kept all NaN values."
                                            "\n- Leaving you with "
                                            f"{nans_left}"
                                            " NaN values.")
            warning_placeholder.empty()
            option_placeholder.empty()
            confirmation_placeholder.empty()
        else:
            st.stop()

    elif option == "interpolate":
        confirmation = confirmation_placeholder.button("Confirm "
                                                       "linear interpolation")
        if confirmation:
            confirmation_placeholder.success("Interpolating NaN values.")
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
            df = df.dropna().reset_index(drop=True)  # remove any leftovers NAs
            st.session_state["modified_df"] = df  # store the modified df
            st.session_state.nan_handling_confirmed = True
            nans_left = df.isna().sum().sum()
            st.session_state.nan_summary = (f"\n- Interpolated NaN values "
                                            "linearly. \n- Leaving you with"
                                            f" {nans_left} NaN values.")
            warning_placeholder.empty()
            option_placeholder.empty()
            confirmation_placeholder.empty()
        else:
            st.stop()

    elif option == "fill with 0":
        confirmation = confirmation_placeholder.button("Confirm filling NaN "
                                                       "values with 0")
        if confirmation:
            confirmation_placeholder.success("Filling NaN values with 0.")
            df = df.fillna(0)
            st.session_state["modified_df"] = df  # store the modified df
            st.session_state.nan_handling_confirmed = True
            nans_left = df.isna().sum().sum()
            st.session_state.nan_summary = ("\n- Filled all NaN values"
                                            " with 0."
                                            "\n- Leaving you with "
                                            f"{nans_left}"
                                            " NaN values.")
            warning_placeholder.empty()
            option_placeholder.empty()
            confirmation_placeholder.empty()
        else:
            st.stop()

# if we have handled the nan values, show a summary
if st.session_state.nan_handling_confirmed:
    st.write(f"#### NaN Handling Summary: {st.session_state.nan_summary}")

    # prompt user to do different handling if they want!
    if st.button("Redo with a different method of NaN Handling"):
        st.session_state.nan_handling_confirmed = False
        del st.session_state["modified_df"]  # reset the data
        st.rerun()

features = list(df.columns)

# st.write("Data types after conversion:")
# st.write(df.dtypes) - useful to check if df is changed/same

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
model_mapping = get_model_parameter_mapping()

if task_type == "Classification":
    model_choices = ["K Nearest Neighbours", "Support Vector Machine",
                     "Multiple Logistic Regression"]
    if df.isna().sum().sum() > 0:
        model_choices = ["Support Vector Machine"]
        st.warning(
            "As there are still NA values, model selection is restricted")
else:
    model_choices = ["Multiple Linear Regression", "Lasso",
                     "XGBoost Regressor"]
    if df.isna().sum().sum() > 0:
        model_choices = ["XGBoost Regressor"]
        st.warning(
            "As there are still NA values, model selection is restricted")

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
if st.session_state.nan_handling_confirmed:
    st.write(f"NaN Handling Summary: {st.session_state.nan_summary}")

# Step 7: Train Pipeline
if st.button("Train Pipeline"):

    # add error exc?
    results = pipeline.execute()
    # not working atm

    st.success("Training complete!")
    st.write("Evaluation Results:")
    # "<autoop.core.ml.metric.RootMeanSquaredError object at 0x000002D36E24D150>"
    metric_pattern = re.compile(r"^.*metric\.([A-Za-z]+).*$")

    for metric_result in results["metrics"]:
        metric_name = re.findall(metric_pattern, metric_result[1])  # change to actually find the pattern name
        st.write(f"Metric: {metric_name}")
        st.write(f"{metric_result[0]} {metric_result[2]:.5f}")
        st.write(f"{metric_result[3]} {metric_result[5]:.5f}")

    predictions = results["predictions"]
    prediction_results = pd.DataFrame(predictions)
    st.write(predictions)
