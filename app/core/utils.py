"""
Module for storing multiple utilities necessary for clean code in the pages
"""

import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io


from autoop.core.ml.model.classification import MultipleLogisticRegressor
from autoop.core.ml.model.classification import SVMClassifier
from autoop.core.ml.model.classification import KNearestNeighbors
from autoop.core.ml.model.regression import MultipleLinearRegression, Lasso
from autoop.core.ml.model.regression import XGBRegressor
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.feature import Feature
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import get_metric
from app.core.system import AutoMLSystem


def get_model_parameter_mapping() -> dict:
    """
    Returns a dictionary mapping model names to model classes and
    hyperparameters.
    :returns: A dictionary mapping model names to model classes and
    hyperparameters.
    """
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


def reset_pipeline() -> None:
    """ Clears the session state to reset the pipeline. """
    if "original_df" in st.session_state:
        del st.session_state["original_df"]
    if "modified_df" in st.session_state:
        del st.session_state["modified_df"]
    if "nan_handling_confirmed" in st.session_state:
        del st.session_state["nan_handling_confirmed"]
    if "nan_summary" in st.session_state:
        del st.session_state["nan_summary"]
    if "training_done" in st.session_state:
        del st.session_state["training_done"]
    if "pipeline" in st.session_state:
        del st.session_state["pipeline"]
    if "model_choices" in st.session_state:
        del st.session_state["model_choices"]
    if "task_type" in st.session_state:
        del st.session_state["task_type"]
    for remaining_key in st.session_state.keys():  # just to make sure
        del st.session_state[remaining_key]
    st.rerun()


def select_dataset(automl) -> pd.DataFrame | None:
    """
    Selects a dataset from the AutoML system and returns it.
    :param automl: The AutoML system instance
    :returns: The selected dataset or None if not selected
    """
    st.subheader("1. Select a Dataset")
    datasets = automl.registry.list(type="dataset")
    if datasets:
        dataset_names = [dataset.name for dataset in datasets]
        selected_dataset_name = st.selectbox("Select a dataset", dataset_names)
        selected_dataset = next(
            ds for ds in datasets if ds.name == selected_dataset_name
        )
        return selected_dataset
    else:
        st.warning("No datasets available. Please upload one.")
        st.stop()


def try_convert(value: str) -> float | str:
    """
    Converts values to numeric if possible, otherwise returns as is
    :param value: The value to be converted
    :returns: The converted value or the original value if conversion fails
    """
    try:
        return pd.to_numeric(value, errors='raise')
    except ValueError:
        return value


def handle_nan_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Handle NaN values based on user input.
    :param df: The dataset
    :returns: A tuple containing the modified dataframe and a dictionary
    containing the summary of NaN handling
    """
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


def select_features_and_target(df: pd.DataFrame) -> tuple[list[Feature],
                                                          Feature]:
    """
    Selects features and target from the dataset.
    :param df: The dataset
    :returns: A tuple containing the selected features and target
    """
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

    return [Feature(name=feat, type='categorical' if df[
        feat].dtype == 'object' else 'numerical'
    ) for feat in input_features], \
        Feature(name=target_feature, type='categorical' if
                df[target_feature].dtype == 'object' else 'numerical')


def revert_task_type(task_type) -> None:
    """
    Reverts the model task type
    :param task_type: The task type
    """
    st.session_state.training_done = False
    if task_type == "Classification":
        st.session_state.task_type = "Regression"
        if len(st.session_state.model_choices) == 1:
            st.session_state.model_choices = ["XGBoost Regressor"]
        else:
            st.session_state.model_choices = ["Multiple Linear Regression",
                                              "Lasso",
                                              "XGBoost Regressor"]
    else:
        st.session_state.task_type = "Classification"
        if len(st.session_state.model_choices) == 1:
            st.session_state.model_choices = ["K Nearest Neighbours"]
        else:
            st.session_state.model_choices = ["K Nearest Neighbours",
                                              "Support Vector Machine",
                                              "Multiple Logistic Regression"]


def choose_model(df, task_type) -> AutoMLSystem:
    """
    Selects a model based on user input.
    :param df: The dataset
    :param task_type: The task type
    :returns: The selected model
    """
    st.subheader("3. Choose Model Type")
    model_mapping = get_model_parameter_mapping()
    if df.isna().sum().sum() > 0:
        st.warning("As there are still NA values, model "
                   "selection is restricted")
        model_choices = [" None",] \
            if st.session_state.task_type == "Classification" else \
                        ["XGBoost Regressor"]
        if model_choices[0] == " None":
            st.warning("There are no models avaliable that can "
                       "handle missing values with classification."
                       " Please choose different task type with "
                       "the features (detect task type) or reset pipeline.")
            st.stop()
    else:
        model_choices = [
            "K Nearest Neighbours", "Support Vector Machine",
            "Multiple Logistic Regression"
        ] if st.session_state.task_type == "Classification" else [
            "Multiple Linear Regression", "Lasso",
            "XGBoost Regressor"]

    st.session_state.model_choices = model_choices
    st.write("Choose the model from the dropdown below.")
    st.info("If you believe the automatically detected task is wrong, \
             click the button below.")
    if st.button("Switch model tasks"):
        st.warning("Misspecified model task may cause the pipeline to fail. "
                   "Please choose the correct task type.")
        revert_task_type(task_type)

    model_choice = st.selectbox("Select a model",
                                st.session_state.model_choices)
    selected_model_class, hyperparams = model_mapping[model_choice]

    chosen_params = {param_name: input_fn(label, *args) for param_name,
                     (input_fn, label, *args
                      ) in hyperparams.items()} if hyperparams else {}
    return selected_model_class(**chosen_params)


def set_train_test_split() -> float:
    """ Sets the train-test split based on user input. """
    st.subheader("4. Set Train-Test Split")
    return st.slider("Training set percentage", min_value=0.1, max_value=0.9,
                     value=0.8)


def select_metrics(task_type) -> list:
    """
    Selects evaluation metrics based on user input.
    :param task_type: The task type
    :returns: The selected metrics
    """
    st.subheader("5. Choose Evaluation Metrics")
    metrics = ["Accuracy", "Recall", "Precision", "F1"
               ] if task_type == "Classification" else [
                   "MSE", "RMSE", "R2"]
    selected_metrics = st.multiselect("Select metrics", metrics)

    if not selected_metrics:
        st.warning("Please select at least one metric.")
        st.stop()

    return [get_metric(name) for name in selected_metrics]


def display_pipeline_summary(pipeline) -> None:
    """
    Displays the pipeline summary.
    :param pipeline: The pipeline
    """
    st.subheader("Pipeline Summary")
    st.write(str(pipeline))


def train_pipeline(pipeline) -> dict:
    """
    Trains the pipeline and reports the results.
    :param pipeline: The pipeline
    """
    try:
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
            st.write("**Predictions:**")
            predictions = results["predictions"]
            prediction_results = pd.DataFrame(predictions, columns=[
                "Predicted Values"])
            st.write(prediction_results)

            return results
    except Exception as e:
        st.error(f"Training failed, please reset pipeline: {e}")


def serialize_pipeline_data(pipeline: Pipeline) -> bytes:
    """
    Serializes the pipeline data into a byte stream.
    :param pipeline: The pipeline to serialize
    :returns: The serialized data (bytes)
    """
    data = {
        "artifacts": dict((artifact.name, artifact.data)
                          for artifact in pipeline.artifacts),
        "model": pipeline.model,
        "metrics": pipeline._metrics
    }

    serialized_data = pickle.dumps(data)

    return serialized_data


def save_pipeline(automl: AutoMLSystem, pipeline: Pipeline) -> None:
    """
    Save the pipeline as an artifact.
    :param automl: The AutoMLSystem instance
    :param pipeline: The pipeline to save
    """
    st.subheader("6. Save Pipeline")
    st.write("Here you may save your pipeline for future use.")
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
                st.success(f"Pipeline '{pl_name}' saved successfully!")
            else:
                st.warning("Please enter a name and version for the pipeline.")


def select_pipeline(automl: AutoMLSystem) -> Pipeline:
    """
    Load a pipeline from the registry.
    :param automl: The AutoMLSystem instance
    :param pl_name: The name of the pipeline to load
    :param pl_version: The version of the pipeline to load
    :returns: The loaded pipeline
    """
    st.subheader("1. Select a Pipeline")
    pipelines = automl.registry.list(type="pipeline")
    if pipelines:
        pipeline_names = [pipeline.name for pipeline in pipelines]
        selected_pipeline_name = st.selectbox("Select a dataset",
                                              pipeline_names)
        selected_pipeline = next(
            ds for ds in pipelines if ds.name == selected_pipeline_name
        )
        return selected_pipeline
    else:
        st.warning("No pipelines available. Please train one and save it!")
        st.stop()


def load_pipeline(selected_pipeline: Artifact) -> Pipeline:
    """
    Load a pipeline from the registry.
    :param selected_pipeline: The pipeline artifact to load
    :returns: The loaded pipeline
    """
    deserialized_data = pickle.loads(selected_pipeline.data)
    artifacts = deserialized_data['artifacts']
    pipeline_config = pickle.loads(artifacts['pipeline_config'])

    # Extract configuration details from pipeline_config
    input_features = pipeline_config['input_features']
    target_feature = pipeline_config['target_feature']
    split = pipeline_config['split']

    print("Input Features:", input_features)
    print("Target Feature:", target_feature)
    print("Split:", split)

    # Initialize the Pipeline object with the deserialized configuration
    loaded_pipeline = Pipeline(
        input_features=input_features,
        target_feature=target_feature,
        split=split,
        model=deserialized_data['model'],
        metrics=deserialized_data['metrics'],
        dataset=None
    )

    return loaded_pipeline


def predict_pipeline(pipeline: Pipeline) -> pd.DataFrame:
    """
    Predict using the loaded pipeline.
    :param pipeline: The pipeline to use for prediction
    :returns: The predicted values
    """
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    st.warning("The CSV file must have the same target features"
               " and input features as the original pipeline."
               " Please make note of the pipeline summary "
               "for this respective features if unsure.")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        asset_path = f"csv_files/{uploaded_file.name}"
        version = "ohio"

        # Create a new Dataset instance
        new_dataset = Dataset.from_dataframe(
            name=uploaded_file.name,
            data=data,
            asset_path=asset_path,
            version=version
        )

        try:
            pipeline.dataset = new_dataset
            results = pipeline.execute()
            st.success("Prediction completed successfully!")
            return results
        except Exception as e:
            st.error("Error! Make sure your dataset has "
                     f"the same input and target features: {str(e)}")


def generate_experiment_report(results: dict) -> None:
    """
    Generates a detailed experiment report with graphs,
    metrics, and other relevant information.
    :param results: A dictionary containing the results
    from the trained pipeline.
    """
    if not results:
        st.warning("No results available to generate the report.")
        return

    # Section 1: Display Metrics
    st.subheader("Evaluation Metrics")
    if "metrics" in results:
        metrics_data = [
            (metric[0], metric[1].__class__.__name__, metric[2], metric[3],
             metric[4], metric[5])
            for metric in results["metrics"]
        ]
        metrics_df = pd.DataFrame(metrics_data, columns=["Phase",
                                                         "Metric Name",
                                                         "Train Score",
                                                         "Test Phase",
                                                         "Metric Object",
                                                         "Test Score"])

        for _, row in metrics_df.iterrows():
            st.write(f"**{row['Metric Name']}**")
            st.write(f"- {row['Phase']} Train Score: {row['Train Score']:.5f}")
            st.write(f"- {row['Test Phase']} "
                     f"Test Score: {row['Test Score']:.5f}")

        # Allow metrics to be downloaded
        metrics_csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="Download Metrics Report as CSV",
            data=metrics_csv,
            file_name="metrics_report.csv",
            mime="text/csv"
        )

    # Section 2: Display Predictions
    st.subheader("Predictions Overview")
    if "predictions" in results:
        predictions = results["predictions"]
        prediction_df = pd.DataFrame(predictions, columns=["Predicted Values"])
        st.write(prediction_df.head())

        # Here we create a histogram of the predicted values
        fig, ax = plt.subplots()
        sns.histplot(prediction_df["Predicted Values"], kde=True, ax=ax)
        ax.set_title("Distribution of Predicted Values")
        st.pyplot(fig)

        # Allow it to be downloadable since user might want it
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png")
        img_buffer.seek(0)  #

        st.download_button(
            label="Download Prediction Distribution Plot as PNG",
            data=img_buffer,
            file_name="prediction_distribution.png",
            mime="image/png"
        )

        # Allow predicitions to be downloaded
        predictions_csv = prediction_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=predictions_csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
