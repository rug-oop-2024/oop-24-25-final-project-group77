"""
Module for storing multiple utilities necessary for clean code in the pages
"""


import streamlit as st
from autoop.core.ml.model.classification import MultipleLogisticRegressor
from autoop.core.ml.model.classification import SVMClassifier
from autoop.core.ml.model.classification import KNearestNeighbors
from autoop.core.ml.model.regression import MultipleLinearRegression, Lasso
from autoop.core.ml.model.regression import XGBRegressor


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
