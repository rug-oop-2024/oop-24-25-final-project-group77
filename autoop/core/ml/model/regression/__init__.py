from autoop.core.ml.model.regression.multiple_linear_regression \
    import MultipleLinearRegression
from autoop.core.ml.model.regression.lasso import Lasso
from autoop.core.ml.model.regression.xgboost_regressor import XGBRegressor


REGRESSION_MODELS_DICT = {
    "MultipleLinearRegression": MultipleLinearRegression,
    "Lasso": Lasso,
    "XGBRegressor": XGBRegressor,
}  # added to pass style checks while being able to access models
