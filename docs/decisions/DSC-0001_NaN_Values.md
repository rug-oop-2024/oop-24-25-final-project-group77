# DSC-0002: Handle NaN Values in Pipeline Training for Regression and Classification
# Date: 2024-11-09
# Decision: Handle NaN values in pipeline training for regression and classification tasks
# Status: Accepted
# Motivation: Many real-world datasets contain NaN values, and handling them is important for more realistic and robust model training.
# Reason: Instead of ignoring or removing NaN values from datasets, we opted to incorporate methods for handling them directly in the pipeline. This ensures that the model is trained to deal with missing or incomplete data, which reflects real-world conditions and improves generalization.
# Limitations: The pipeline may become more complex, and certain techniques may not perform well if NaN values are too prevalent.
# Alternatives: Imputation methods (e.g., mean/median imputation), removing rows or columns with NaN values, only using algorithms that natively handle missing data (e.g., some tree-based models).