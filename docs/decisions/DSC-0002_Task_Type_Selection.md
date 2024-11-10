# DSC-0001: Task Type Selection 
# Date: 2024-10-09
# Decision: Allow the user to change the automatically selected task type
# Status: Accepted
# Motivation: Datasets tend to be formatted in multiple ways. For example, the labels can be misformatted and encoded via e.g. floats, leading to our software detecting those as continous labels and selecting the task as regression. In that case, the user would have to know how our pipeline works and format the dataset himself in an appropriate way. We do not want that, hence decided to add a button that informs the user that misuing it will cause the system to fail. In that case, the user can click the button again to go back to the original task.
# Reason: Robustness of software. In case our automated checks go wrong, the user can decide for himself what task should the model perform
# Limitations: Relies on user responsibility.
# Alternatives: Find heuristics and logical statements that will automatically detect any misformatted data type and properly classify it.