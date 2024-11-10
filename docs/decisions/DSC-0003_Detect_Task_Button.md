# DSC-0001: Detect Task Button
# Date: 2024-11-09
# Decision: Add a detect task button instead of rerunning every time the user selects a new target feature.
# Status: Accepted
# Motivation: Every time the user chooses a new target feature in the st.multiselect, the whole page used to rerun leading to crashes in other parts of the form i.e. overwriting the model chosen. Adding the button allows the user to change the target multiple types (they can change their minds) and rerun the website in a controlled manner after the final selection is made, preventing any bugs. 
# Reason: Errors with automatic handling of multiple streamlit input items.
# Limitations: Requires the user to click the button anytime the target was changed - is not automatic and can reduce user satisfaction.
# Alternatives: Smart use of nested st.forms or any other alternatives 