# DSC-0001: Reset Pipeline Button
# Date: 2024-11-09
# Decision: Add a reset pipeline button on top of the page that resets the pipeline and user input history.
# Status: Accepted
# Motivation: As the user might want to train multiple pipelines in an easy way, we added this button to allow for a behavior similar to a full site refresh, removing all of the inputs and session states of the user. This improves user experience as it allows them to train multiple pipelines without ever leaving the site.
# Reason: Improving user experience.
# Limitations: Again, requires the user to click the button, not done in an automatic manner.
# Alternatives: Timeout-based resetting or resetting only after a pipeline is saved.