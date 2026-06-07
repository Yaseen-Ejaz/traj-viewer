# simple-trajectory-parser

Use script.py to parse trajectories that you need.

The classifiers folder is to hold modules that script uses to classify parts of code if needed.

## Framework/Concept

- Target file in patch, use string search to find relevant trajectory messages that touch target file in patch
  - saves the trajectory immediately before and after the found instance. (let this be configurable)
- Have modules to implement different parsers for different agent trajectory formats
- outputs a json file

## To Run Apps

- `streamlit run gui/app.py`

## For multiple keywords

- `streamlit run trajectory_viewer.py`

## Notes

- All the GUIs are in the gui folder, the GUI is vibe coded, vibe code additional GUIs or modify (but don't break functionality plz) at your pleasure.
