Input:
    - folder of agent

Output:
    - jsonl with the following tags:
      - edit_type
        - type of lookup, pure_deletion, pure_addition, deletion, addition, replacement
      - agent_type
        - str, the type of agent
      - agent_run_id
        - str the run name
      - instance_id
        - str of the name of the problem
      - file_name
        - str of the edited file in the patch
      - churn
      - deletions
      - additions
      - test_classification
        - str of types made with Reza or NA which means not applicable
      - file_classification
        - code, doc, test
      - interaction_numbers
        - list of int
      - interactionsStr
        - list of str correspoinding to the interaction_numbers
      - fileStr
        - the file diff that was modified

Supporting data structure for output:
      - agent_type
        - str, the type of agent
      - agent_run_id
      - instance_id
      - trajectory
        - list of dict in order as in trajectory
      - patch_diff_file
        - str
      - passFail
        - bool


checklist:
    - output all the files 