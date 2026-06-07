from typing import Any, Dict, List
import os
import json
import copy
import unidiff
from classifiers.patch_file_type_classifier import classify_patch_file
from classifiers.deterministic_rq2_labeler import classify_patch_test_file


class AgentParser:
    def __init__(self, agent_run_id: str):
        """
        agent_run_id is the name of the run ex: 20250425_Refact_Agent
        """
        self.directoryPath = None
        self.agent_run_id = None
        self.agent_type = None

    def setHomeDirectory(self, homeDirectory: str) -> None:
        self.directoryPath = os.path.join(os.getcwd(), homeDirectory)
        self.agent_run_id = os.path.normpath(self.directoryPath)

    def findInstances(self, toSearch: str) -> Dict:
        """Finds trajectory instances that match the string

        Args:

        Returns:

        """
        raise NotImplementedError

    def getPatches(self) -> List[str]:
        raise NotImplementedError

    def getFiles(self) -> List[Dict]:
        """Returns a list of dictionaries where each dictionary has
        the following structure {"instance_id":str, "fileTouched":str}

        """
        raise NotImplementedError

    def getTrajectories(self) -> List[Dict]:
        raise NotImplementedError

    def streamlitTrajectoryBrowser(self):
        raise NotImplementedError


class RefactAI(AgentParser):
    """
    128k token context window claude
    o4-mini 100k token context window

    accepted actions:
        addition
        deletion
        any



    USAGE:
    example:
    rf = RefactAI()
    rf.setHomeDirectory("20250425_Refact_Agent")

    20250425_Refact_Agent must be in project folder

    rf.output_data("deletion")


    Trajectory format is as follows:
    [
            {
                "role": "system",
                "content": "",
                "finish_reason": "",
                "tool_call_id": "",
            },
            {
                "role": "user",
                "content": "",
                "finish_reason": "",
                "tool_call_id": "",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "toolu_01Xc9dkQZqWPT3t51fiKxmuw",
                        "function": {"arguments": "{}", "name": "tree"},
                        "type": "function",
                    }
                ],
                "finish_reason": "stop",
                "tool_call_id": "",
            },
            {
                "role": "tool",
                "content": "",
                "finish_reason": "",
                "tool_call_id": "toolu_01Xc9dkQZqWPT3t51fiKxmuw",
            },
        ]

    """

    def __init__(
        self,
        mainLLMContextWindowSize: int = 128000,
        assistantLLMContextWindowSize: int = 100000,
        mainModel: str = "claude-3-7-sonnet-20250219",
        assistantModel: str = "gpt-4o-mini",
    ):
        self.agent_type = "refactAI"

    def getPatches(self) -> List[str]:
        diffPaths = os.path.join(self.directoryPath, "logs")
        diffs = []
        for diff in os.listdir(diffPaths):
            with open(
                os.path.join(diffPaths, os.path.join(diff, "patch.diff")), "r"
            ) as fl:
                diffs.append(fl.read())
            fl.close()
        return diffs

    def getTrajectories(self) -> Dict[Any]:
        """
        key is instance_id of the trajectory
        value is a string representation of the trajectory json file

        returns a dictionary, key is instance_id, value is json string
        """
        trajPaths = os.path.join(self.directoryPath, "trajs")
        trajectories = {}

        for trajPath in os.listdir(trajPaths):
            with open(os.path.join(trajPaths, trajPath), "r") as fl:
                tempTraj = fl.read()
            fl.close()
            trajectories[f"{os.path.basename(trajPath)[:-5]}"] = tempTraj
        return trajectories

    def _objToJsonlSaveToDisk(self, data: list[dict], output_name: str):
        with open(
            os.path.join(os.getcwd(), "data/" + output_name), "w", encoding="utf-8"
        ) as fl:
            for item in data:
                fl.write(json.dumps(item) + "\n")

    def getInteractionNumbers(
        self,
        trajectory: List[Dict[Any]],
        patch: unidiff.PatchFile,
        action: str,
        keyword: str,
    ) -> List[int]:
        """
        accepted actions:
        addition
        deletion
        any

        """

        interactionNumbers = set()
        if action == "deletion":
            for idx, interaction in enumerate(trajectory):
                if "content" not in interaction:
                    interaction["content"] = "NONE"
                if patch.removed > 0 and keyword in interaction["content"]:
                    if idx > 0:
                        interactionNumbers.add(idx - 1)
                    interactionNumbers.add(idx)
                    if idx < (len(trajectory) - 1):
                        interactionNumbers.add(idx + 1)
        elif action == "any":
            for idx, interaction in enumerate(trajectory):
                if "content" not in interaction:
                    interaction["content"] = "NONE"
                if keyword in interaction["content"]:
                    if idx > 0:
                        interactionNumbers.add(idx - 1)
                    interactionNumbers.add(idx)
                    if idx < (len(trajectory) - 1):
                        interactionNumbers.add(idx + 1)
        elif action == "addition":
            for idx, interaction in enumerate(trajectory):
                if "content" not in interaction:
                    interaction["content"] = "NONE"
                if patch.added > 0 and keyword in interaction["content"]:
                    if idx > 0:
                        interactionNumbers.add(idx - 1)
                    interactionNumbers.add(idx)
                    if idx < (len(trajectory) - 1):
                        interactionNumbers.add(idx + 1)

        return list(interactionNumbers)

    def getInteractionStrFromInteractionNumbers(
        self, interactionNumbers: List[int], trajectory: List[Dict[Any]]
    ) -> List[str]:
        """
        IF context value does not have anything, then it is set to "NONE"

        returns a list of json strings corresponding to the interaction numbers

        """
        interactionStr = []
        for number in interactionNumbers:
            if "content" not in trajectory[number]:
                trajectory[number]["content"] = "NONE"
            interactionStr.append(json.dumps(trajectory[number]))
        return interactionStr

    def output_data(
        self,
        edit_type: str,
        outputJsonl: bool,
        jsonlFileName: str = "OUTPUT_DATA.jsonl",
    ) -> List[Dict[Any]]:
        """
        out is list of dict

        RETURNS:
        list of dictionary with following keys:

        "edit_type": str
            Type of edit done on patch, see class docstring for accepted types.
        "agent_type": str
            Name of agent.
        "agent_run_id": str
            Name of the trial in SWE bench.
        "instance_id": str
            Name of the problem in the SWE bench.
        "file_name": str
            Name of the file in the patch for the problem.
        "churn": int
        "deletions": int
        "additions": int
        "file_classification": str
            code, documentation, or test
        "test_class_deterministic_label": str
            The final deterministic label. This is empty when the result is
            flagged as requiring manual review.
        "test_class_suggested_label": str
            The label suggested by the deterministic decision tree.
        "test_class_confidence": str
            The confidence assigned by the original classification logic.
            One of: "high", "medium", or "low".
        "test_class_needs_review": bool
            Whether the classification should be manually reviewed.
        "test_class_review_reason": str
            Explanation for why the label was selected or why review is needed.
        "test_class_signals": str
            List of json string with the following format:
            The extracted deterministic signals used by the classifier, such as
            test additions, test deletions, assertion counts, expected-output
            line counts, standalone-file share, support-file share, and deletion
            share.
        "interaction_numbers": list[int]
            List of integers representing the interactions in the trajectory
            relevant to the edit type. It contains the interaction prior and
            preceding as well to give more context.
        "interactions_str": list[str]
            List of json string representing the interactions in interaction_numbers.
            The json strings have the following format:

        "file_str": str
            The string representation of the diff for the relevant file

        """
        out = []
        tempStructur2 = []

        row_template_01 = {
            "edit_type": "",
            "agent_type": "",
            "agent_run_id": "",
            "instance_id": "",
            "file_name": "",
            "churn": -999,
            "deletions": -999,
            "additions": -999,
            "passFail": None,
            "file_classification": "",
            "test_classification": [""],
            "interaction_numbers": [-999],
            "interactions_str": [""],
            "file_str": "",
        }

        row_template_02 = {
            "agent_type": "",
            "agent_run_id": "",
            "instance_id": "",
            "trajectory": [{}],
            "patch_diff_file": unidiff.PatchSet(""),
            "passFail": None,
            "retryAmount": -999,
        }

        trajectories = self.getTrajectories()

        # get diff file paths
        diffPaths = os.path.join(self.directoryPath, "logs")
        for diffPath in os.listdir(diffPaths):
            newRow = copy.deepcopy(row_template_02)
            # agent_type is given to us
            newRow["agent_type"] = self.agent_type

            # agent_run_id is the str of the run name
            newRow["agent_run_id"] = os.path.basename(self.agent_run_id)

            # instance_id is the name of the problem that the agent tried to solve
            newRow["instance_id"] = os.path.basename(diffPath)

            # trajectory is list of dictionary representing the trajectory in order of interactions (i.e. the trajectory)
            newRow["trajectory"] = json.loads(trajectories[newRow["instance_id"]])

            # patch diff file as a unidiff.PatchSet type
            with open(
                os.path.join(diffPaths, os.path.join(diffPath, "patch.diff")), "r"
            ) as fl:
                patch = unidiff.PatchSet(fl.read())
            fl.close()
            newRow["patch_diff_file"] = patch

            # passFail
            with open(
                os.path.join(diffPaths, os.path.join(diffPath, "report.json")), "r"
            ) as fl:
                report = json.loads(fl.read())
            fl.close()
            newRow["passFail"] = report[newRow["instance_id"]]["resolved"]

            tempStructur2.append(copy.deepcopy(newRow))

        for file in tempStructur2:
            for patched_file in file["patch_diff_file"]:
                newRow = copy.deepcopy(row_template_01)

                # edit type
                newRow["edit_type"] = edit_type

                # agent type
                newRow["agent_type"] = file["agent_type"]

                # agent run id
                newRow["agent_run_id"] = file["agent_run_id"]

                # instance_id
                newRow["instance_id"] = file["instance_id"]

                # file_name
                newRow["file_name"] = os.path.basename(patched_file.source_file)

                # churn
                newRow["churn"] = patched_file.added + patched_file.removed

                # deletions
                newRow["deletions"] = patched_file.removed

                # additions
                newRow["additions"] = patched_file.added

                # passFail
                newRow["passFail"] = file["passFail"]

                # file classification
                newRow["file_classification"] = classify_patch_file(patched_file)

                # test classification
                if newRow["file_classification"] == "test":
                    test_class = classify_patch_test_file(patched_file)
                    newRow["test_class_deterministic_label"] = test_class[
                        "deterministic_label"
                    ]
                    newRow["test_class_suggested_label"] = test_class["suggested_label"]
                    newRow["test_class_confidence"] = test_class["confidence"]
                    newRow["test_class_needs_review"] = test_class["needs_review"]
                    newRow["test_class_review_reason"] = test_class["review_reason"]
                    newRow["test_class_signals"] = json.dumps(test_class["signals"])
                else:
                    newRow["test_class_deterministic_label"] = ""
                    newRow["test_class_suggested_label"] = ""
                    newRow["test_class_confidence"] = ""
                    newRow["test_class_needs_review"] = ""
                    newRow["test_class_review_reason"] = ""
                    newRow["test_class_signals"] = ""

                # interaction_numbers
                newRow["interaction_numbers"] = self.getInteractionNumbers(
                    file["trajectory"],
                    patched_file,
                    "deletion",
                    newRow["file_name"],
                )

                # interactions_str
                newRow["interaction_str"] = (
                    self.getInteractionStrFromInteractionNumbers(
                        newRow["interaction_numbers"], file["trajectory"]
                    )
                )

                # file_str
                newRow["file_str"] = str(patched_file)  # str(file["patch_diff_file"])
            out.append(copy.deepcopy(newRow))
        if outputJsonl:
            self._objToJsonlSaveToDisk(out, jsonlFileName)
        return out


if __name__ == "__main__":
    rf = RefactAI()
    rf.setHomeDirectory("20250425_Refact_Agent")
    rf.output_data("any", True, "20250425_Refact_Agent_parsed_trajectories.jsonl")
    # print(rf.output_data("deletion", False)[1]["file_classification"])
