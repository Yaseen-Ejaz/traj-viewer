from typing import Any, Dict, List
import os
import json


class AgentParser:
    def __init__(self, agent_run_id: str):
        """
        agent_run_id is the name of the run ex: 20250425_Refact_Agent
        """
        self.directoryPath = None
        self.agent_run_id = None

    def setHomeDirectory(self, homeDirectory: str) -> None:
        self.directoryPath = homeDirectory
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

    def __init__(self):
        pass

    def _toolCallToString(self, tool_call: dict) -> str:
        """Takes the tool_call in assistant dictionary and converts it
        to a readable string

        tool_call has a format as so:

        {
            "id": "toolu_01Xc9dkQZqWPT3t51fiKxmuw",
            "function": {"arguments": "{}", "name": "tree"},
            "type": "function",
        }
        """
        out = f"""
        ID:
            {tool_call[id]}
        FUNCTION:
            ARGUMENTS:
                {tool_call["function"]["argument"]}
            NAME:
                {tool_call["function"]["name"]}
        TYPE:
            {tool_call["type"]}
        """
        return out

    def getTrajectories(self) -> List[Dict]:
        """
        It is returned in this format so that it is ready as a panda obj
        """
        trajPaths = os.path.join(self.directoryPath, "trajs")
        trajectories = {
            "agent_run_id": [],
            "instance_id": [],
            "role": [],
            "content": [],
            "tool_calls": [],
            "finish_reason": [],
            "tool_call_id": [],
        }
        for trajPath in os.listdir(trajPaths):
            with open(os.path.join(trajPaths, trajPath), "r") as fl:
                tempTraj = json.loads(fl.read())
                trajectories["agent_run_id"].append(self.agent_run_id)
                trajectories["instance_id"].append(os.path.normpath(trajPath))
                trajectories["role"].append()
                trajectories["content"].append()
                trajectories["tool_calls"].append()
                trajectories["finish_reason"].append()
                trajectories["tool_call_id"].append()
            fl.close()
        return trajectories

    def getPatches(self) -> List[str]:
        diffPaths = os.path.join(self.directoryPath, "logs")
        diffs = []
        for diff in os.listdir(diffPaths):
            with open(
                os.path.join(diffPaths, os.path.join(diff, "patch.diff")), "r"
            ) as fl:
                diffs.append(fl.read())
        return diffs
