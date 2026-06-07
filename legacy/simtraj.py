import json

import numpy as np
import pandas as pd
from modules import RefactAI, AgentParser
from config import SUPPORTED_AGENTS, AGENT_OBJS
import config
import os
import streamlit as st
import random

st.set_page_config(
    page_title="Deletion Interaction Explorer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("Simple Trajectory Parser")
    st.sidebar.header("Configurations")

    chosenAgent = st.sidebar.selectbox("What agent are you using?", SUPPORTED_AGENTS)
    agentFolder = st.sidebar.text_input("Agent home directory", value="EnterPath")

    agentObj = AgentParser()

    if st.sidebar.button("Refresh"):
        agentObj = AGENT_OBJS[chosenAgent]


# jsonl_path_text = st.sidebar.text_input("JSONL path", value=str("test"))

if __name__ == "__main__":
    main()
