import json
from modules import *
import os


def temp():
    x = RefactAI(os.getcwd() + "/20250425_Refact_Agent")
    gg = x.getTrajectories()
    print(len(gg))
    print(gg[0])


temp()
