from .agent import ObjDetectAgent
from .tools import yolo, yoloE, yoloE_promptFree
from langchain_core.messages import HumanMessage

def main() -> None:
    config = {
        "model": "lfm3-omni",
        "tools": [yolo, yoloE, yoloE_promptFree],
        "system_prompt": "No matter the user message, respond with 'testing...'"
    }
    agent = ObjDetectAgent(config)
    msg = HumanMessage(content="Hello!")
    print(agent.invoke(msg))