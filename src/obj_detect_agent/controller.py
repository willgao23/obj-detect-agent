from .agent import ObjDetectAgent
from .tools import test_tool

def main() -> None:
    config = {
        "model": "phi4-mini",
        "tools": [test_tool],
        "system_prompt": "You are a helpful assistant"
    }
    agent = ObjDetectAgent(config)
    print(agent.agent)