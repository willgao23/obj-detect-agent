from langchain_ollama import ChatOllama
from langchain.agents import create_agent

class ObjDetectAgent():
    """Local LLM-based agent for dynamically detemining the appropriate object detection algorithm"""

    def __init__(self, config) -> None:
        """Initialize an agent from a config dict"""
        params = set(["model", "tools", "system_prompt"])
        assert set(config.keys()).issuperset(params)
        model = ChatOllama(model=config["model"])
        agent = create_agent(model=model, tools=config["tools"], system_prompt=config["system_prompt"])
        self.agent = agent
