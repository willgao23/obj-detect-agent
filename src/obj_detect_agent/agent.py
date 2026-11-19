from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

class ObjDetectAgent():
    """Local LLM-based agent for dynamically detemining the appropriate object detection algorithm"""

    def __init__(self, config: object) -> None:
        """Initialize an agent from a config dict"""
        params = set(["model", "tools", "system_prompt"])
        assert set(config.keys()).issuperset(params)
        model = ChatOllama(model=config["model"])
        agent = create_agent(model=model, tools=config["tools"], system_prompt=config["system_prompt"])
        self.agent = agent

    def invoke(self, msg: HumanMessage) -> AIMessage:
        return self.agent.invoke(msg)
