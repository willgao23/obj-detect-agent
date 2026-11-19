from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
import os

load_dotenv()
HF_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

class ObjDetectAgent():
    """Agent for dynamically detemining the appropriate object detection """

    def __init__(self, config) -> None:
        """Initialize an agent from a config dict"""
        params = set(["model", "tools", "system_prompt"])
        assert set(config.keys()).issuperset(params)
        model = init_chat_model(model=config["model"], model_provider="huggingface")
        agent = create_agent(model=model, tools=config["tools"], system_prompt=config["system_prompt"])
        self.agent = agent
