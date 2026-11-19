from langchain.agents import create_agent
from dotenv import load_dotenv
import os

load_dotenv()
HF_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def test_tool(x: str) -> str:
    """Stub tool for initial agent config"""
    return f"test_tool was called with param: {x}"

def main() -> None:
    """Instantiates and runs the object detection agent"""
    model="huggingface:microsoft/Phi-3-mini-4k-instruct"
    tools=[test_tool]
    system_prompt="You are a helpful assistant"
    agent = create_agent(model=model, tools=tools, system_prompt=system_prompt)
