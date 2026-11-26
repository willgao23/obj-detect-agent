from dataclasses import dataclass
from langchain_ollama import ChatOllama
import json
from asyncio import Semaphore
from typing import TextIO
from langchain_core.messages import HumanMessage, AIMessage

@dataclass
class Context:
    b64: str
    isStub: bool

class ObjDetectAgent():
    """Local LLM-based agent for dynamically detemining the appropriate object detection algorithm"""

    def __init__(self, config: object) -> None:
        """Initialize an agent from a config dict"""
        params = set(["model", "tools", "system_prompt"])
        assert set(config.keys()).issuperset(params)
        llm = ChatOllama(model=config["model"], num_predict=256)
        self.llm_with_tools = llm.bind_tools(config["tools"])
        self.system_prompt=config["system_prompt"]

    async def ainvoke_tool_eval(self, msg: HumanMessage, sem: Semaphore, out: TextIO) -> AIMessage:
        """Invokes the LLM and returns the first tool routing decision made"""
        async with sem:
            ai_msg = await self.llm_with_tools.ainvoke([self.system_prompt, msg])
        out.write(json.dumps(ai_msg.tool_calls[0]) + "\n")
        out.flush()
        return ai_msg
