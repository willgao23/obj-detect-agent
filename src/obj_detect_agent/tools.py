from langchain.tools import tool

@tool
def test_tool(x: str) -> str:
    """Stub tool for initial agent config"""
    return f"test_tool was called with param: {x}"