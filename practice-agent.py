from json import load
import os
from langgraph.graph import END
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from nodes import SYSTEM_MESSAGE, tool_node
from react import tools, llm


load_dotenv()

@tool
def triple(num: float) -> float:
    """Schema for tripling the number received and return result"""
    return float(num * 3)


tools= [TavilySearch(max_results=2,search_depth="basic"),triple]

llm = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq", temperature=0).bind_tools(tools=tools)


SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""

def run_agent_node(state: MessagesState):
    """Schema for passing the ruuning the node"""
    response = llm.invoke([SystemMessage(content=SYSTEM_MESSAGE), *state["messages"]])

    return {"messages": [response]}

tool_node = ToolNode(tools)

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1

def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_node)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)
flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END: END,
    ACT: ACT
})

flow.add_edge(ACT,AGENT_REASON)
app = flow.compile()

if __name__ == "__main__":
    res = app.invoke({"messages":[HumanMessage(content="What is the temperature in tokyo? list it and triple it and after that give me 3 pointers on Elon musk.")]})

    print(res["messages"][LAST].content)

