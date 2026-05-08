from urllib import response
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from typing import List

from pydantic.types import T

from React_Langgraph_fun_call import AGENT_REASON
from nodes import SYSTEM_MESSAGE

load_dotenv()

@tool
def curreny_converter(currency: float) -> float:
    """Schema for converting the currency received to Eur with standard rate"""

    return round(currency * 0.85, 2)

@tool
def rating_avg(rating: List[float]) -> float:
    """schema for calculating the average rating for the list of ratings received."""

    avg_rating = round(sum(rating)/len(rating))

    return avg_rating

tools = [TavilySearch(max_results=4, search_depth="advanced"),curreny_converter, rating_avg]
llm = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq", temperature=0).bind_tools(tools)


SYSTEM_MESSAGE = """
     You are assitant who searched for Movie rating from various webistes and box-office of that movie.
     """ 
def run_search_agent(state:MessagesState):
    """You are assitant that will search for query and give results"""
    res = llm.invoke([
        SystemMessage(content=SYSTEM_MESSAGE),
        *state["messages"]
    ])
    return {"messages":[res]}

tool_node = ToolNode(tools)

ACT = "act"
AGENT_REASON = "agent_reason"
LAST = -1

def should_continue(state:MessagesState):
    last_message = state["messages"][LAST]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return ACT
    return END

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_search_agent)
flow.add_node(ACT, tool_node)
flow.set_entry_point(AGENT_REASON)
flow.add_conditional_edges(AGENT_REASON,should_continue,path_map={END:END, ACT:ACT})
 

app = flow.compile()

print(app.get_graph().draw_mermaid())

if __name__ == "__main__":
    print("Hello people, i am your movie rating agent")
    response = app.invoke({"messages":[ 
    HumanMessage(content="""What is the box office earnings of Inception? convert it to British pounds and also what is the average rating of these scores: 8.5, 9.0, 7.5, 8.0""")]})
    print(response["messages"][LAST].content)
