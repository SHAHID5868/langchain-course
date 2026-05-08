from typing_extensions import Format
from dotenv import load_dotenv
from langchain_core.    tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()

class NewsResponse(BaseModel):
    topic: str =Field(description="Expalin what is the topic that was searched in web")
    summary: str = Field(description="Give the summary of the news that was extracted")
    sentiment: str = Field(description="Give the sentiement if it is 'positive' 'negative' or 'neutral' ")
    sentiment_score: float = Field(description="give us the score for the sentiment")
    top_words: list[str] = Field(description="Give us the top word that is repitative from the news")

@tool
def sentiment(rating: str) -> float:
    """Schema for calulcating the score of sentiment that will be passed as "positive"
    "negative", "neutral"""
    score = {"positive": 1.0, "neutral": 0.5, "negative": 0.2}

    return score.get(rating, 0)

tools = [TavilySearch(max_results=2, search_depth ="basic"), sentiment]

llm= init_chat_model(model="gemini-2.0-flash", model_provider="google_genai", temperature=0, ).bind_tools(tools)

SYSTEM_MESSAGE = """
   you are helping assistent the will take news from web and give it a rating according to the sentiment that is "positive", "negative" and "neutral"
   according to the sentiment give a rating between 0-1, postive the highest and negative the lowest"""

def run_sentiment(state: MessagesState):
    res = llm.invoke([
        SystemMessage(content=SYSTEM_MESSAGE),
        *state["messages"]
    ])
    return {"messages": [res]}

def format_response_node(state: MessagesState):           # ✅ new format node
    """Structures final output into NewsResponse"""
    llm_structured = init_chat_model(
        model="llama-3.3-70b-versatile",
        model_provider="groq",
        temperature=0
    ).with_structured_output(NewsResponse)

    result = llm_structured.invoke([
        SystemMessage(content="Based on the conversation, extract and structure the final response"),
        *state["messages"]
    ])
    return {"messages": [AIMessage(content=result.model_dump_json(indent=2))]}


tool_node = ToolNode(tools)

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1
FORMAT = "format"

flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_sentiment)
flow.add_node(FORMAT, format_response_node)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

def should_continue(state:MessagesState):
    if not state["messages"][LAST].tool_calls:
        return FORMAT
    return ACT

flow.add_conditional_edges(AGENT_REASON, should_continue, path_map={FORMAT:FORMAT, ACT:ACT})
flow.add_edge(ACT,AGENT_REASON)
flow.add_edge(FORMAT, END)

app = flow.compile()

if __name__ == "__main__":
    responde = app.invoke({"messages":
    [
        HumanMessage(content="""Search for latest news about Tesla, 
 analyze the sentiment of the results 
 and find the top 3 most frequent words""")
    ]})
    print(responde["messages"][LAST].content)
