from pyexpat import model
from textwrap import indent
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

load_dotenv()

 
class Source(BaseModel):
    """Schema for a source used by agent"""

    url:str = Field(description="The URL of the source")

class AgentReponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] =Field(default_factory=list, description="List of sources used to generate the answers")

tools = [TavilySearch()]
llm = ChatOllama(model="qwen2.5:3b ",model_provider="ollama")
agent = create_react_agent(tools=tools,model=llm, response_format=AgentReponse, prompt=SystemMessage(content="You must always use the search tool to answer questions. Never answer from your own knowledge. Always search first."))



def main():
    print("Hello, World!")
    result = agent.invoke({"messages": [HumanMessage(content="Search for 3 job postings for AI engineer using langchain in the bay area on Linkedin and list their details")]})
    print(result["structured_response"].model_dump_json(indent=2))

if __name__ == "__main__":  # ✅ fix 3
    main()