from dataclasses import field
from pydoc import describe
from textwrap import indent
from langchain import tools
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from typing import List, Literal, Annotated
from pydantic import BaseModel, Field, HttpUrl
from dotenv import load_dotenv
from langchain_tavily import TavilySearch


load_dotenv()

class source(BaseModel):
    """Schema links of source from which agent will return results"""
    url: HttpUrl = Field(description="links of source for the query")

class response(BaseModel):
    """Format in which agent will return the response"""

    answer: str = Field(description="response from the agent")
    sources: List[source] = Field(default_factory=list, description="URL links that agent used for the query response")

tools = [TavilySearch()]
llm = ChatGroq(temperature=0,model="llama-3.3-70b-versatile")

def main():
    agent= create_react_agent(tools=tools, model=llm, response_format=response,prompt="Always return the query using Tools, don't answer the query with out proper search or thinking")
    result = agent.invoke({"messages":[HumanMessage(content="what is the weather in tokyo?")]})
    print(result["structured_response"].model_dump_json(indent=2))
    
    
if __name__ == "__main__":  # ✅ fix 3
        main()