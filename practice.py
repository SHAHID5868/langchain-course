from dotenv import load_dotenv
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from ollama import Tool
from langchain_tavily import TavilySearch
from urllib3 import response
from langchain_core.messages import HumanMessage,SystemMessage

load_dotenv()



tools = [TavilySearch()]
llm = ChatOllama(model="qwen2.5:3b")
agent = create_react_agent(tools=tools, model=llm, prompt=SystemMessage(content="You must always use the search tool to answer questions. Never answer from your own knowledge. Always search first."))

def main():
    print("Hello")
    response = agent.invoke( {"messages": [HumanMessage(content="What is the weather in Tokyo?")]}) 
    for message in response["messages"]:
        print(type(message).__name__, ":", message.content)



if __name__ == "__main__":
    main()

