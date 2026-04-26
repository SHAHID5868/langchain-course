from pyexpat import model
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """Search the web for the given query and return the results."""
    print(f"Searching the web for: {query}")
    return tavily.search(query=query)


load_dotenv()

tools = [search]
llm = ChatOllama(model="qwen2.5:3b ",model_provider="ollama")
agent = create_react_agent(tools=tools,model=llm)



def main():
    print("Hello, World!")
    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
    print(result["messages"][-1].content)

if __name__ == "__main__":  # ✅ fix 3
    main()