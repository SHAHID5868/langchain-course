from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent, tools_condition
from langchain.tools import tool


load_dotenv()

@tool
def search(query: str) -> str:
    """Search the web for given query and return the results.
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching the web for {query}")
    return "Tokyo is sunny today"
    

tools = [search]
llm = ChatOllama(model="qwen2.5:3b",model_provider="ollama")
agent = create_react_agent(tools=tools, model=llm)

def main():
    response = agent.invoke({"messages":[HumanMessage(content="What is the weather in Tokyo")]})
    print(response["messages"][-1].content)

if __name__ == "__main__":
    main()

