from pyexpat import model
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


load_dotenv()

tools = [TavilySearch()]
llm = ChatOllama(model="qwen2.5:3b ",model_provider="ollama")
agent = create_react_agent(tools=tools,model=llm)



def main():
    print("Hello, World!")
    result = agent.invoke({"messages": [HumanMessage(content="Search for 3 job postings for AI engineer using langchain in the bay area on Linkedin and list their details")]})
    print(result["messages"][-1].content)

if __name__ == "__main__":  # ✅ fix 3
    main()