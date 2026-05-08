from dotenv import load_dotenv
from langchain_core.prompts import prompt
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num: float) -> float:
    """
    param num: a number to be tripled
    returns: the triple of the input nnumber
    """
    print(f">>>Executing triple with {num}")

    return float(num)* 3

tools = [TavilySearch(max_results=1,search_depth="basic"), triple]

llm = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq", temperature=0, prompt="Always use the tools to answer the question and don't assume values and give result.").bind_tools(tools=tools)

