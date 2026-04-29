from email import message
from mimetypes import init
from pyexpat import model

from langchain_core.messages.tool import tool_call
import main
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessageChunk, ToolMessage
from typing import List, Annotated, Literal
from pydantic import BaseModel, Field

MAX_ITERATIONS = 10
MODEL = "llama-3.3-70b-versatile" 


# ---- Tools (langchain @tool decorator) ---

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"Executing get_product_price(product={product})")
    prices= {"laptop": 1299.99, "headphones": 149.55, "keyboard": 89.50}
    return prices.get(product, 0)
@tool
def apply_discount(price:float, discount_tier:str) -> float:
    """Apply discount tier to a price and return the final price
       Available tiers: bronze, silver, gold."""
    print(f"Excuting apply_discount(price={price}, discount_tier={discount_tier})")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount= discount_percentages.get(discount_tier,0)
    return round(price*(1-discount/100), 2)

def run_agent(question:str):
    tools =[get_product_price, apply_discount]
    tools_dict={t.name: t for t in tools}
    llm = init_chat_model(MODEL,model_provider="groq", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)
    messages = [
        SystemMessage(content=("You are a helpful shopping assitant."
                              "You have access to a product catalog tool"
                              "and a discount tool.\n\n"
                              "STRICT RULES - you must follow these exactly:\n"
                              "1. Never guess or assume any product price."
                              " You MUST call get_product_price first to get the real price"
                              " 2. Ony call apply_discount AFTER you have received"
                              " a price from get_product_price. Pass the price "
                              " returned by get_product_price - do NOT pass a made-up number.\n"
                              "3.NEVER calculate discounts yourself using math"
                              " Always use the apply_discount tool.\n"
                              "4. If the user does not specify a discount tier"
                              " ask them which tier to use -do NOT assume one")
        ),
        HumanMessage(content=question)
    ]
    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n--- iteration {iteration} ---")
        ai_messages= llm_with_tools.invoke(messages)
        tool_calls= ai_messages.tool_calls
        #if no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_messages.content}")
            return ai_messages.content

        # Process onlt first tool call -force tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args= tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"  [Tool Selected] {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        messages.append(ai_messages)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
        





if __name__ == "__main__":
    print("Hello Lanchain Agent (.bind_tools)!")
    print()
    result = run_agent("what is the price of a laptop after applying a gold discount?")