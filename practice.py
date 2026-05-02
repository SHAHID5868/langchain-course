from anthropic.types import Model
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_groq import ChatGroq


MAX_ITERATION = 10
MODEL = "llama-3.3-70b-versatile"

@tool
def get_product_price(product:str) -> float:
    """Agent has use this tool to search for product price and return results"""
    print(f">>> Executing get_product_price to get price of {product}")
    prices = {"laptop": 1299.99, "headphones": 149.55, "keyboard": 89.50}

    return prices.get(product, 0) 

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Agent has to calculate the final price with the discount tier applied
    available discount_tier are "bronze", "silver", "gold" """

    print(f">>> Executing apply_discount to calcualte the final disocunt with original price: {price} & disocunt_tier: {discount_tier}")
    discount_percentage = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentage.get(discount_tier, 0)
    final_price = round(price *(1- discount/100), 2)
    return final_price

def run_agent(question:str):
    tools=[get_product_price, apply_discount]
    tool_dict = {t.name: t for t in tools}
    llm = init_chat_model(model=MODEL, model_provider="groq", temperature=0)
    tool_with_llm = llm.bind_tools(tools)

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
    
    print(f"question:{question}")
    print("="* 60)

    for iteration in range(1, MAX_ITERATION+1):
        print(f"Iteration: {iteration}")
        print("_" *30)
        ai_message = tool_with_llm.invoke(messages)
        tool_calls = ai_message.tool_calls
        if  not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        tool_to_use = tool_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

if __name__ == "__main__":
    print("Hello Lanchain Agent (.bind_tools)!")
    print()
    result = run_agent("what is the price of a keyboard after applying a bronze discount?")









