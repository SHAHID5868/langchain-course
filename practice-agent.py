from pyexpat.errors import messages
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq

MAX_ITERATIONS = 10
MODEL = "llama-3.3-70b-versatile"

@tool
def get_product_price(product: str) -> float:
    """Look for the product and return the price"""
    print(f" >> Executing the get_product_price for {product}")
    price = {"laptop": 1299.99, "headphones": 149.55, "keyboard": 89.50}

    return price.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount tier to a price and return the final price
       Available tiers: bronze, silver, gold."""

    print(f" >> Executing apply_discount to calculate the final price")
    discount_percentage={"bronze": 5, "silver": 12, "gold": 23} 
    discount = discount_percentage.get(discount_tier, 0)

    return round(price*(1- discount/100), 2)

def run_agent(question: str):
    Tools = [get_product_price, apply_discount]
    tool_dict = {t.name: t for t in Tools}
    llm = init_chat_model(temperature=0, model_provider="groq", model=MODEL)
    llm_with_tool = llm.bind_tools(Tools)

    print(f"Question: {question}")
    print("=" *60)
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
        ai_message= llm_with_tool.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\n final answer:{ai_message.content}")
            return ai_message.content
        
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"  [Tool Selected] {tool_name} with args {tool_args}")

        tool_to_use = tool_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))


if __name__ == "__main__":
    print("Hello this is the agent")
    result = (run_agent("what is the price of laptop with silver tier discount?"))