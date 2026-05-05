import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages.tool import ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter




load_dotenv()
MAX_ITERATIONS = 10
@tool
def apply_discount(price:float, discount_tier:str) -> float:
    """Given a price and discount_tier, calculate and return the discounted price.
       Available tiers: 'bronze', 'silver', 'gold'"""
    print(f">>> Executing apply_discount with original price: {price} & discount_tier: {discount_tier}")
    discount_tiers = {"bronze": 5, "silver": 15,"gold": 25 }
    discount= discount_tiers.get(discount_tier, 0)
    return round(price * (1 - discount/100), 2)

embeddings= HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = PineconeVectorStore(index_name= os.getenv("INDEX_NAME"),embedding=embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return"\n\n".join(doc.page_content for doc in docs)

prompt_template = ChatPromptTemplate.from_template(
    """You are a helpful shopping assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Provide a clear and detailed answer:"""
)

tools = [apply_discount]
llm = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq", temperature=0)
llm_with_tools= llm.bind_tools(tools=tools)

rag_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
    )
    | prompt_template
    |llm
    |StrOutputParser()
)

def run_agent(question:str):
    tools_dict = {t.name: t for t in tools}

    print(f"question: {question}")
    print("=" *60)

    rag_answer = rag_chain.invoke({"question": question})

    messages = [
        SystemMessage(content=(
            "You are a helpful shopping assistant. "
            "The following is retrieved information from our database:\n\n"
            f"{rag_answer}\n\n"  # RAG answer injected here as context
            "Use this information to answer questions. "
            "Use the apply_discount tool to calculate discounts. "
            "NEVER calculate discounts yourself."
        )),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        ai_messages = llm_with_tools.invoke(messages)
        tool_calls = ai_messages.tool_calls
        if not tool_calls:
            print(f"Final answer: {ai_messages.content}")
            return ai_messages.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Tool being used{tool_name}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)

        messages.append(ai_messages)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
    raise RuntimeError(f"Agent did not converge after {MAX_ITERATIONS} iterations")

if __name__ == "__main__":
    print("running agent")
    result = run_agent("What is the price of a laptop after a gold discount?")
    print(result)

    
    
