from operator import itemgetter
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
INDEX_NAME = "medium-blogs-embeddings-index"  # replace with yours

# ── 1. Embeddings + Vector Store ──────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# k=3 means fetch the 3 most relevant documents from Pinecone


# ── 2. Tool ───────────────────────────────────────────────────
@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
       Available tiers: bronze (5%), silver (12%), gold (23%)."""
    print(f" >> Applying {discount_tier} discount to ${price}")
    discounts = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discounts.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# ── 3. Format docs helper ─────────────────────────────────────
def format_docs(docs):
    # joins all retrieved documents into one clean string
    # each doc separated by a blank line
    return "\n\n".join(doc.page_content for doc in docs)


# ── 4. Prompt Template ────────────────────────────────────────
prompt_template = ChatPromptTemplate.from_template(
    """You are a helpful shopping assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Provide a clear and detailed answer:"""
)


# ── 5. LLM + RAG Chain ────────────────────────────────────────
llm = init_chat_model(model=MODEL, model_provider="groq", temperature=0)
llm_with_tools = llm.bind_tools([apply_discount])

rag_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
        # itemgetter("question") → pulls the question from the input dict
        # retriever              → searches Pinecone for relevant docs
        # format_docs            → converts docs to a plain string
    )
    | prompt_template   # fills {context} and {question}
    | llm_with_tools    # LLM that knows about the discount tool
    | StrOutputParser() # extracts plain text
)


# ── 6. Agent Loop ─────────────────────────────────────────────
def run_rag_agent(question: str):
    tools_dict = {"apply_discount": apply_discount}

    print(f"\nQuestion: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(content=(
            "You are a helpful shopping assistant. "
            "Use the provided context to find prices. "
            "Use the apply_discount tool to calculate discounts. "
            "NEVER calculate discounts yourself."
        )),
        HumanMessage(content=question)
    ]

    for iteration in range(1, 11):
        print(f"\n--- Iteration {iteration} ---")

        # first retrieve context from Pinecone then call LLM
        ai_message = llm_with_tools.invoke(
            messages,
            # inject retrieved context into the system
            config={"configurable": {"context": (
                retriever | format_docs
            ).invoke(question)}}
        )

        tool_calls = ai_message.tool_calls

        # no tool calls means LLM has final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # process tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"  [Tool Selected] {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)
        print(f"  [Tool Result] ${observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call_id
        ))


# ── 7. Cleaner approach — RAG chain feeds the agent ───────────
def run_rag_agent_v2(question: str):
    """
    Cleaner version:
    Step 1 → RAG chain fetches context and generates initial answer
    Step 2 → Agent uses that answer + tool to calculate discount
    """
    tools_dict = {"apply_discount": apply_discount}

    print(f"\nQuestion: {question}")
    print("=" * 60)

    # Step 1: RAG chain gets relevant info from Pinecone
    print("\n[Step 1] Fetching context from Pinecone...")
    rag_answer = rag_chain.invoke({"question": question})
    print(f"[RAG Answer] {rag_answer}")

    # Step 2: Agent loop uses RAG answer + tool for discount
    print("\n[Step 2] Running agent with RAG context...")
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

    for iteration in range(1, 11):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"  [Tool Selected] {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)
        print(f"  [Tool Result] ${observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call_id
        ))


# ── 8. Run it ─────────────────────────────────────────────────
if __name__ == "__main__":
    run_rag_agent_v2("What is the price of a laptop after a gold discount?")