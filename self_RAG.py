from dotenv import load_dotenv
from langchain_core import documents
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.outputs import generation
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing import Any, List, Dict
from langgraph.graph import MessagesState, StateGraph, END  # Fix 1: END from langgraph, not tkinter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

class GraphState(MessagesState):
    question: str
    generation: str
    web_search: bool
    documents: List[str]

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(embedding=embedding, index_name="advanced-rag")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def retrieve(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

llm = init_chat_model(model="gemini-2.5-flash", model_provider="google-genai", temperature=0)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents"""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])

retrieval_grader = grade_prompt | structured_llm_grader


def grade_documents(state: GraphState) -> Dict[str, Any]:
    print("----CHECK DOCUMENT RELEVANCE TO QUESTION----")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False

    for d in documents:
        score = retrieval_grader.invoke({       # Fix 2: indentation — all inside loop
            "question": question,
            "document": d.page_content
        })
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True

    return {"documents": filtered_docs, "question": question, "web_search": web_search}


web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})  # Fix 3: returns a list, iterate correctly
    joined_tavily_results = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]  # Fix 4: tavily_result not tavily_results
    )
    web_results = Document(page_content=joined_tavily_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


prompt = ChatPromptTemplate.from_messages([
    ("human", """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 
Context: {context} 
Answer:""")
]) 

generation_chain = prompt | llm | StrOutputParser()

def generate(state: GraphState) -> Dict[str, Any]:
    print("--GENERATE--")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})  # Fix 6: variable name was shadowing import
    return {"documents": documents, "question": question, "generation": generation}


RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEBSEARCH = "websearch"

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    if state["web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS RELEVANT, INCLUDE WEB SEARCH---")
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE

class GradeHalluciations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

structured_llm_grader_grade = llm.with_structured_output(GradeHalluciations)

system = """You are agrader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
        Give binary score 'yes' or 'no'.'Yes means the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation:{generation}")
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader_grade

class GradeAnswer(BaseModel):
    binary_score: bool = Field(description="Answer addresses the question, 'yes' or 'no'")

structured_llm_grader_gradeanswer= llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an asnwer addresses / resolves a question.\n
        Give a binary score 'yes' or 'no'. Yes means that the answer resolves teh question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader_gradeanswer

def grade_generation_grounded_in_documents_and_question(state:GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation =state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    if hallucination_grader := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION VS QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grader := score.binary_score:
            print("DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISON: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "not supported"



flow = StateGraph(GraphState)
flow.add_node(RETRIEVE, retrieve)
flow.add_node(GRADE_DOCUMENTS, grade_documents)
flow.add_node(GENERATE, generate)
flow.add_node(WEBSEARCH, web_search)
flow.set_entry_point(RETRIEVE)
flow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
flow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate, path_map={WEBSEARCH: WEBSEARCH, GENERATE: GENERATE})
flow.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question, path_map={"not supoorted": GENERATE, "useful": END, "not useful": WEBSEARCH})
flow.add_edge(WEBSEARCH, GENERATE)
flow.add_edge(GENERATE, END)

app = flow.compile()

if __name__ == "__main__":
    response = app.invoke(input={"question": "what is agent memory?"})
    print(response["messages"])