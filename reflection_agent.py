from tkinter import LAST
from urllib import response
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model

load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "you are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet"
            "Always provide detailed recommendations, including requests for lenth, virality, style, etc"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are twitter techie influencer assistant tasked with writing excellent twitter posts."
            "Generate the best twitter post possible for the uder's request."
            "If the user provides critique, respond with a revised version of your previous attempts"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

llm = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq", temperature=0)
reflection_chain = reflection_prompt | llm
generation_chain = generation_prompt | llm



class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph):
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
    res = reflection_chain.invoke({"messages": state["messages"]})

    return {"messages": [HumanMessage(content=res.content)]}


flow = StateGraph(MessageGraph)

flow.add_node(GENERATE, generation_node)
flow.add_node(REFLECT, reflection_node)
flow.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

flow.add_conditional_edges(GENERATE, should_continue, path_map={END:END, REFLECT:REFLECT})
flow.add_edge(REFLECT, GENERATE)

graph = flow.compile()

print(graph.get_graph().draw_mermaid())

if __name__ == "__main__":
    print("Hello langgraph")
    input = HumanMessage(content=""" Make this te=weet better:"
                                @LAngchainAI
        - newly tool calling feature is seriously underratted.
        After a long wait, it's here-making the implementation of agents across diffrent models with function calling.
        made a video converting their newest blog post""")
    response = graph.invoke(input)
    print(response["messages"][-1].content)
    