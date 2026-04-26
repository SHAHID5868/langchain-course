from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import init_chat_model

def get_messages(information):
    messages = [
        SystemMessage(content="You are a helpful assistant that can answer questions and help with tasks."),
        HumanMessage(content=information)
    ]
    return messages

def get_response(messages):
    model = init_chat_model(model="gemma3:270m",model_provider="ollama")
    response = model.invoke(messages)
    return response.content

print(get_response(get_messages('''
     1. i want to build a API that can connect with D365 F&O application.
     2. I want to know how to do App registration of my application in D365 F&O''')))