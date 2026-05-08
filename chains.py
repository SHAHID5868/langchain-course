from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "you are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet",
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
reflection_chain = llm | reflection_prompt
generation_chain = llm | generation_prompt
