import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import List
import json

load_dotenv()

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing")
    superfluous: str = Field(description="Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    """Answer the question."""
    answer: str = Field(description="~250 word detailed answer")
    reflection: Reflection = Field(description="Your reflection on the initial answer")
    search_queries: List[str] = Field(description="1-3 search queries")

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer."""
    references: List[str] = Field(description="Citations motivating your updated answer")

# Simulate what the graph state looks like during revise_node
fake_history = [
    HumanMessage(content="Write about AI-Powered SOC startups that raised capital."),
    AIMessage(
        content="",
        tool_calls=[{
            "id": "call_123",
            "name": "AnswerQuestion",
            "args": {
                "answer": "AI SOCs use ML to detect threats...",
                "reflection": {
                    "missing": "Missing funding amounts",
                    "superfluous": "Too much background"
                },
                "search_queries": ["AI SOC startups 2024", "autonomous SOC funding"]
            }
        }]
    ),
    ToolMessage(
        content="Search results: Vectra AI raised $200M, Darktrace raised $230M...",
        tool_call_id="call_123"
    ),
]

working_models = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "openai/gpt-oss-20b",
]

for model_id in working_models:
    try:
        llm = init_chat_model(
            model=model_id,
            model_provider="groq",
            temperature=0
        ).bind_tools([ReviseAnswer], tool_choice="ReviseAnswer")

        res = llm.invoke(fake_history)

        if res.tool_calls:
            args = res.tool_calls[0]["args"]
            # Check if answer contains markdown table (causes JSON parse error)
            if "|" in args.get("answer", ""):
                print(f"⚠️  {model_id} → tool called BUT answer has markdown table (will break)")
            else:
                print(f"✅ {model_id} → perfect, no issues")
        else:
            print(f"❌ {model_id} → no tool call with long history")

    except Exception as e:
        print(f"❌ {model_id} → {str(e)[:100]}")