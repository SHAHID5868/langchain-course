import datetime
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser, PydanticToolsParser
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph, MessagesState, END
import time


load_dotenv()
tavily_search = TavilySearch(max_results=2)




class Reflection(BaseModel):
    missing: str = Field(description=" Critique of what is missing")
    superfluous: str = Field(description= "Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word detailed answer to teh question.")
    reflection: Reflection = Field(description="Your reflection on the intial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvments to address the critique of your current answer"
    )
class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )

llm = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq", temperature=0,model_kwargs={"tool_choice": "auto"} )
parser = JsonOutputKeyToolsParser(key_name="AnswerQuestion",return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            """You are expert researcher.
            Current time: {time}
            1. {first_instruction}
            2. Reflect and critque your answer. BE severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer."""
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using required format"),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat()
)

first_responder_prompt_tempalte = actor_prompt_template.partial(
    first_instruction="provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt_tempalte | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice = "AnswerQuestion"
)

revise_instructions = """Revise your previous answer using the new information.
    - You MUST use the ReviseAnswer tool to structure your response.
    - Do NOT use markdown tables, bullet points, or special formatting in your answer field.
    - Write the answer as plain flowing prose only.
    - You MUST include numerical citations like [1], [2] inline in the text.
    - Add a References list at the end.
    - Keep under 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

def run_queries(search_queries: list[str], **kwargs):
    """run the generated queries"""
    return tavily_search.batch([{"query": query} for query in search_queries])

execute_tools = ToolNode([
    StructuredTool.from_function(run_queries, name="AnswerQuestion"),
    StructuredTool.from_function(run_queries, name="ReviseAnswer")
])

MAX_ITERATION = 1

def draft_node(state: MessagesState):
    """Draft the intial response"""
    time.sleep(3) 
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages":[response]}

def revise_node(state: MessagesState):
    """Draft the intial response"""
    time.sleep(3) 
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages":[response]}

def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine whether to continue or end based on iteration count."""
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATION:
        return END
    return "execute_tools"

flow = StateGraph(MessagesState)
flow.add_node("draft", draft_node)
flow.add_node("execute_tools", execute_tools)
flow.add_node("revise", revise_node)
flow.add_edge(START, "draft")
flow.add_edge("draft", "execute_tools")
flow.add_edge("execute_tools", "revise")
flow.add_conditional_edges("revise", event_loop, ["execute_tools", END])

app = flow.compile()

print(app.get_graph().draw_mermaid())


if __name__ == "__main__":
   
    res = app.invoke({
        "messages":[{
            "role": "user",
            "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital"
        }
        ]
    })
    print(res["messages"][-1].content)


# 1. app.invoke()
#         ↓
# 2. draft_node runs
#    → first_responder fills AnswerQuestion form
#    → state gets AIMessage with answer + reflection + search_queries
#         ↓
# 3. execute_tools runs
#    → sees search_queries: ["AI SOC startups 2024", ...]
#    → runs tavily_search on each query
#    → state gets ToolMessage with web results
#         ↓
# 4. revise_node runs
#    → revisor sees full history (question + draft + web results)
#    → fills ReviseAnswer form with improved answer + citations
#    → state gets new AIMessage
#         ↓
# 5. event_loop checks
#    → count_tool_visits = 1
#    → 1 < 2 → go back to execute_tools
#         ↓
# 6. execute_tools runs again with new search_queries
#         ↓
# 7. revise_node runs again → better answer
#         ↓
# 8. event_loop checks
#    → count_tool_visits = 2
#    → 2 = MAX_ITERATION, not > 2 → go back again!
#         ↓
# 9. execute_tools runs third time
#         ↓
# 10. revise_node runs third time
#         ↓
# 11. event_loop checks
#     → count_tool_visits = 3
#     → 3 > 2 → END ✅
#         ↓
# 12. print(res["messages"][-1].content)
#     → prints final revised answer