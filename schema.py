from typing import List
from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description=" Critique of what is missing")
    superfluous: str = Field(description= "Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word detailed answer to teh question.")
    reflection: Reflection = Field(description="Your reflection on the intial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvments to address teh critique of your current answer"
    )