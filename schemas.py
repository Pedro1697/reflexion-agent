from typing import List
from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superflous: str = Field(description="Critique of what is superflous")


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 words detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to adress the critiqued of your current answer."
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citation motivating your updated answer."
    )
