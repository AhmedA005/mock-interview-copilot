from __future__ import annotations

from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate

from .generate import generate_text


class QwenLLM(LLM):
    """LangChain wrapper around our generate_text helper."""

    def _call(self, prompt: str, stop=None) -> str:
        return generate_text(prompt, max_new_tokens=1200)

    @property
    def _llm_type(self) -> str:  # metadata only
        return "custom_qwen"


response_schemas = [
    ResponseSchema(
        name="technical_questions",
        description="List of objects with fields: question, suggested_answer, relevance_score (0-1).",
    ),
    ResponseSchema(
        name="behavioral_questions",
        description="List of objects with fields: question, suggested_answer, relevance_score (0-1).",
    ),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

prompt = PromptTemplate(
    input_variables=["resume", "job_description", "format_instructions"],
    template="""
You are an expert interview coach.

Given the candidate resume:
---
{resume}
---

And the job description:
---
{job_description}
---

Produce a mock interview JSON with:
- technical_questions: array of [question, suggested_answer, relevance_score]
- behavioral_questions: array of [question, suggested_answer, relevance_score]

Notes:
- provide 6-7 technical questions
- provide 3-5 behavioral questions
- Prioritize role-specific technical topics from the job description.
- Suggested answers should be concise (1-3 sentences).
- relevance_score should be the model's estimate (0.0 - 1.0) of how well the candidate can answer.
Return ONLY the JSON.

{format_instructions}
"""
)


def get_chain() -> LLMChain:
    return LLMChain(llm=QwenLLM(), prompt=prompt)

