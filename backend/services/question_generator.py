"""
Question generation service.
Handles interview question generation using LLM and fallback mechanisms.
"""

import json
import re
from typing import Dict, List, Optional

from ..config import settings
from ..models import LLMManager, EmbeddingManager
from .text_processor import TextProcessor


class QuestionGenerator:
    """Generates personalized interview questions."""

    def __init__(self):
        self.llm = LLMManager()
        self.embedder = EmbeddingManager()
        self.text_processor = TextProcessor()

    def generate(
        self,
        resume_text: str,
        job_description: str,
        num_technical: int = None,
        num_behavioral: int = None,
    ) -> Dict:
        """
        Generate interview questions based on resume and job description.
        
        Args:
            resume_text: Full text from the candidate's resume.
            job_description: Target job description.
            num_technical: Number of technical questions to generate.
            num_behavioral: Number of behavioral questions to generate.
            
        Returns:
            Dictionary with technical_questions and behavioral_questions lists.
        """
        num_technical = num_technical or settings.DEFAULT_TECHNICAL_QUESTIONS
        num_behavioral = num_behavioral or settings.DEFAULT_BEHAVIORAL_QUESTIONS

        # Prepare embeddings for relevance scoring
        chunks = self.text_processor.chunk_text(resume_text)
        if not chunks:
            raise ValueError("Resume appears to be empty")

        embeddings = self.embedder.encode(chunks)
        faiss_index = self.embedder.create_index(embeddings)

        # Try LLM generation with retries
        parsed = None
        for attempt in range(settings.GENERATION_RETRIES):
            raw_output = self._generate_raw(
                resume_text[:settings.RESUME_MAX_LENGTH],
                job_description,
            )
            parsed = self._parse_output(raw_output, resume_text, job_description)
            if parsed and self._is_valid(parsed):
                break

        # Use fallback if LLM generation failed
        if not parsed or not self._is_valid(parsed):
            parsed = self._create_fallback(resume_text, job_description)

        # Compute relevance scores
        self._add_relevance_scores(parsed, faiss_index, chunks)

        return parsed

    def _generate_raw(self, resume_summary: str, job_description: str) -> str:
        """Generate raw LLM output for interview questions."""
        prompt = self._build_prompt(resume_summary, job_description)
        return self.llm.generate(prompt)

    def _build_prompt(self, resume_summary: str, job_description: str) -> str:
        """Build the prompt for question generation."""
        return f"""<|im_start|>system
You are an expert technical interviewer. Generate interview questions that assess BOTH the candidate's experience AND the job requirements.

CRITICAL RULES:
1. For RESUME-BASED questions: Use SPECIFIC technologies, projects, tools from the resume
2. For JOB-BASED questions: Focus on skills/technologies mentioned in job description
3. For ALL suggested_answers: Write COMPLETE, DETAILED answers (150-200 words) in first person
4. Output ONLY valid JSON

<|im_end|>
<|im_start|>user
CANDIDATE RESUME:
{resume_summary}

TARGET JOB DESCRIPTION:
{job_description}

Generate 5 technical questions (2 resume-based, 3 job-based) and 3 behavioral questions.

JSON FORMAT:
{{
  "technical_questions": [
    {{
      "question": "Clear, specific question",
      "suggested_answer": "Complete example answer (150-200 words)",
      "relevance_score": 0.0-1.0,
      "question_type": "resume_based|job_based"
    }}
  ],
  "behavioral_questions": [
    {{
      "question": "Behavioral question",
      "suggested_answer": "Complete example answer using STAR method",
      "relevance_score": 0.0-1.0,
      "question_type": "resume_based|behavioral"
    }}
  ]
}}

IMPORTANT: Use REAL details from resume and ACTUAL requirements from job description. NO placeholders.

JSON:
<|im_end|>
<|im_start|>assistant
"""

    def _parse_output(
        self,
        raw_output: str,
        resume_text: str,
        job_description: str,
    ) -> Optional[Dict]:
        """Parse LLM output into structured questions."""
        # Remove common markers
        markers = ["<|im_start|>assistant", "JSON:", "assistant\n", "<|im_end|>"]
        start_pos = 0
        for marker in markers:
            pos = raw_output.rfind(marker)
            if pos != -1:
                start_pos = max(start_pos, pos + len(marker))

        text_to_parse = raw_output[start_pos:].strip()
        text_to_parse = re.sub(r"```json\s*", "", text_to_parse)
        text_to_parse = re.sub(r"```\s*", "", text_to_parse)

        # Extract JSON
        start_idx = text_to_parse.find("{")
        end_idx = text_to_parse.rfind("}")
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return None

        json_str = text_to_parse[start_idx : end_idx + 1]
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, dict):
            return None

        # Ensure required keys exist
        parsed.setdefault("technical_questions", [])
        parsed.setdefault("behavioral_questions", [])

        return parsed

    def _is_valid(self, parsed: Dict) -> bool:
        """Check if parsed output contains valid questions."""
        for key in ["technical_questions", "behavioral_questions"]:
            questions = parsed.get(key, [])
            for item in questions:
                question = item.get("question", "") if isinstance(item, dict) else ""
                # Check for unresolved placeholders
                if "[" in question or "from resume" in question.lower():
                    return False
        return bool(parsed.get("technical_questions") or parsed.get("behavioral_questions"))

    def _add_relevance_scores(self, parsed: Dict, faiss_index, chunks: List[str]):
        """Add or update relevance scores for all questions."""
        for bucket in ("technical_questions", "behavioral_questions"):
            questions = parsed.get(bucket, [])
            for question in questions:
                if "relevance_score" not in question or not question["relevance_score"]:
                    question["relevance_score"] = self.embedder.compute_relevance(
                        question.get("question", ""),
                        faiss_index,
                        chunks,
                    )
                question["relevance_score"] = float(round(question["relevance_score"], 3))

    def _create_fallback(self, resume_text: str, job_description: str) -> Dict:
        """Create fallback questions based on extracted resume information."""
        skills = self.text_processor.extract_skills(resume_text)
        projects = self.text_processor.extract_projects(resume_text)
        companies = self.text_processor.extract_companies(resume_text)

        primary_skill = skills[0] if skills else "your technical skills"
        secondary_skills = ", ".join(skills[1:3]) if len(skills) > 1 else "various technologies"
        tertiary_skill = skills[3] if len(skills) > 3 else "relevant technologies"
        project_context = projects[0] if projects else "your projects"
        company_context = companies[0] if companies else "your work experience"
        job_preview = job_description[:100] if job_description else "the target role"

        return {
            "technical_questions": [
                {
                    "question": f"Can you walk me through your experience with {primary_skill}? What specific challenges did you face and how did you overcome them?",
                    "suggested_answer": (
                        f"In my work with {primary_skill}, I encountered a significant challenge when building {project_context}. "
                        "The main issue was optimizing performance while maintaining readability. I profiled the application, tuned queries, and "
                        "implemented caching, which reduced response times by over 3x while keeping the codebase maintainable."
                    ),
                    "relevance_score": 0.82,
                },
                {
                    "question": f"I see you worked on {project_context}. How would you apply that experience to this role, especially regarding {secondary_skills}?",
                    "suggested_answer": (
                        f"During {project_context}, I owned the backend services that relied heavily on {secondary_skills}. "
                        "I designed event-driven services, introduced monitoring, and optimized latency. Those lessons transfer directly to your stack, "
                        "particularly around scaling and instrumentation."
                    ),
                    "relevance_score": 0.78,
                },
                {
                    "question": f"How would you approach designing a scalable system for {job_preview}? What technologies would you choose and why?",
                    "suggested_answer": (
                        "I start by mapping workloads, then partition the architecture into independently scalable services. "
                        f"For {job_preview}, I would pair a reliable relational store with a caching tier, asynchronous workers, and granular observability "
                        "so we can iterate based on real signals."
                    ),
                    "relevance_score": 0.85,
                },
                {
                    "question": f"What's your experience with {tertiary_skill}? Can you describe a situation where you had to optimize code or system performance?",
                    "suggested_answer": (
                        f"I used {tertiary_skill} extensively to diagnose a latency spike. Profiling revealed inefficient database access, so I reworked indexes, "
                        "added batching, and introduced connection pooling, cutting P95 latency from seconds to milliseconds."
                    ),
                    "relevance_score": 0.8,
                },
                {
                    "question": "How do you ensure code quality and maintainability in your projects? What testing strategies do you follow?",
                    "suggested_answer": (
                        "I enforce automated checks (formatting, linting, unit + integration tests) and keep documentation close to the code. "
                        "CI gates every change, and we review architectural decisions so the team shares context."
                    ),
                    "relevance_score": 0.83,
                },
            ],
            "behavioral_questions": [
                {
                    "question": f"Tell me about your experience at {company_context}. Describe a situation where you had to collaborate with others to solve a complex problem.",
                    "suggested_answer": (
                        f"At {company_context}, an intermittent production incident affected key clients. I coordinated backend, DevOps, and QA teammates, "
                        "set up a war room, traced the race condition in our cache, and shipped a fix with targeted regression tests."
                    ),
                    "relevance_score": 0.8,
                },
                {
                    "question": "Tell me about a time when you had to work with a difficult team member or stakeholder. How did you handle the situation?",
                    "suggested_answer": (
                        "I schedule a dedicated conversation, surface data behind each option, and highlight common goals. "
                        "By acknowledging valid concerns and offering compromises where possible, we keep discussions constructive."
                    ),
                    "relevance_score": 0.78,
                },
                {
                    "question": f"Describe a time when you had to quickly learn a new technology or adapt to significant changes while working on {project_context}.",
                    "suggested_answer": (
                        "I built a structured learning plan, paired with experienced teammates, and documented every insight in our wiki. "
                        "Within weeks I became the go-to contact for that component, proving the value of deliberate practice and knowledge sharing."
                    ),
                    "relevance_score": 0.77,
                },
            ],
        }
