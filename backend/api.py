"""
Interview Readiness API
Reimplements the Kaggle backend inside the repo so we can run it locally.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import nest_asyncio
import numpy as np
import requests
import torch
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

nest_asyncio.apply()

API_KEY = os.getenv("API_KEY", "secret123")
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2100"))
GENERATION_RETRIES = int(os.getenv("GENERATION_RETRIES", "2"))
ENABLE_NGROK = os.getenv("ENABLE_NGROK", "0") not in {"0", "false", "False"}
NGROK_BINARY = os.getenv("NGROK_BINARY", "./ngrok")
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")

tokenizer = None
model = None
embedder = None


def load_llm() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load (or reuse) the HuggingFace model + tokenizer."""
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return tokenizer, model

    print("🚀 Loading Qwen model...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
    ).eval()
    print("✅ Model ready")
    return tokenizer, model


def get_embedder() -> SentenceTransformer:
    """Load the embedding model once."""
    global embedder
    if embedder is None:
        print("🔎 Loading embedding model...")
        embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return embedder


def generate_interview_questions(resume_text: str, job_description: str) -> str:
    tokenizer, model = load_llm()
    resume_summary = resume_text[:1500]

    prompt = f"""<|im_start|>system
You are an expert technical interviewer. Generate interview questions that assess BOTH the candidate's experience AND the job requirements.

- (IMPORTANT) GENERATE 5 Technical questions and 3 behavioral questions

CRITICAL RULES:
1. For RESUME-BASED questions:
   - Use SPECIFIC technologies, projects, tools from the resume
   - Ask about their ACTUAL experience and results
   - Reference real details (no placeholders)
   
2. For JOB-BASED questions:
   - Focus on skills/technologies mentioned in job description
   - Ask how they would approach problems relevant to the role
   - Test knowledge needed for the position
   
3. For ALL suggested_answers:
   - Write COMPLETE, DETAILED answers (150-200 words)
   - Use first person ("I", "We", "My team")
   - Include specific technical details
   - Show problem-solving process
   - Demonstrate measurable results
   
4. Output ONLY valid JSON

<|im_end|>
<|im_start|>user
CANDIDATE RESUME:
{resume_summary}

TARGET JOB DESCRIPTION:
{job_description}

Generate 5 technical questions:
- 2 questions: Based on candidate's resume experience (specific projects/tech they've used)
- 3 questions: Based on job requirements (skills needed for role, theoretical/practical)

Generate 3 behavioral questions:
- 1 question: About specific experience from their resume
- 1 question: About teamwork/collaboration
- 1 question: About handling challenges or learning new skills

For EACH question, provide:
- question: Clear, specific question
- suggested_answer: Complete example answer (150-200 words) showing how candidate could respond
- relevance_score: 0.0-1.0 (higher = more relevant to this specific role)

JSON FORMAT:
{
  "technical_questions": [
    {
      "question": "Tell me about your experience with [specific tech from resume] in your [specific project from resume].",
      "suggested_answer": "In my role at [Company from resume], I worked extensively with [technology]. For example, when building [specific project], I [detailed technical approach with specific tools/methods]. The challenge was [specific problem]. I solved it by [detailed solution with code/architecture decisions]. This resulted in [measurable outcome like performance improvement, user impact, etc.].",
      "relevance_score": number [0-1],
      "question_type": "resume_based"
    },
    {
      "question": "How would you design a [system/feature from job description] using [technology required for job]?",
      "suggested_answer": "I would approach this by first [analysis step]. Based on the requirements, I'd use [specific technologies from job description] because [technical reasoning]. The architecture would consist of [detailed design with components]. For scalability, I would [specific approach]. I have experience with similar challenges when [brief reference to related resume experience if applicable, or 'In my current learning/exploration of this area']. Key considerations would be [technical factors like performance, maintainability, security].",
      "relevance_score": number [0-1],
      "question_type": "job_based"
    },
    { 
      "question": "How would you optimize a [system/feature from job description] to improve performance using [technology required for job]?", 
      "suggested_answer": "I would approach this by profiling the current [system/feature from job description] to identify bottlenecks. Based on the requirements, I'd apply [technology required for job] because it offers tools and patterns suitable for performance optimization. The architecture would be refined to reduce latency, optimize resource usage, and streamline data processing. For scalability, I would ensure asynchronous workflows and caching where appropriate. I encountered similar performance challenges in previous development tasks. Key considerations would be response time, throughput, and system reliability.", 
      "relevance_score": number [0-1], 
      "question_type": "job_based" 
    }
  ],
  "behavioral_questions": [
    {
      "question": "Describe a challenging technical problem you solved in your [specific project from resume].",
      "suggested_answer": "In [specific project from resume], we faced [specific challenge]. My role was [responsibility]. The situation was particularly challenging because [context]. I took the following approach: First, [action 1 with details]. Then, [action 2]. I collaborated with [team members/stakeholders] to [specific collaboration]. The result was [measurable outcome]. This experience taught me [key learning] which I now apply when [how it's relevant to future work].",
      "relevance_score": number [0-1],
      "question_type": "resume_based"
    },
    {
      "question": "Tell me about a time when you had to collaborate with a difficult team member or stakeholder. How did you handle it?",
      "suggested_answer": "Based on my experience, I would approach this by [initial assessment]. I believe in [key principle relevant to situation]. I would start by [first step], ensuring [consideration]. When working with [team type], I would [collaboration approach]. Drawing from my experience, I would [specific actions]. I'd measure success by [metrics/outcomes]. My goal would be [desired result aligned with job requirements].",
      "relevance_score": number [0-1],
      "question_type": "behavioral"
    },
    {
      "question": "Describe a time when you had to quickly learn a new technology or adapt to changing requirements.",
      "suggested_answer": "When facing [situation], I needed to quickly learn [technology/skill]. I took a structured approach: First, [learning step 1]. Then, [learning step 2]. I also [additional action like seeking mentorship or practicing]. Within [timeframe], I was able to [achievement]. This experience taught me [key learning about adaptability]. I now apply this approach when [how it's relevant to future challenges].",
      "relevance_score": number [0-1],
      "question_type": "behavioral"
    }
  ]
}

IMPORTANT: 
- Generate EXACTLY 5 technical questions and 3 behavioral questions
- Mix resume-based and job-based questions
- Use REAL details from resume and ACTUAL requirements from job description
- NO placeholders like [specific tech from resume] in the actual output

JSON:
<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
    return response.strip()


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for idx in range(0, len(words), step):
        chunk = " ".join(words[idx : idx + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def embed_chunks(chunks: List[str]) -> Tuple[SentenceTransformer, np.ndarray]:
    if not chunks:
        raise ValueError("No text found in resume.")
    model = get_embedder()
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return model, embeddings


def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def search_index(query: str, model: SentenceTransformer, index, chunks: List[str], k: int = 5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return distances[0], [chunks[i] for i in indices[0]]


def compute_relevance(question: str, embed_model, faiss_index, resume_chunks):
    distances, _ = search_index(question, embed_model, faiss_index, resume_chunks, k=3)
    avg_dist = float(distances.mean())
    score = 1 / (1 + avg_dist)
    return min(max(score, 0.0), 1.0)


def extract_skills_from_text(text: str) -> List[str]:
    skills = []
    skill_section_patterns = [
        r"(?:skills|technologies|technical skills|expertise)[\s:]*([^\n]+(?:\n[^\n]+){0,10})",
        r"(?:proficient in|experienced with|familiar with)[\s:]*([^\n]+)",
    ]

    for pattern in skill_section_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            potential = re.split(r"[,;|•\n]", match)
            for skill in potential:
                skill = re.sub(r"^[-•*\s]+", "", skill.strip())
                if 3 <= len(skill) <= 50 and not skill.isdigit():
                    skills.append(skill)

    tech_pattern = r"\b[A-Z][a-zA-Z]*(?:\.[A-Z][a-zA-Z]*)*\b|\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b"
    tech_matches = re.findall(tech_pattern, text)
    common_words = {
        "Bachelor",
        "Master",
        "University",
        "College",
        "Project",
        "Work",
        "Experience",
        "Education",
        "Skills",
        "Summary",
        "Objective",
    }

    for tech in tech_matches:
        if tech not in common_words and len(tech) >= 3:
            skills.append(tech)

    compound_pattern = r"\b[A-Z][a-z]+\s+(?:[A-Z][a-z]+\s+)*[A-Z][a-z]+\b"
    compound_matches = re.findall(compound_pattern, text)
    for compound in compound_matches:
        if len(compound) <= 40 and compound not in common_words:
            skills.append(compound)

    unique = []
    seen = set()
    for skill in skills:
        lowered = skill.lower()
        if lowered not in seen:
            seen.add(lowered)
            unique.append(skill)
    return unique[:15]


def extract_projects_from_text(text: str) -> List[str]:
    projects = []
    project_patterns = [
        r"(?:project|projects)[\s:]*\n([^\n]+(?:\n(?!\n)[^\n]+)*)",
        r"(?:developed|built|created|implemented)\s+([^\.\n]{10,80})",
    ]
    for pattern in project_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            cleaned = match.strip()[:100]
            if cleaned and len(cleaned) > 10:
                projects.append(cleaned)
    return projects[:5]


def extract_companies_from_text(text: str) -> List[str]:
    companies = []
    lines = text.split("\n")
    job_keywords = ["intern", "developer", "engineer", "analyst", "specialist", "manager", "consultant", "designer", "architect", "lead"]
    for idx, line in enumerate(lines):
        lower = line.lower()
        if any(keyword in lower for keyword in job_keywords):
            company_line = line.strip()
            if idx + 1 < len(lines):
                next_line = lines[idx + 1].strip()
                if next_line and len(next_line) < 60 and not next_line[0].isdigit():
                    company_line = f"{company_line} at {next_line}"
            if len(company_line) > 5:
                companies.append(company_line[:80])
    return companies[:3]


def create_smart_fallback(resume_text: str, job_description: str) -> Dict:
    skills = extract_skills_from_text(resume_text)
    projects = extract_projects_from_text(resume_text)
    companies = extract_companies_from_text(resume_text)
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


def load_pdf_text(pdf_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        reader = PdfReader(tmp_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    finally:
        os.remove(tmp_path)
    return text.strip()


def parse_questions(raw_output: str, resume_text: str, job_description: str) -> Dict:
    markers = [
        "<|im_start|>assistant",
        "JSON:",
        "assistant\n",
        "<|im_end|>",
    ]
    start_pos = 0
    for marker in markers:
        pos = raw_output.rfind(marker)
        if pos != -1:
            start_pos = max(start_pos, pos + len(marker))
    text_to_parse = raw_output[start_pos:].strip()
    text_to_parse = re.sub(r"```json\s*", "", text_to_parse)
    text_to_parse = re.sub(r"```\s*", "", text_to_parse)
    start_idx = text_to_parse.find("{")
    end_idx = text_to_parse.rfind("}")
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        return create_smart_fallback(resume_text, job_description)
    json_str = text_to_parse[start_idx : end_idx + 1]
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return create_smart_fallback(resume_text, job_description)

    if not isinstance(parsed, dict):
        return create_smart_fallback(resume_text, job_description)

    for key in ["technical_questions", "behavioral_questions"]:
        parsed.setdefault(key, [])

    def valid_questions(items):
        for item in items:
            question = item.get("question", "") if isinstance(item, dict) else ""
            if "[" in question or "from resume" in question.lower():
                return False
        return True

    if not valid_questions(parsed["technical_questions"]) or not valid_questions(parsed["behavioral_questions"]):
        return create_smart_fallback(resume_text, job_description)
    return parsed


app = FastAPI(title="Interview Readiness API")
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Interview Readiness API is active",
        "model": MODEL_ID,
    }


@app.post("/interview")
async def generate_interview_questions_endpoint(
    file: UploadFile = File(...),
    job_description: str = Form(...),
    token: str = Depends(verify_token),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    if len(job_description.strip()) < 20:
        raise HTTPException(status_code=400, detail="Job description must be at least 20 characters")

    resume_bytes = await file.read()
    resume_text = load_pdf_text(resume_bytes)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Unable to extract text from resume")

    resume_chunks = chunk_text(resume_text)
    if not resume_chunks:
        raise HTTPException(status_code=400, detail="Resume appears to be empty")

    embed_model, embeddings = embed_chunks(resume_chunks)
    faiss_index = create_faiss_index(embeddings)

    parsed = None
    for attempt in range(GENERATION_RETRIES):
        raw_output = generate_interview_questions(resume_text[:1500], job_description)
        parsed = parse_questions(raw_output, resume_text, job_description)
        if parsed and (parsed.get("technical_questions") or parsed.get("behavioral_questions")):
            break

    if not parsed:
        parsed = create_smart_fallback(resume_text, job_description)

    for bucket in ("technical_questions", "behavioral_questions"):
        questions = parsed.get(bucket, [])
        for question in questions:
            if "relevance_score" not in question or not question["relevance_score"]:
                question["relevance_score"] = compute_relevance(
                    question.get("question", ""),
                    embed_model,
                    faiss_index,
                    resume_chunks,
                )
            question["relevance_score"] = float(round(question["relevance_score"], 3))

    return JSONResponse(content=parsed)


def start_ngrok(port: int):
    binary_path = Path(NGROK_BINARY)
    if not binary_path.exists():
        resolved = shutil.which("ngrok")
        if not resolved:
            print("⚠️ ngrok binary not found; skipping tunnel setup.")
            return
        binary_path = Path(resolved)

    if NGROK_AUTHTOKEN:
        subprocess.run([str(binary_path), "config", "add-authtoken", NGROK_AUTHTOKEN], check=False)

    process = subprocess.Popen(
        [str(binary_path), "http", str(port), "--log=stdout"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(4)
    try:
        response = requests.get("http://localhost:4040/api/tunnels", timeout=5)
        tunnels = response.json()
        public_url = tunnels["tunnels"][0]["public_url"]
        print("\n" + "=" * 70)
        print("🚀 INTERVIEW READINESS API IS RUNNING!")
        print("=" * 70)
        print(f"📱 Public URL: {public_url}")
        print(f"🏠 Local URL:  http://localhost:{port}")
        print(f"🔑 API Key:    {API_KEY}")
        print("=" * 70 + "\n")
    except Exception as exc:
        print(f"⚠️ Unable to query ngrok tunnel: {exc}")


def run():
    load_llm()
    port = int(os.getenv("PORT", "8000"))
    if ENABLE_NGROK:
        thread = threading.Thread(target=start_ngrok, args=(port,), daemon=True)
        thread.start()
        time.sleep(5)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    run()

