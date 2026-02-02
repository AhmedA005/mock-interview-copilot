"""
API routes for the interview copilot service.
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..config import settings
from ..services import PDFProcessor, QuestionGenerator
from .auth import verify_api_key

router = APIRouter()


@router.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "Mock Interview Copilot API is active",
        "model": settings.MODEL_ID,
    }


@router.get("/health")
async def health():
    """Detailed health check."""
    from ..models import LLMManager, EmbeddingManager
    
    llm = LLMManager()
    embedder = EmbeddingManager()
    
    return {
        "status": "healthy",
        "llm_loaded": llm.is_loaded,
        "embedder_loaded": embedder.is_loaded,
        "model_id": settings.MODEL_ID,
    }


@router.post("/interview")
async def generate_interview_questions(
    file: UploadFile = File(...),
    job_description: str = Form(...),
    _: str = Depends(verify_api_key),
):
    """
    Generate personalized interview questions.
    
    Args:
        file: Resume PDF file.
        job_description: Target job description text.
        
    Returns:
        JSON with technical_questions and behavioral_questions.
    """
    # Validate input
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if len(job_description.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="Job description must be at least 20 characters"
        )

    # Extract resume text
    try:
        resume_bytes = await file.read()
        resume_text = PDFProcessor.extract_text(resume_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing PDF: {str(e)}"
        )

    # Generate questions
    try:
        generator = QuestionGenerator()
        result = generator.generate(resume_text, job_description)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating questions: {str(e)}"
        )

    return JSONResponse(content=result)


# Alias endpoint for compatibility
@router.post("/generate-questions")
async def generate_questions_alias(
    file: UploadFile = File(...),
    job_description: str = Form(...),
    num_technical: int = Form(default=5),
    num_behavioral: int = Form(default=3),
    _: str = Depends(verify_api_key),
):
    """Alias endpoint for backward compatibility."""
    # Validate input
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if len(job_description.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="Job description must be at least 20 characters"
        )

    # Extract resume text
    try:
        resume_bytes = await file.read()
        resume_text = PDFProcessor.extract_text(resume_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Generate questions
    try:
        generator = QuestionGenerator()
        result = generator.generate(
            resume_text,
            job_description,
            num_technical=num_technical,
            num_behavioral=num_behavioral,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(content={"results": result})
