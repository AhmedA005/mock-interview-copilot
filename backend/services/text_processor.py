"""
Text processing service.
Handles chunking and information extraction from text.
"""

import re
from typing import List

from ..config import settings


class TextProcessor:
    """Handles text processing operations for resumes."""

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = None,
        overlap: int = None,
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk.
            chunk_size: Number of words per chunk.
            overlap: Number of overlapping words between chunks.
            
        Returns:
            List of text chunks.
        """
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP

        words = text.split()
        chunks = []
        step = max(1, chunk_size - overlap)

        for idx in range(0, len(words), step):
            chunk = " ".join(words[idx : idx + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)

        return chunks

    @staticmethod
    def extract_skills(text: str) -> List[str]:
        """Extract skills from resume text."""
        skills = []

        # Pattern-based skill extraction
        skill_patterns = [
            r"(?:skills|technologies|technical skills|expertise)[\s:]*([^\n]+(?:\n[^\n]+){0,10})",
            r"(?:proficient in|experienced with|familiar with)[\s:]*([^\n]+)",
        ]

        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                potential = re.split(r"[,;|•\n]", match)
                for skill in potential:
                    skill = re.sub(r"^[-•*\s]+", "", skill.strip())
                    if 3 <= len(skill) <= 50 and not skill.isdigit():
                        skills.append(skill)

        # Technology name patterns
        tech_pattern = r"\b[A-Z][a-zA-Z]*(?:\.[A-Z][a-zA-Z]*)*\b|\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b"
        tech_matches = re.findall(tech_pattern, text)

        common_words = {
            "Bachelor", "Master", "University", "College", "Project",
            "Work", "Experience", "Education", "Skills", "Summary", "Objective",
        }

        for tech in tech_matches:
            if tech not in common_words and len(tech) >= 3:
                skills.append(tech)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for skill in skills:
            lowered = skill.lower()
            if lowered not in seen:
                seen.add(lowered)
                unique.append(skill)

        return unique[:15]

    @staticmethod
    def extract_projects(text: str) -> List[str]:
        """Extract project descriptions from resume text."""
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

    @staticmethod
    def extract_companies(text: str) -> List[str]:
        """Extract company/work experience from resume text."""
        companies = []
        lines = text.split("\n")

        job_keywords = [
            "intern", "developer", "engineer", "analyst", "specialist",
            "manager", "consultant", "designer", "architect", "lead",
        ]

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
