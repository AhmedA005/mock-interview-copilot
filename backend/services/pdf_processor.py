"""
PDF processing service.
Handles extraction of text content from PDF files.
"""

import os
import tempfile
from typing import Optional

from PyPDF2 import PdfReader


class PDFProcessor:
    """Handles PDF text extraction."""

    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        """
        Extract text content from PDF bytes.
        
        Args:
            pdf_bytes: Raw PDF file content as bytes.
            
        Returns:
            Extracted text from all pages, concatenated.
            
        Raises:
            ValueError: If no text could be extracted.
        """
        tmp_path: Optional[str] = None
        try:
            # Write to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            # Extract text from PDF
            reader = PdfReader(tmp_path)
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        finally:
            # Clean up temporary file
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        text = text.strip()
        if not text:
            raise ValueError("Unable to extract text from PDF")
        
        return text
