"""
Basic security utilities for local development
"""
import os
from fastapi import HTTPException, UploadFile
from config import get_settings

settings = get_settings()

def validate_file_upload(file: UploadFile) -> bool:
    """Validate uploaded file for basic security"""

    # Check file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in settings.allowed_file_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(settings.allowed_file_types)}"
        )

    return True

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent basic security issues"""
    # Remove path components
    filename = os.path.basename(filename)

    # Remove dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')

    return filename

def sanitize_document_text(text: str) -> str:
    """Sanitize document text content"""
    if not text:
        return ""

    # Remove excessive whitespace
    text = ' '.join(text.split())

    # Basic sanitization - remove potentially harmful characters
    # Keep alphanumeric, spaces, and basic punctuation
    import re
    text = re.sub(r'[^\w\s\.,!?\-\(\)]', '', text)

    return text.strip()

def sanitize_text_input(text: str) -> str:
    """Sanitize user text input"""
    if not text:
        return ""

    # Remove leading/trailing whitespace
    text = text.strip()

    # Remove excessive whitespace
    text = ' '.join(text.split())

    # Basic sanitization - remove potentially harmful characters
    import re
    text = re.sub(r'[^\w\s\.,!?\-\(\)]', '', text)

    return text
