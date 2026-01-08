from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any
import sys
import os
import structlog

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pdf_processor import pdf_processor
    from agents import note_summarizer
    from security import validate_file_upload, sanitize_filename, sanitize_document_text
    from config import get_settings
except ImportError as e:
    print(f"Import error in summarize route: {e}")
    # Create fallback functions for development
    class MockProcessor:
        async def extract_text_with_metadata(self, file):
            try:
                import PyPDF2
                import io

                # Read the uploaded file
                content = await file.read()
                pdf_file = io.BytesIO(content)

                # Extract text using PyPDF2
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

                return {
                    "full_text": text.strip() if text.strip() else "Could not extract text from PDF",
                    "filename": file.filename,
                    "page_count": len(pdf_reader.pages)
                }
            except Exception as e:
                print(f"PDF processing error: {e}")
                return {
                    "full_text": f"Error processing PDF: {str(e)}",
                    "filename": file.filename,
                    "page_count": 0
                }

    class MockSummarizer:
        def summarize_notes(self, text):
            if "Error processing PDF" in text:
                return "Unable to generate summary due to PDF processing error. Please ensure the PDF is not corrupted and try again."

            # Simple text summarization fallback
            sentences = text.split('.')[:5]  # Take first 5 sentences
            summary = '. '.join(sentences).strip()
            if summary:
                return f"Summary: {summary}..."
            else:
                return "Summary: The document appears to contain text but could not be properly processed for summarization."

    class MockSettings:
        is_production = False

    pdf_processor = MockProcessor()
    note_summarizer = MockSummarizer()

    def validate_file_upload(file):
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        return True

    def sanitize_filename(filename):
        return filename

    def get_settings():
        return MockSettings()

try:
    logger = structlog.get_logger(__name__)
except:
    import logging
    logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

class SummarizeResponse(BaseModel):
    summary: str
    filename: str
    page_count: int
    success: bool

@router.post("/", response_model=SummarizeResponse)
async def summarize_pdf(request: Request, file: UploadFile = File(...)):
    """
    Upload a PDF file and get a summary of its contents
    """
    try:
        logger.info("PDF summarization request", filename=file.filename)
        
        # Validate file upload
        validate_file_upload(file)
        
        # Extract text from PDF with optimized limits for faster processing
        pdf_data = await pdf_processor.extract_text_with_metadata(file, max_pages=30, max_chars=15000)

        if not pdf_data['full_text'].strip():
            logger.warning("Empty PDF uploaded", filename=file.filename)
            raise HTTPException(status_code=400, detail="PDF appears to be empty or could not extract text")

        # Sanitize extracted text
        sanitized_text = sanitize_document_text(pdf_data['full_text'])

        # Further limit for summarization API (keep reasonable size for quality)
        max_length = 8000
        if len(sanitized_text) > max_length:
            sanitized_text = sanitized_text[:max_length] + "..."
            logger.warning(f"Text truncated to {max_length} characters for summarization", filename=pdf_data['filename'])

        # Log processing info
        logger.info("PDF processed for summarization",
                   filename=pdf_data['filename'],
                   total_pages=pdf_data['page_count'],
                   pages_processed=pdf_data['pages_processed'],
                   chars_extracted=len(pdf_data['full_text']),
                   truncated=pdf_data.get('truncated', False))

        # Generate summary using AI agent
        summary = note_summarizer.summarize_notes(sanitized_text)
        
        logger.info("PDF summarization completed", 
                   filename=pdf_data['filename'],
                   page_count=pdf_data['page_count'])
        
        return SummarizeResponse(
            summary=summary,
            filename=pdf_data['filename'],
            page_count=pdf_data['page_count'],
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("PDF processing error", error=str(e), filename=getattr(file, 'filename', 'unknown'))
        if settings.is_production:
            raise HTTPException(status_code=500, detail="Error processing PDF")
        else:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    return {"status": "healthy", "service": "summarize"}
