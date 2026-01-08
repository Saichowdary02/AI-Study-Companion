from PyPDF2 import PdfReader
from typing import List, Dict, Any
import io

class PDFProcessor:
    def __init__(self):
        pass
    
    async def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            # Read the PDF file asynchronously
            pdf_content = await pdf_file.read()
            
            # Create a BytesIO object from the content
            pdf_stream = io.BytesIO(pdf_content)
            
            # Open PDF with pypdf
            pdf_reader = PdfReader(pdf_stream)
            
            text_content = ""
            
            # Extract text from each page
            for page in pdf_reader.pages:
                text_content += page.extract_text()
                text_content += "\n\n"  # Add spacing between pages
            
            return text_content
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    async def extract_text_with_metadata(self, pdf_file, max_pages: int = 50, max_chars: int = 50000) -> Dict[str, Any]:
        """Extract text and metadata from PDF with size limits for faster processing"""
        try:
            pdf_content = await pdf_file.read()
            pdf_stream = io.BytesIO(pdf_content)
            pdf_reader = PdfReader(pdf_stream)

            text_content = ""
            page_texts = []
            total_pages = len(pdf_reader.pages)

            # Limit pages to prevent excessive processing time
            pages_to_process = min(total_pages, max_pages)

            # Extract text from each page with character limit
            for i, page in enumerate(pdf_reader.pages):
                if i >= pages_to_process:
                    break

                page_text = page.extract_text()
                page_texts.append(page_text)

                # Add page text to content
                text_content += page_text + "\n\n"

                # Check if we've exceeded character limit
                if len(text_content) >= max_chars:
                    text_content = text_content[:max_chars] + "...\n\n[Content truncated due to length]"
                    break

            # Get PDF metadata
            metadata = pdf_reader.metadata

            return {
                'full_text': text_content.strip(),
                'page_texts': page_texts,
                'page_count': total_pages,
                'pages_processed': len(page_texts),
                'metadata': metadata,
                'filename': pdf_file.filename if hasattr(pdf_file, 'filename') else 'unknown.pdf',
                'truncated': len(text_content) >= max_chars or len(page_texts) < total_pages
            }

        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

# Global PDF processor instance
pdf_processor = PDFProcessor()
