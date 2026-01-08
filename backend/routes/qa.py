from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any
import sys
import os
import structlog
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pdf_processor import pdf_processor
    from vector_store import vector_store
    from agents import qa_agent
    from security import validate_file_upload, sanitize_filename, sanitize_document_text, sanitize_text_input
    from config import get_settings
except ImportError as e:
    print(f"Import error in qa route: {e}")
    # Create fallback functions for development
    class MockProcessor:
        async def extract_text_with_metadata(self, file):
            return {"full_text": "Mock text", "filename": file.filename, "page_count": 1}

    class MockVectorStore:
        def add_documents(self, documents, metadatas):
            pass
        def similarity_search(self, query, k=5):
            return [{"content": "Mock content", "metadata": {}, "distance": 0.5}]
        def get_collection_info(self):
            return {"status": "mock", "count": 0}

    class MockQAAgent:
        def answer_question(self, question, docs):
            return "Mock answer"

    class MockSettings:
        is_production = False

    pdf_processor = MockProcessor()
    vector_store = MockVectorStore()
    qa_agent = MockQAAgent()

    def validate_file_upload(file):
        return True

    def sanitize_filename(filename):
        return filename

    def sanitize_document_text(text):
        return text

    def sanitize_text_input(text):
        return text

    def get_settings():
        return MockSettings()

logger = structlog.get_logger(__name__)
settings = get_settings()

router = APIRouter()

class UploadNotesResponse(BaseModel):
    message: str
    filename: str
    page_count: int
    success: bool

class QARequest(BaseModel):
    question: str

    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        # Removed character limit to allow longer questions
        return v.strip()

class QAResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    success: bool

@router.post("/upload_notes", response_model=UploadNotesResponse)
async def upload_notes(request: Request, file: UploadFile = File(...)):
    """
    Upload PDF notes and store embeddings in ChromaDB
    """
    try:
        logger.info("PDF upload for Q&A", filename=file.filename)
        
        # Validate file upload
        validate_file_upload(file)
        
        # Extract text from PDF with optimized limits for faster processing
        pdf_data = await pdf_processor.extract_text_with_metadata(file, max_pages=50, max_chars=30000)

        if not pdf_data['full_text'].strip():
            logger.warning("Empty PDF uploaded for Q&A", filename=file.filename)
            raise HTTPException(status_code=400, detail="PDF appears to be empty or could not extract text")

        # Sanitize extracted text
        sanitized_text = sanitize_document_text(pdf_data['full_text'])

        # Log processing info
        logger.info("PDF processed for Q&A",
                   filename=pdf_data['filename'],
                   total_pages=pdf_data['page_count'],
                   pages_processed=pdf_data['pages_processed'],
                   chars_extracted=len(pdf_data['full_text']),
                   truncated=pdf_data.get('truncated', False))
        
        # Store in vector database
        try:
            vector_store.add_documents(
                documents=[sanitized_text],
                metadatas=[{
                    'filename': pdf_data['filename'],
                    'page_count': pdf_data['page_count'],
                    'upload_timestamp': str(pdf_data.get('metadata', {}).get('creationDate', '')),
                    'text_length': len(sanitized_text)
                }]
            )

            # Verify document was stored
            collection_info = vector_store.get_collection_info()
            logger.info("PDF uploaded to vector store",
                       filename=pdf_data['filename'],
                       page_count=pdf_data['page_count'],
                       text_length=len(sanitized_text),
                       total_documents=collection_info.get('total_documents', 0))

        except Exception as store_error:
            logger.error("Failed to store document in vector store", error=str(store_error))
            raise HTTPException(status_code=500, detail=f"Failed to store document: {str(store_error)}")
        
        return UploadNotesResponse(
            message="Notes uploaded and processed successfully",
            filename=pdf_data['filename'],
            page_count=pdf_data['page_count'],
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("PDF upload error", error=str(e), filename=getattr(file, 'filename', 'unknown'))
        if settings.is_production:
            raise HTTPException(status_code=500, detail="Error uploading notes")
        else:
            raise HTTPException(status_code=500, detail=f"Error uploading notes: {str(e)}")

@router.post("/ask", response_model=QAResponse)
async def ask_question(http_request: Request, request: QARequest):
    """
    Ask a question and get an answer using RAG
    """
    try:
        logger.info("Q&A request", question_length=len(request.question))
        
        # Sanitize question input
        sanitized_question = sanitize_text_input(request.question)
        
        # Check if we have any documents in the collection
        try:
            collection_info = vector_store.get_collection_info()
            total_chunks = collection_info.get('total_documents', 0)
            has_documents = total_chunks > 0
            logger.info("Document availability check",
                       has_documents=has_documents,
                       total_chunks=total_chunks,
                       collection_name=collection_info.get('collection_name', 'unknown'),
                       vector_store_id=id(vector_store))
        except Exception as count_error:
            logger.warning("Collection info check failed", error=str(count_error))
            has_documents = False

        if not has_documents:
            logger.info("No documents available in knowledge base")
            return QAResponse(
                answer="I don't have any documents in my knowledge base yet. Please upload some study notes first so I can help answer your questions.",
                sources=[],
                success=False
            )

        # For general questions like "what is this document about", be more lenient
        is_general_question = any(phrase in sanitized_question.lower() for phrase in [
            "what is", "what's", "tell me about", "document about", "about this", "overview", "summary"
        ])

        # Use different thresholds based on question type
        if is_general_question:
            # For general questions, use very low threshold to get all available documents
            relevant_docs = vector_store.similarity_search(sanitized_question, k=15, similarity_threshold=0.01)
        else:
            # For specific questions, use moderate threshold
            relevant_docs = vector_store.similarity_search(sanitized_question, k=12, similarity_threshold=0.1)

        logger.info("Similarity search results",
                   docs_found=len(relevant_docs),
                   question=sanitized_question[:100],
                   is_general_question=is_general_question)

        if is_general_question and len(relevant_docs) > 0:
            # For general questions, use any available documents
            logger.info("General question detected, using available documents")
            high_confidence_docs = relevant_docs
        else:
            # For specific questions, maintain higher confidence threshold
            high_confidence_docs = [doc for doc in relevant_docs if not doc.get('low_confidence', False)]

        logger.info("Document confidence analysis",
                   high_confidence=len(high_confidence_docs),
                   total_found=len(relevant_docs),
                   is_general_question=is_general_question)

        # If we have any documents at all, try to answer
        if len(relevant_docs) > 0 and len(high_confidence_docs) == 0:
            # Use the best available documents even if confidence is low
            logger.info("Using low-confidence documents as fallback")
            high_confidence_docs = relevant_docs[:3]  # Use top 3 documents

        # Generate answer using AI agent with only high-confidence documents
        answer = qa_agent.answer_question(sanitized_question, high_confidence_docs)

        # Format sources
        sources = []
        for doc in high_confidence_docs:
            similarity_score = doc.get('similarity_score', doc.get('relevance_score', 1 - doc.get('distance', 0)))
            sources.append({
                'content': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                'metadata': doc['metadata'],
                'relevance_score': similarity_score
            })
        
        logger.info("Q&A response generated", sources_count=len(sources))
        
        return QAResponse(
            answer=answer,
            sources=sources,
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Q&A processing error", error=str(e))
        if settings.is_production:
            raise HTTPException(status_code=500, detail="Error processing question")
        else:
            raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@router.get("/collection_info")
async def get_collection_info():
    """
    Get information about the vector collection
    """
    try:
        info = vector_store.get_collection_info()
        logger.info("Collection info endpoint", total_documents=info.get('total_documents', 0), vector_store_id=id(vector_store))
        return info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collection info: {str(e)}")

@router.delete("/clear")
async def clear_knowledge_base():
    """
    Clear all documents from the vector store
    """
    try:
        logger.info("Clearing knowledge base")

        # Clear the collection using the vector store's method
        vector_store.clear_collection()

        # Verify the collection is cleared
        info = vector_store.get_collection_info()
        cleared_count = info.get('total_documents', 0)

        logger.info("Knowledge base cleared", remaining_documents=cleared_count)

        if cleared_count == 0:
            return {"message": "Knowledge base cleared successfully", "remaining_documents": 0}
        else:
            return {"message": f"Knowledge base partially cleared, {cleared_count} documents remaining", "remaining_documents": cleared_count}

    except Exception as e:
        logger.error("Error clearing knowledge base", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error clearing knowledge base: {str(e)}")

@router.get("/test_storage")
async def test_storage():
    """Test endpoint to verify document storage is working"""
    try:
        # Add a test document
        test_doc = "This is a test document for verifying storage functionality."
        vector_store.add_documents(
            documents=[test_doc],
            metadatas=[{"test": True, "timestamp": str(datetime.now())}]
        )

        # Check if it was stored
        info = vector_store.get_collection_info()
        return {
            "message": "Test document added successfully",
            "total_documents": info.get('total_documents', 0),
            "collection_name": info.get('collection_name', 'unknown')
        }
    except Exception as e:
        return {
            "error": f"Failed to add test document: {str(e)}",
            "total_documents": vector_store.get_collection_info().get('total_documents', 0)
        }

@router.get("/debug_vector_store")
async def debug_vector_store():
    """Debug endpoint to check vector store state"""
    try:
        info = vector_store.get_collection_info()
        # Try to get a sample document
        try:
            sample = vector_store.collection.get(limit=1)
            sample_count = len(sample.get('documents', [])) if sample.get('documents') else 0
        except Exception as sample_error:
            sample_count = f"Error: {str(sample_error)}"

        return {
            "vector_store_id": id(vector_store),
            "collection_info": info,
            "sample_document_count": sample_count,
            "collection_object_id": id(vector_store.collection) if hasattr(vector_store, 'collection') else "No collection",
            "embeddings_available": vector_store.embeddings is not None if hasattr(vector_store, 'embeddings') else "Unknown"
        }
    except Exception as e:
        return {"error": str(e), "vector_store_id": id(vector_store)}

@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    return {"status": "healthy", "service": "qa"}
