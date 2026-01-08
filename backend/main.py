import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv('.env')  # Load from current directory when running from backend/

# Verify OpenAI API key is loaded
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key and not openai_key.startswith("#") and len(openai_key.strip()) > 0:
    print(f"✅ OpenAI API key loaded successfully (length: {len(openai_key)})")
else:
    print("❌ OpenAI API key not found or invalid in .env file")

# Import configuration
from config import get_settings, validate_required_settings
from routes import router as summarize, tasks_router as tasks, qa_router as qa
from database import get_db_session, engine
from middleware import RequestLoggingMiddleware
from models import Base

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# Validate configuration
try:
    validate_required_settings()
    logger.info("Configuration validated successfully")
except ValueError as e:
    logger.error(f"Configuration validation failed: {str(e)}")
    raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info(f"Starting Smart Study Buddy API - Version: {settings.api_version}")

    # Create database tables
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise

    yield
    # Shutdown
    logger.info("Shutting down Smart Study Buddy API")

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    debug=settings.debug,
    lifespan=lifespan
)

# Add custom middleware
app.add_middleware(RequestLoggingMiddleware)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:5173/", "http://localhost:3000/", "http://localhost:5000", "http://localhost:5001", "http://localhost:5000/", "http://localhost:5001/"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {str(exc)} - Path: {request.url.path}, Method: {request.method}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

# Include routers
app.include_router(summarize, prefix="/api/v1/summarize", tags=["summarize"])
app.include_router(tasks, prefix="/api/v1/tasks", tags=["tasks"])
app.include_router(qa, prefix="/api/v1/qa", tags=["qa"])

# Handle OPTIONS requests for CORS preflight
@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle CORS preflight OPTIONS requests"""
    return {"message": "OK"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Smart Study Buddy API is running!",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "version": settings.api_version
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
