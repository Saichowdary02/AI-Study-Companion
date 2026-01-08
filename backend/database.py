import os
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force SQLite usage (ignore any PostgreSQL environment variables)
DATABASE_URL = "sqlite+aiosqlite:///./studdy.db"

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Log SQL statements
    future=True  # Use future-compatible APIs
)

# Create session factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

@asynccontextmanager
async def get_db_session():
    """Async context manager for database sessions"""
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Database error: {str(e)}", exc_info=True)
        raise
    finally:
        await session.close()
