#!/usr/bin/env python3
"""Initialize SQLite database for Smart Study Buddy"""

import asyncio
import os
import sys
from sqlalchemy.ext.asyncio import create_async_engine
from models import Base

async def init_database():
    """Initialize the database tables"""
    # Ensure we're using SQLite
    database_url = "sqlite+aiosqlite:///./studdy.db"
    print(f"Initializing database: {database_url}")
    
    # Create async engine
    engine = create_async_engine(database_url, echo=True)
    
    try:
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("Database tables created successfully!")
        
    except Exception as e:
        print(f"Error creating database tables: {e}")
        raise
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(init_database())