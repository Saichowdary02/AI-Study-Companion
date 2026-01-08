from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class Task(Base):
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    due_date = Column(String(10))  # YYYY-MM-DD format
    status = Column(String(20), default='pending')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    plan_id = Column(Integer, ForeignKey('plans.id', ondelete='SET NULL'))
    
    plan = relationship("Plan", back_populates="tasks")

class Plan(Base):
    __tablename__ = 'plans'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    assignment_description = Column(Text)
    due_date = Column(String(10))  # YYYY-MM-DD format
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    tasks = relationship("Task", back_populates="plan", cascade="all, delete-orphan")

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    content_type = Column(String(50))
    vector_id = Column(String(50))  # Reference to ChromaDB vector ID
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    doc_metadata = Column(Text)  # JSON string for additional metadata
