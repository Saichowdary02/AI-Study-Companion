from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db_session, AsyncSessionLocal
from models import Task, Plan
import structlog

# Import the task planner agent
try:
    from agents import TaskPlannerAgent
    task_planner = TaskPlannerAgent()
except ImportError as e:
    print(f"Warning: Could not import task planner: {e}")
    # Create a fallback task planner
    class MockTaskPlanner:
        def create_task_plan(self, assignment_description, due_date, steps=5):
            return [
                {"title": "Start Assignment", "description": "Begin working on the assignment", "deadline_percentage": 20},
                {"title": "Research Phase", "description": "Gather information and resources", "deadline_percentage": 40},
                {"title": "Draft Work", "description": "Create initial draft", "deadline_percentage": 60},
                {"title": "Review and Edit", "description": "Review and improve the work", "deadline_percentage": 80},
                {"title": "Final Submission", "description": "Complete and submit assignment", "deadline_percentage": 100}
            ]
    task_planner = MockTaskPlanner()

router = APIRouter()
logger = logging.getLogger(__name__)

class TaskCreateRequest(BaseModel):
    title: str
    description: str
    due_date: str

    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Title must be a non-empty string")
        return v.strip()

    @field_validator('due_date')
    @classmethod
    def validate_due_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Due date must be in YYYY-MM-DD format")

class TaskResponse(BaseModel):
    id: int
    title: str
    description: str
    due_date: str
    status: str
    created_at: datetime
    plan_id: Optional[int] = None

class PlanCreateRequest(BaseModel):
    name: str
    description: str
    assignment_description: str
    due_date: str

class PlanResponse(BaseModel):
    id: int
    name: str
    description: str
    assignment_description: str
    due_date: str
    created_at: datetime

class TaskPlanRequest(BaseModel):
    assignment_description: str
    due_date: str
    steps: Optional[int] = 5

    @field_validator('assignment_description')
    @classmethod
    def validate_assignment_description(cls, v):
        if not v or not isinstance(v, str) or len(v.strip()) == 0:
            raise ValueError("Assignment description must be a non-empty string")
        return v.strip()

    @field_validator('due_date')
    @classmethod
    def validate_due_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Due date must be in YYYY-MM-DD format")

    @field_validator('steps')
    @classmethod
    def validate_steps(cls, v):
        if v is not None and (v < 1 or v > 20):
            raise ValueError("Steps must be between 1 and 20")
        return v

class TaskPlanItem(BaseModel):
    title: str
    description: str
    deadline_percentage: int

class TaskPlanResponse(BaseModel):
    tasks: List[TaskPlanItem]
    success: bool

@router.post("/plans", response_model=PlanResponse)
async def create_plan(plan: PlanCreateRequest):
    try:
        async with get_db_session() as db:
            new_plan = Plan(
                name=plan.name,
                description=plan.description,
                assignment_description=plan.assignment_description,
                due_date=plan.due_date
            )
            db.add(new_plan)
            await db.commit()
            await db.refresh(new_plan)
            return new_plan
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating plan: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/plans", response_model=List[PlanResponse])
async def get_plans():
    try:
        async with get_db_session() as db:
            result = await db.execute(select(Plan))
            plans = result.scalars().all()
            return plans
    except Exception as e:
        logger.error(f"Error retrieving plans: {str(e)}")
        return []

@router.get("/plans/{plan_id}", response_model=PlanResponse)
async def get_plan(plan_id: int):
    try:
        async with get_db_session() as db:
            result = await db.execute(select(Plan).where(Plan.id == plan_id))
            plan = result.scalar_one_or_none()
            if not plan:
                raise HTTPException(status_code=404, detail="Plan not found")
            return plan
    except Exception as e:
        logger.error(f"Error retrieving plan: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/plans/{plan_id}/tasks", response_model=List[TaskResponse])
async def get_plan_tasks(plan_id: int):
    try:
        async with get_db_session() as db:
            result = await db.execute(select(Task).where(Task.plan_id == plan_id))
            tasks = result.scalars().all()
            return tasks
    except Exception as e:
        logger.error(f"Error retrieving plan tasks: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/plans/{plan_id}")
async def delete_plan(plan_id: int):
    try:
        async with get_db_session() as db:
            result = await db.execute(select(Plan).where(Plan.id == plan_id))
            plan = result.scalar_one_or_none()
            if not plan:
                raise HTTPException(status_code=404, detail="Plan not found")
            
            await db.execute(delete(Task).where(Task.plan_id == plan_id))
            await db.delete(plan)
            await db.commit()
            return {"message": "Plan and associated tasks deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting plan: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/", response_model=TaskResponse)
async def create_task(task: TaskCreateRequest, plan_id: Optional[int] = None):
    try:
        async with get_db_session() as db:
            new_task = Task(
                title=task.title,
                description=task.description,
                due_date=task.due_date,
                plan_id=plan_id
            )
            db.add(new_task)
            await db.commit()
            await db.refresh(new_task)
            return new_task
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/", response_model=List[TaskResponse])
async def get_tasks():
    try:
        async with get_db_session() as db:
            result = await db.execute(select(Task))
            tasks = result.scalars().all()
            return tasks
    except Exception as e:
        logger.error(f"Error retrieving tasks: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/{task_id}/status")
async def update_task_status(task_id: int, status: str):
    try:
        async with get_db_session() as db:
            result = await db.execute(select(Task).where(Task.id == task_id))
            task = result.scalar_one_or_none()
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task.status = status
            await db.commit()
            return {"message": "Task status updated successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating task status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{task_id}")
async def delete_task(task_id: int):
    try:
        async with get_db_session() as db:
            result = await db.execute(select(Task).where(Task.id == task_id))
            task = result.scalar_one_or_none()
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            
            await db.delete(task)
            await db.commit()
            return {"message": "Task deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting task: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/plan", response_model=TaskPlanResponse)
async def generate_task_plan(request: TaskPlanRequest):
    """
    Generate an AI-powered task plan for an assignment and save it to database
    """
    try:
        logger.info(f"Generating task plan for assignment: {request.assignment_description[:100]}")

        # Use AI agent to generate the task plan
        ai_tasks = task_planner.create_task_plan(
            assignment_description=request.assignment_description,
            due_date=request.due_date,
            steps=request.steps or 5
        )

        logger.info(f"Task plan generated successfully with {len(ai_tasks)} tasks")

        async with get_db_session() as db:
            # Create a plan entry
            plan_name = f"AI Plan: {request.assignment_description[:50]}{'...' if len(request.assignment_description) > 50 else ''}"
            plan_description = f"AI-generated task plan for: {request.assignment_description[:100]}{'...' if len(request.assignment_description) > 100 else ''}"

            new_plan = Plan(
                name=plan_name,
                description=plan_description,
                assignment_description=request.assignment_description,
                due_date=request.due_date
            )
            db.add(new_plan)
            await db.commit()
            await db.refresh(new_plan)

            # Create tasks associated with this plan
            created_tasks = []
            for ai_task in ai_tasks:
                # Calculate actual due date based on deadline_percentage
                due_date_obj = datetime.strptime(request.due_date, "%Y-%m-%d")
                today = datetime.now()

                # Safely get and convert deadline_percentage
                deadline_pct = ai_task.get('deadline_percentage', 100)
                try:
                    deadline_pct = int(deadline_pct)
                except (ValueError, TypeError):
                    deadline_pct = 100

                if deadline_pct > 0:
                    # Calculate days from today based on percentage
                    total_days = (due_date_obj - today).days
                    task_days = int(total_days * deadline_pct / 100)
                    task_due_date = today + timedelta(days=task_days)
                    task_due_str = task_due_date.strftime("%Y-%m-%d")
                else:
                    task_due_str = request.due_date

                new_task = Task(
                    title=ai_task.get('title', 'Untitled Task'),
                    description=ai_task.get('description', 'No description provided'),
                    due_date=task_due_str,
                    plan_id=new_plan.id
                )
                db.add(new_task)
                created_tasks.append(new_task)

            await db.commit()

            # Refresh all tasks to get their IDs
            for task in created_tasks:
                await db.refresh(task)

            # Convert to response format - use the calculated percentages from the loop above
            task_items = []
            for i, task in enumerate(created_tasks):
                # Use the percentage we calculated earlier in the loop
                deadline_pct = 100  # Default fallback
                if i < len(ai_tasks):
                    ai_task = ai_tasks[i]
                    try:
                        deadline_pct = int(ai_task.get('deadline_percentage', 100))
                    except (ValueError, TypeError):
                        deadline_pct = 100

                task_items.append(TaskPlanItem(
                    title=task.title,
                    description=task.description,
                    deadline_percentage=deadline_pct
                ))

        logger.info(f"Task plan saved to database with plan ID {new_plan.id} and {len(created_tasks)} tasks")

        return TaskPlanResponse(
            tasks=task_items,
            success=True
        )

    except Exception as e:
        logger.error(f"Task plan generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating task plan: {str(e)}")

@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    return {"status": "healthy", "service": "tasks"}
