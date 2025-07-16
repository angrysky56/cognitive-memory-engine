"""
Task Management for Non-blocking MCP Operations

This module provides a task management system that allows MCP tools to return
immediately while long-running operations continue in the background.
"""

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskInfo:
    """Information about a background task"""
    task_id: str
    status: TaskStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any | None = None
    error: str | None = None
    progress: float = 0.0
    description: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, TaskStatus):
                data[key] = value.value
        return data


class TaskManager:
    """Manages background tasks for non-blocking operations"""

    def __init__(self):
        self.tasks: dict[str, TaskInfo] = {}
        self.running_tasks: dict[str, asyncio.Task] = {}

    def create_task(self, description: str = "") -> str:
        """Create a new task and return its ID"""
        task_id = str(uuid.uuid4())
        task_info = TaskInfo(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            description=description
        )
        self.tasks[task_id] = task_info
        return task_id

    async def start_background_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        **kwargs
    ) -> None:
        """Start a background task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task_info = self.tasks[task_id]
        task_info.status = TaskStatus.RUNNING
        task_info.started_at = datetime.now()

        # Create the actual async task
        async def task_wrapper():
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Update task info on success
                task_info.status = TaskStatus.COMPLETED
                task_info.completed_at = datetime.now()
                task_info.result = result
                task_info.progress = 1.0

            except Exception as e:
                # Update task info on failure
                task_info.status = TaskStatus.FAILED
                task_info.completed_at = datetime.now()
                task_info.error = str(e)
                task_info.progress = 1.0

            finally:
                # Clean up the running task reference
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]

        # Start the background task
        background_task = asyncio.create_task(task_wrapper())
        self.running_tasks[task_id] = background_task

    def get_task_status(self, task_id: str) -> TaskInfo | None:
        """Get the current status of a task"""
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> dict[str, TaskInfo]:
        """Get all tasks"""
        return self.tasks.copy()

    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> None:
        """Clean up old completed tasks"""
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)

        to_remove = []
        for task_id, task_info in self.tasks.items():
            if (task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] and
                task_info.completed_at and
                task_info.completed_at.timestamp() < cutoff):
                to_remove.append(task_id)

        for task_id in to_remove:
            del self.tasks[task_id]


# Global task manager instance
task_manager = TaskManager()


def create_immediate_response(task_id: str, description: str) -> dict:
    """Create an immediate response for non-blocking operations"""
    return {
        "status": "accepted",
        "task_id": task_id,
        "message": f"Task started: {description}",
        "timestamp": datetime.now().isoformat(),
        "note": "This operation is running in the background. Use the task_id to check status."
    }
