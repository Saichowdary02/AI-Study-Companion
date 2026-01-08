"""
Simple middleware for local development
"""
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Simple request logging middleware"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        logger.info(f"{request.method} {request.url.path}")

        response = await call_next(request)

        logger.info(f"Response: {response.status_code}")
        return response
