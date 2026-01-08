#!/usr/bin/env python3
import os
import sys

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add backend directory to path for relative imports
backend_path = os.path.join(project_root, "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Import and run the backend
from backend.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
