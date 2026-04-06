from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

# Create the FastAPI application
app = FastAPI(
    title="Local LLM Inference API",
    description="API for serving language models locally",
    version="1.0.0"
)

# Health check endpoint
@app.get("/health")
def health_check():
    """
    Returns the status of the API server.
    Use this endpoint to verify the server is running.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "API server is running"
    }

# Root endpoint
@app.get("/")
def root():
    """
    Root endpoint with basic information.
    """
    return {
        "service": "Local LLM Inference API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/", "method": "GET", "description": "Service information"}
        ]
    }