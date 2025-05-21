"""
FastAPI application for the AudioBookSum book pipeline.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import our modules
from src.logging import configure_logging, get_logger
from src.models import PipelineResult
import main as pipeline_main
from src.config import OUTPUT_DIR, DATA_DIR
from src.metrics import start_metrics_server

# Configure logging
logger = configure_logging()

# Start metrics server
metrics_started = start_metrics_server(8001)
if metrics_started:
    logger.info("Metrics server started", port=8001)
else:
    logger.warning("Failed to start metrics server")

# Create FastAPI app
app = FastAPI(
    title="AudioBookSum API",
    description="API for processing PDFs and extracting key points",
    version="1.0.0",
)

# Serve static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="api/templates")

# Background tasks
running_tasks = {}

class PipelineRequest(BaseModel):
    """Request model for pipeline processing."""
    output_dir: Optional[str] = OUTPUT_DIR
    use_mistral: bool = False
    use_chunking_v1: bool = False
    chunk_size: int = 50000
    chunk_overlap: int = 100
    save_intermediate: bool = True


class TaskStatus(BaseModel):
    """Status model for background tasks."""
    id: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    started_at: float
    completed_at: Optional[float] = None


@app.get("/")
async def read_root():
    """API root endpoint."""
    return {
        "message": "Welcome to AudioBookSum API",
        "endpoints": {
            "POST /process": "Process a PDF file",
            "GET /status/{task_id}": "Get task status",
            "GET /download/{filename}": "Download a processed file"
        }
    }


async def process_pdf_task(
    task_id: str, 
    file_path: str,
    request: PipelineRequest
):
    """
    Background task for processing a PDF file.
    
    Args:
        task_id: Task ID
        file_path: Path to the PDF file
        request: Pipeline request parameters
    """
    try:
        running_tasks[task_id]["status"] = "running"
        running_tasks[task_id]["message"] = "Processing PDF..."
        
        # Process the PDF
        result = pipeline_main.process_pdf(
            pdf_path=file_path,
            output_dir=request.output_dir,
            use_mistral=request.use_mistral,
            use_chunking_v2=not request.use_chunking_v1,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            save_intermediate=request.save_intermediate
        )
        
        # Update task status
        if result.status == "success":
            running_tasks[task_id]["status"] = "completed"
            running_tasks[task_id]["message"] = "PDF processed successfully"
            running_tasks[task_id]["result"] = {
                "summary": result.summary,
                "evidence_count": len(result.evidence) if result.evidence else 0,
                "highlighted_pdf": os.path.basename(result.output_file) if result.output_file else None,
                "processing_time": result.processing_time
            }
        else:
            running_tasks[task_id]["status"] = "failed"
            running_tasks[task_id]["message"] = f"Processing failed: {result.error}"
    
    except Exception as e:
        logger.exception("Task failed", task_id=task_id, error=str(e))
        running_tasks[task_id]["status"] = "failed"
        running_tasks[task_id]["message"] = f"Error: {str(e)}"
    
    finally:
        running_tasks[task_id]["completed_at"] = time.time()


@app.post("/process")
async def process_pdf(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    request: PipelineRequest = PipelineRequest()
):
    """
    Process a PDF file.
    
    Args:
        background_tasks: FastAPI background tasks
        pdf_file: Uploaded PDF file
        request: Pipeline request parameters
        
    Returns:
        Task ID for checking status
    """
    try:
        # Create upload directory if it doesn't exist
        upload_dir = os.path.join(DATA_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(upload_dir, pdf_file.filename)
        with open(file_path, "wb") as f:
            f.write(await pdf_file.read())
        
        # Create task ID
        task_id = f"task_{int(time.time())}_{Path(pdf_file.filename).stem}"
        
        # Create task status
        running_tasks[task_id] = {
            "id": task_id,
            "status": "queued",
            "message": "Task queued",
            "started_at": time.time(),
            "completed_at": None,
            "result": None
        }
        
        # Start background task
        background_tasks.add_task(
            process_pdf_task,
            task_id=task_id,
            file_path=file_path,
            request=request
        )
        
        return {"task_id": task_id, "message": "PDF processing started"}
    
    except Exception as e:
        logger.exception("Failed to start processing", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get task status.
    
    Args:
        task_id: Task ID
        
    Returns:
        Task status information
    """
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return running_tasks[task_id]


@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download a processed file.
    
    Args:
        filename: Filename to download
        
    Returns:
        File response
    """
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )


@app.get("/list_files")
async def list_files():
    """
    List all processed files.
    
    Returns:
        List of processed files
    """
    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        return {"files": []}
    
    # List files in output directory
    files = os.listdir(OUTPUT_DIR)
    
    # Filter for PDF, JSON, and TXT files
    files = [f for f in files if f.endswith((".pdf", ".json", ".txt"))]
    
    # Group files by base name
    grouped_files = {}
    for file in files:
        base_name = Path(file).stem.split("_")[0]
        
        if base_name not in grouped_files:
            grouped_files[base_name] = []
        
        grouped_files[base_name].append({
            "filename": file,
            "size": os.path.getsize(os.path.join(OUTPUT_DIR, file)),
            "created": os.path.getctime(os.path.join(OUTPUT_DIR, file))
        })
    
    return {"files": grouped_files}