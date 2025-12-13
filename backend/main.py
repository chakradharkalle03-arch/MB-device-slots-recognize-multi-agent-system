"""
FastAPI Backend - Main API endpoints
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import uuid
from orchestrator import Orchestrator
from config import CORS_ORIGINS

app = FastAPI(title="Manufacturing SOP Automation API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = Orchestrator()

class TaskRequest(BaseModel):
    task: str

class SOPResponse(BaseModel):
    status: str
    task: str
    target_component: dict
    sop_steps: list
    explanations: list
    qa_result: dict
    pdf_path: Optional[str] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Manufacturing SOP Automation API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}

@app.post("/api/generate-sop", response_model=SOPResponse)
async def generate_sop(
    image: UploadFile = File(...),
    task: str = Form(None)
):
    """
    Generate SOP from motherboard image (task automatically detected)
    
    Args:
        image: Motherboard image file
        task: Optional task description (if not provided, will be auto-detected)
    
    Returns:
        Complete SOP data including PDF path
    """
    try:
        # Validate file type
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file
        file_extension = os.path.splitext(image.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        upload_path = os.path.join(upload_dir, unique_filename)
        
        with open(upload_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # Execute workflow (task is optional, will be auto-detected if not provided)
        result = orchestrator.execute_workflow(upload_path, task if task else None)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
        # Prepare response
        response_data = {
            "status": result["status"],
            "task": result["task"],
            "target_component": result.get("target_component", {}),
            "all_connectors": result.get("all_components_enriched", []),
            "sop_steps": result.get("sop_data", {}).get("sop_steps", []),
            "explanations": result.get("explanations", []),
            "qa_result": result.get("qa_result", {}),
            "pdf_path": result.get("pdf_path", ""),
            "error": result.get("error")
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/download-pdf/{filename}")
async def download_pdf(filename: str):
    """
    Download generated PDF report
    
    Args:
        filename: PDF filename
    
    Returns:
        PDF file
    """
    pdf_path = os.path.join(os.path.dirname(__file__), "outputs", "pdfs", filename)
    
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=filename
    )

@app.get("/api/download-annotated/{filename}")
async def download_annotated(filename: str):
    """
    Download annotated image
    
    Args:
        filename: Image filename
    
    Returns:
        Annotated image file
    """
    image_path = os.path.join(os.path.dirname(__file__), "outputs", "annotated", filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Annotated image not found")
    
    return FileResponse(image_path, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

