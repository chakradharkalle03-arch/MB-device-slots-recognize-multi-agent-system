"""
FastAPI Backend - Main API endpoints
"""
import warnings
# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")
warnings.filterwarnings("ignore", message=".*use_fast.*")

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

@app.get("/api/status")
async def get_system_status():
    """
    Get comprehensive system status including all agents and models
    """
    import torch
    from config import VISION_MODEL, LANGUAGE_MODEL, HUGGINGFACE_API_KEY
    
    status = {
        "backend": {
            "status": "running",
            "version": "1.0.0",
            "port": 8001,
            "framework": "FastAPI"
        },
        "agents": {},
        "models": {},
        "system": {}
    }
    
    # Check Vision Agent
    try:
        vision_status = "healthy"
        vision_model_loaded = orchestrator.vision_agent.model is not None
        vision_device = orchestrator.vision_agent.device
        status["agents"]["vision_agent"] = {
            "status": vision_status,
            "model_loaded": vision_model_loaded,
            "device": vision_device,
            "model_name": VISION_MODEL,
            "capabilities": ["Component Detection", "Image Annotation", "Bounding Box Detection"]
        }
    except Exception as e:
        status["agents"]["vision_agent"] = {
            "status": "error",
            "error": str(e),
            "model_name": VISION_MODEL
        }
    
    # Check Knowledge Agent
    try:
        knowledge_db_size = len(orchestrator.knowledge_agent.component_database)
        status["agents"]["knowledge_agent"] = {
            "status": "healthy",
            "component_database_size": knowledge_db_size,
            "capabilities": ["Component Mapping", "Hardware Knowledge", "Risk Assessment"]
        }
    except Exception as e:
        status["agents"]["knowledge_agent"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check Task Detection Agent
    try:
        task_patterns = len(orchestrator.task_detection_agent.task_patterns)
        status["agents"]["task_detection_agent"] = {
            "status": "healthy",
            "supported_tasks": task_patterns,
            "capabilities": ["Auto Task Detection", "Component Analysis", "Pattern Matching"]
        }
    except Exception as e:
        status["agents"]["task_detection_agent"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check SOP Planning Agent
    try:
        sop_templates = len(orchestrator.sop_planning_agent.step_templates)
        status["agents"]["sop_planning_agent"] = {
            "status": "healthy",
            "sop_templates": sop_templates,
            "capabilities": ["SOP Generation", "Step Planning", "Safety Protocol Integration"]
        }
    except Exception as e:
        status["agents"]["sop_planning_agent"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check Explanation Agent
    try:
        explanation_templates = len(orchestrator.explanation_agent.explanation_templates)
        status["agents"]["explanation_agent"] = {
            "status": "healthy",
            "explanation_templates": explanation_templates,
            "capabilities": ["Step Explanations", "Why/How Details", "Common Mistakes"]
        }
    except Exception as e:
        status["agents"]["explanation_agent"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check QA Agent
    try:
        required_elements = len(orchestrator.qa_agent.required_elements)
        status["agents"]["qa_agent"] = {
            "status": "healthy",
            "required_elements": required_elements,
            "capabilities": ["SOP Validation", "Safety Checks", "Quality Assurance"]
        }
    except Exception as e:
        status["agents"]["qa_agent"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check PDF Generator Agent
    try:
        status["agents"]["pdf_generator_agent"] = {
            "status": "healthy",
            "capabilities": ["PDF Report Generation", "Image Integration", "Professional Formatting"]
        }
    except Exception as e:
        status["agents"]["pdf_generator_agent"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Model Information
    status["models"] = {
        "vision_model": {
            "name": VISION_MODEL,
            "provider": "Hugging Face",
            "type": "Object Detection (DETR)",
            "architecture": "Detection Transformer (ResNet-50)",
            "status": "configured",
            "api_key_configured": bool(HUGGINGFACE_API_KEY and HUGGINGFACE_API_KEY != "your_huggingface_api_key_here")
        },
        "language_model": {
            "name": LANGUAGE_MODEL,
            "provider": "Hugging Face",
            "type": "DialoGPT",
            "status": "configured",
            "note": "Available for future enhancements"
        }
    }
    
    # System Information
    status["system"] = {
        "python_version": f"{torch.__version__ if 'torch' in dir() else 'N/A'}",
        "pytorch_available": torch.cuda.is_available() if 'torch' in dir() else False,
        "cuda_available": torch.cuda.is_available() if 'torch' in dir() else False,
        "device": orchestrator.vision_agent.device if hasattr(orchestrator.vision_agent, 'device') else "cpu"
    }
    
    # Calculate overall health
    agent_statuses = [agent.get("status") for agent in status["agents"].values()]
    healthy_count = sum(1 for s in agent_statuses if s == "healthy")
    total_agents = len(status["agents"])
    
    status["overall"] = {
        "health": "healthy" if healthy_count == total_agents else "degraded",
        "healthy_agents": healthy_count,
        "total_agents": total_agents,
        "health_percentage": round((healthy_count / total_agents) * 100, 1) if total_agents > 0 else 0
    }
    
    return JSONResponse(content=status)

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
        
        # Prepare response with all connectors and their SOPs
        all_connectors_data = []
        for i, sop_item in enumerate(result.get("all_sop_data", [])):
            component = sop_item.get("component", {})
            sop_data = sop_item.get("sop_data", {})
            explanations = result.get("all_explanations", [])[i].get("explanations", []) if i < len(result.get("all_explanations", [])) else []
            qa_result = result.get("all_qa_results", [])[i].get("qa_result", {}) if i < len(result.get("all_qa_results", [])) else {}
            
            all_connectors_data.append({
                "component": component,
                "sop_steps": sop_data.get("sop_steps", []),
                "explanations": explanations,
                "qa_result": qa_result
            })
        
        response_data = {
            "status": result["status"],
            "task": result.get("task", "Analyze All Connectors"),
            "target_component": result.get("target_component", {}),
            "all_connectors": result.get("all_components_enriched", []),
            "all_connectors_with_sops": all_connectors_data,  # New: All connectors with their SOPs
            "sop_steps": result.get("sop_data", {}).get("sop_steps", []),  # Primary SOP (backward compatibility)
            "explanations": result.get("explanations", []),  # Primary explanations (backward compatibility)
            "qa_result": result.get("qa_result", {}),  # Primary QA (backward compatibility)
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

