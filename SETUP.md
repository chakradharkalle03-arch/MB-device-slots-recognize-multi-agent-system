# Setup Guide

## Quick Start

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# The .env file is already configured with your Hugging Face API key
# If needed, create .env file:
# HUGGINGFACE_API_KEY=your_huggingface_api_key_here
# CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# Start backend server
python main.py
```

Backend will run on: `http://localhost:8000`

### 2. Frontend Setup

Open a new terminal:

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Start frontend server
npm start
```

Frontend will run on: `http://localhost:3000`

### 3. Using the Application

1. Open browser: `http://localhost:3000`
2. Upload a motherboard image (PNG, JPG, JPEG)
3. Enter task description (e.g., "Install Keyboard", "Install RAM")
4. Click "Generate SOP"
5. Wait for processing (may take 30-60 seconds)
6. Review generated SOP steps and explanations
7. Click "Download PDF Report" to get the complete PDF

## Windows Quick Start (Using Batch Files)

1. Double-click `start_backend.bat` to start backend
2. Double-click `start_frontend.bat` to start frontend (in a new window)
3. Open browser to `http://localhost:3000`

## Troubleshooting

### Backend Issues

**Import errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

**Vision model not loading:**
- Check internet connection (first run downloads models)
- Verify Hugging Face API key in config.py or .env
- System will fallback to rule-based detection if model fails

**Port already in use:**
- Change port in `main.py`: `uvicorn.run(app, host="0.0.0.0", port=8001)`

### Frontend Issues

**Cannot connect to backend:**
- Verify backend is running on port 8000
- Check CORS settings in `backend/config.py`
- Update BACKEND_URL in `frontend/public/index.html` if backend is on different port

**npm install fails:**
- Ensure Node.js 14+ is installed
- Try: `npm install --legacy-peer-deps`

## Project Structure

```
MB/
├── backend/
│   ├── agents/          # AI agents (Vision, Knowledge, SOP, etc.)
│   ├── uploads/         # Uploaded images (created automatically)
│   ├── outputs/         # Generated outputs (created automatically)
│   │   ├── annotated/   # Annotated images
│   │   └── pdfs/        # Generated PDF reports
│   ├── main.py          # FastAPI application
│   ├── orchestrator.py  # Multi-agent orchestrator
│   ├── config.py        # Configuration
│   └── requirements.txt # Python dependencies
├── frontend/
│   ├── public/
│   │   └── index.html   # Frontend UI
│   ├── server.js        # Express server
│   └── package.json     # Node.js dependencies
├── start_backend.bat    # Windows backend starter
├── start_frontend.bat   # Windows frontend starter
└── README.md            # Project documentation
```

## API Endpoints

### POST `/api/generate-sop`
Generate SOP from image and task

**Request:**
- `image`: Image file (multipart/form-data)
- `task`: Task description (form field)

**Response:**
```json
{
  "status": "completed",
  "task": "Install Keyboard",
  "target_component": {...},
  "sop_steps": [...],
  "explanations": [...],
  "qa_result": {...},
  "pdf_path": "sop_report_xxx.pdf"
}
```

### GET `/api/download-pdf/{filename}`
Download generated PDF report

### GET `/api/download-annotated/{filename}`
Download annotated image

## Notes

- First run may take longer as Hugging Face models are downloaded
- Vision detection uses DETR model; falls back to rule-based if model unavailable
- PDF generation requires ReportLab library
- All file paths are handled relative to backend directory

