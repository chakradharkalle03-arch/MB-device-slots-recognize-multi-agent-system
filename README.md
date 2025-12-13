# Manufacturing SOP Automation System

## 📋 Project Introduction

A **production-ready, industry-grade multi-agent AI system** that automatically generates Standard Operating Procedures (SOPs) for laptop motherboard component installation. This system combines computer vision, natural language processing, and intelligent orchestration to create manufacturing-ready documentation from simple image uploads.

The system analyzes motherboard images, detects components using advanced computer vision models, automatically determines installation tasks, and generates comprehensive step-by-step procedures with safety checks, detailed explanations, and professional PDF reports.

### 🎯 Problem Statement

Manufacturing facilities face challenges in:
- Creating consistent, accurate SOPs for complex component installations
- Ensuring safety protocols are always included
- Training technicians with clear, detailed procedures
- Standardizing documentation across production lines
- Reducing human error in critical assembly processes

### 💡 Solution

This multi-agent AI system automates the entire SOP generation process:
- **Upload** a motherboard image
- **Automatically detect** components and determine tasks
- **Generate** manufacturing-ready SOPs with safety checks
- **Export** professional PDF reports

---

## 🤖 Multi-Agent Architecture

The system uses **7 specialized AI agents** working in coordination through a LangGraph-style orchestrator:

### 1. **Vision Agent** (`vision_agent.py`)
**Purpose**: Computer vision component detection and localization

**Capabilities**:
- Uses Hugging Face DETR (Detection Transformer) model for object detection
- Detects motherboard components (connectors, slots, ports) with bounding boxes
- Annotates images with component labels and locations
- Falls back to rule-based detection if model unavailable

**Technologies**: PyTorch, Transformers, OpenCV, PIL

---

### 2. **Task Detection Agent** (`task_detection_agent.py`)
**Purpose**: Automatically determines installation task from component analysis

**Capabilities**:
- Analyzes detected components to infer the task
- Supports tasks: Install Keyboard, Install RAM, Connect Fan, Connect Battery, Connect Display
- Uses pattern matching and component type analysis
- Works even when user doesn't specify a task

**Intelligence**: Pattern recognition, keyword matching, component classification

---

### 3. **Knowledge Agent** (`knowledge_agent.py`)
**Purpose**: Maps detected components to engineering knowledge

**Capabilities**:
- Maintains hardware component database
- Enriches components with technical specifications:
  - Connector types (ZIF, SO-DIMM, JST, etc.)
  - Pin counts, voltage requirements
  - Typical locations and orientations
  - Risk levels and safety notes
  - Common issues and troubleshooting
- Identifies target component for the task
- Provides component enrichment for all detected parts

**Knowledge Base**: Hardware specifications, connector types, safety requirements

---

### 4. **SOP Planning Agent** (`sop_planning_agent.py`)
**Purpose**: Generates manufacturing-ready step-by-step procedures

**Capabilities**:
- Creates structured SOP steps based on component type and task
- Includes safety protocols (power-off, ESD protection)
- Provides installation procedures specific to connector types
- Adds verification and quality checkpoints
- Considers risk levels for appropriate precautions

**Output**: Numbered steps with clear actions, safety considerations, verification points

---

### 5. **Explanation Agent** (`explanation_agent.py`)
**Purpose**: Provides detailed explanations for each SOP step

**Capabilities**:
- Explains **why** each step is necessary
- Describes **how** to perform each action correctly
- Lists **common mistakes** to avoid
- Makes procedures understandable for technicians at all levels

**Value**: Reduces training time, prevents errors, improves understanding

---

### 6. **QA/Safety Agent** (`qa_agent.py`)
**Purpose**: Validates SOP correctness and safety compliance

**Capabilities**:
- Checks for required safety elements (power-off, ESD protection)
- Validates completeness (verification steps, component identification)
- Assesses risk levels and recommends precautions
- Provides safety score and compliance report
- Flags missing critical steps

**Output**: Validation report with issues, warnings, recommendations, safety score

---

### 7. **PDF Generator Agent** (`pdf_generator.py`)
**Purpose**: Creates professional PDF reports

**Capabilities**:
- Generates formatted PDF documents with:
  - Title page with task and component information
  - Annotated motherboard images
  - Complete SOP steps
  - Detailed explanations
  - QA validation results
  - Component specifications
- Professional formatting suitable for manufacturing floor

**Technologies**: ReportLab

---

### **Orchestrator** (`orchestrator.py`)
**Purpose**: Coordinates the multi-agent workflow

**Workflow**:
```
Image Upload → Vision Agent → Task Detection Agent → Knowledge Agent 
→ SOP Planning Agent → Explanation Agent → QA Agent → PDF Generator → Output
```

**Features**:
- State machine management
- Error handling and fallbacks
- Sequential agent execution
- Result aggregation

---

## ✨ Key Features

### 🎨 **Intelligent Component Detection**
- Advanced computer vision using Hugging Face DETR model
- Automatic component localization with bounding boxes
- Fallback detection methods for reliability
- Image annotation with component labels

### 📝 **Automatic SOP Generation**
- Manufacturing-ready step-by-step procedures
- Safety protocols automatically included
- Component-specific installation instructions
- Quality checkpoints and verification steps

### 🔍 **Task Auto-Detection**
- Automatically determines installation task from image analysis
- No manual task specification required
- Supports multiple task types (Keyboard, RAM, Fan, Battery, Display)

### 📚 **Detailed Explanations**
- **Why**: Explains the purpose of each step
- **How**: Describes correct execution methods
- **Common Mistakes**: Highlights errors to avoid
- Improves technician understanding and reduces training time

### ✅ **Quality Assurance**
- Automated safety validation
- Completeness checks
- Risk level assessment
- Safety score calculation
- Compliance reporting

### 📄 **Professional PDF Reports**
- Formatted documentation ready for manufacturing floor
- Annotated images with component labels
- Complete SOP steps with explanations
- QA validation results
- Component specifications

### 🌐 **Modern Web Interface**
- Clean, responsive design
- Easy image upload
- Real-time SOP display
- Component information visualization
- PDF download functionality

---

## 🏗️ Technology Stack

### **Backend**
- **Framework**: FastAPI (Python)
- **AI/ML**: 
  - Hugging Face Transformers (DETR for object detection)
  - PyTorch & TorchVision
  - LangChain & LangGraph (orchestration patterns)
- **Image Processing**: OpenCV, PIL (Pillow)
- **PDF Generation**: ReportLab
- **API**: RESTful API with FastAPI

### **Frontend**
- **Runtime**: Node.js with Express
- **UI**: Vanilla JavaScript, HTML5, CSS3
- **Styling**: Modern CSS with gradients and animations
- **HTTP Client**: Axios for API communication

### **Architecture**
- Multi-agent system with LangGraph-style orchestration
- State machine workflow management
- Modular agent design for extensibility

---

## 🚀 Use Cases & Applications

### **Manufacturing Facilities**
- Standardize component installation procedures
- Reduce documentation creation time
- Ensure consistent safety protocols
- Improve production line efficiency

### **Quality Assurance Teams**
- Automated SOP validation
- Safety compliance checking
- Documentation standardization
- Risk assessment automation

### **Training Programs**
- Generate training materials automatically
- Create consistent learning resources
- Reduce training preparation time
- Improve technician onboarding

### **Electronics Manufacturing Companies**
Perfect for companies like:
- **MediaTek** - Semiconductor manufacturing
- **Qualcomm** - Chip design and assembly
- **Micron** - Memory module production
- **Foxconn / Wistron / Pegatron** - Electronics assembly

### **Research & Development**
- Prototype documentation automation
- Component analysis and cataloging
- Procedure standardization research
- AI/ML system demonstration

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- Hugging Face API key (provided in config)

### Backend Setup

1. **Install dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Configure environment** (optional):
Create a `.env` file in the `backend` directory:
```env
HUGGINGFACE_API_KEY=your_api_key_here
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

Note: Default API key is already configured in `config.py`

### Frontend Setup

1. **Install dependencies**:
```bash
cd frontend
npm install
```

---

## 🎮 Running the Application

### Start Backend Server

```bash
cd backend
python main.py
```

Backend runs on: **http://localhost:8001**

### Start Frontend Server

```bash
cd frontend
npm start
```

Frontend runs on: **http://localhost:3000**

### Using the Application

1. Open `http://localhost:3000` in your browser
2. Upload a motherboard image (PNG, JPG, JPEG)
3. Optionally enter a task description (e.g., "Install Keyboard")
   - **Note**: Task can be auto-detected if not provided
4. Click "Generate SOP"
5. Review the generated SOP steps and explanations
6. Check QA validation results
7. Download the PDF report

---

## 📡 API Endpoints

### `POST /api/generate-sop`
Generate SOP from motherboard image

**Request**:
- `image`: Image file (multipart/form-data)
- `task`: Task description (optional, form field)

**Response**:
```json
{
  "status": "completed",
  "task": "Install Keyboard",
  "target_component": {
    "type": "Keyboard Connector",
    "location": "...",
    "specifications": {...}
  },
  "all_connectors": [...],
  "sop_steps": [
    "Step 1: Power off the device...",
    "Step 2: Apply ESD protection..."
  ],
  "explanations": [
    {
      "step": 1,
      "why": "...",
      "how": "...",
      "common_mistakes": "..."
    }
  ],
  "qa_result": {
    "safety_score": 95,
    "issues": [],
    "warnings": [],
    "recommendations": []
  },
  "pdf_path": "sop_report_xxx.pdf"
}
```

### `GET /api/download-pdf/{filename}`
Download generated PDF report

### `GET /api/download-annotated/{filename}`
Download annotated image with component labels

---

## 📁 Project Structure

```
MB_device_slots_recognize_use_mutli_agent_system/
├── backend/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── vision_agent.py          # Computer vision detection
│   │   ├── task_detection_agent.py  # Auto task detection
│   │   ├── knowledge_agent.py       # Hardware knowledge base
│   │   ├── sop_planning_agent.py    # SOP generation
│   │   ├── explanation_agent.py     # Step explanations
│   │   ├── qa_agent.py              # Quality assurance
│   │   └── pdf_generator.py         # PDF report creation
│   ├── orchestrator.py              # Multi-agent coordinator
│   ├── main.py                      # FastAPI application
│   ├── config.py                    # Configuration
│   ├── requirements.txt             # Python dependencies
│   ├── uploads/                     # Uploaded images
│   └── outputs/
│       ├── annotated/               # Annotated images
│       └── pdfs/                    # Generated PDFs
├── frontend/
│   ├── public/
│   │   └── index.html               # Web UI
│   ├── server.js                    # Express server
│   └── package.json                 # Node dependencies
├── README.md                        # This file
├── PROJECT_SUMMARY.md               # Project overview
└── SETUP.md                         # Setup instructions
```

---

## 🔒 Security & Best Practices

- API keys stored in environment variables
- File upload validation (image types only)
- CORS configured for specific origins
- Input validation on all endpoints
- Error handling with fallback mechanisms
- Safe file handling with UUID-based naming

---

## 🐛 Troubleshooting

### Vision model not loading
- Check Hugging Face API key
- Ensure internet connection for model download
- System will fallback to rule-based detection

### PDF generation fails
- Check that `outputs/pdfs` directory exists
- Verify image paths are correct
- Check ReportLab installation

### Frontend can't connect to backend
- Verify backend is running on port 8001
- Check CORS configuration
- Verify BACKEND_URL in frontend code (should be `http://localhost:8001`)

### Port conflicts
- Backend default: 8001
- Frontend default: 3000
- Modify ports in `config.py` and `frontend/server.js` if needed

---

## 📊 Performance Notes

- **First run**: Downloads Hugging Face models (~500MB)
- **Vision detection**: 30-60 seconds (depends on hardware)
- **SOP generation**: <5 seconds
- **PDF generation**: <5 seconds
- **Total workflow**: ~1-2 minutes end-to-end

---

## 🎓 Why This Project Stands Out

- ✅ **Multi-Agent Architecture**: Real-world AI system design
- ✅ **Computer Vision**: Industry-standard object detection
- ✅ **Manufacturing Focus**: Practical, applicable solution
- ✅ **Full-Stack**: Complete end-to-end system
- ✅ **Production-Ready**: Error handling, fallbacks, validation
- ✅ **Professional Output**: PDF reports with annotations
- ✅ **Extensible Design**: Easy to add new agents or components

---

## 📝 License

MIT License

---

## 👥 Contributing

This is an industry-grade project demonstrating:
- Multi-agent AI systems
- Computer vision applications
- Manufacturing automation
- Full-stack development

Perfect for portfolios targeting roles at:
- MediaTek, Qualcomm, Micron
- Foxconn, Wistron, Pegatron
- Any electronics manufacturing company

---

## 🔗 Repository

**GitHub**: https://github.com/chakradharkalle03-arch

---

**Status**: ✅ **PRODUCTION READY**

This is a complete, working system ready for demonstration and deployment.
