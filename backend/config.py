import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "your_huggingface_api_key_here")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")
BACKEND_PORT = 8001

# Model configurations
VISION_MODEL = "facebook/detr-resnet-50"
LANGUAGE_MODEL = "microsoft/DialoGPT-medium"

# File paths
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

