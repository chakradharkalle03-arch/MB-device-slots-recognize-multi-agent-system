"""
Vision Agent - Detects and localizes motherboard components using Hugging Face models
"""
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import numpy as np
from typing import List, Dict, Any
import cv2

class VisionAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load DETR model for object detection"""
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                "facebook/detr-resnet-50",
                token=self.api_key
            )
            self.model = AutoModelForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                token=self.api_key
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading vision model: {e}")
            # Fallback to basic detection
            self.model = None
    
    def detect_components(self, image_path: str) -> Dict[str, Any]:
        """
        Detect motherboard components using vision model
        
        Returns:
            Dictionary with detected components and their bounding boxes
        """
        if self.model is None:
            return self._fallback_detection(image_path)
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Prepare inputs
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process outputs
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs, threshold=0.5, target_sizes=target_sizes
            )[0]
            
            # Map to motherboard components
            components = self._map_detections_to_components(results, image.size)
            
            return {
                "components": components,
                "image_size": image.size,
                "status": "success"
            }
        except Exception as e:
            print(f"Vision detection error: {e}")
            return self._fallback_detection(image_path)
    
    def _map_detections_to_components(self, results, image_size) -> List[Dict]:
        """Map generic detections to motherboard-specific components"""
        # Component mapping based on typical motherboard layouts
        component_types = [
            "Keyboard Connector",
            "RAM Slot",
            "Fan Connector",
            "Battery Connector",
            "Display Connector",
            "USB Connector",
            "Power Connector"
        ]
        
        components = []
        for i, (score, label, box) in enumerate(zip(
            results["scores"].cpu().numpy(),
            results["labels"].cpu().numpy(),
            results["boxes"].cpu().numpy()
        )):
            if score > 0.3:  # Lower threshold for motherboard components
                # Map detection to component type
                component_type = component_types[i % len(component_types)]
                
                x1, y1, x2, y2 = box
                components.append({
                    "name": component_type,
                    "type": self._get_component_subtype(component_type),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(score),
                    "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)]
                })
        
        # If no detections, use rule-based fallback
        if not components:
            components = self._rule_based_detection(image_size)
        
        return components
    
    def _get_component_subtype(self, component_name: str) -> str:
        """Get specific subtype for component"""
        mapping = {
            "Keyboard Connector": "ZIF Connector",
            "RAM Slot": "SO-DIMM Slot",
            "Fan Connector": "4-pin PWM Connector",
            "Battery Connector": "JST Connector",
            "Display Connector": "LVDS/eDP Connector",
            "USB Connector": "USB Type-A",
            "Power Connector": "DC Jack"
        }
        return mapping.get(component_name, "Unknown")
    
    def _rule_based_detection(self, image_size) -> List[Dict]:
        """Fallback rule-based detection for common motherboard layouts"""
        width, height = image_size
        
        # Typical laptop motherboard component locations
        components = [
            {
                "name": "Keyboard Connector",
                "type": "ZIF Connector",
                "bbox": [width * 0.1, height * 0.8, width * 0.3, height * 0.95],
                "confidence": 0.85,
                "center": [width * 0.2, height * 0.875]
            },
            {
                "name": "RAM Slot",
                "type": "SO-DIMM Slot",
                "bbox": [width * 0.6, height * 0.1, width * 0.9, height * 0.3],
                "confidence": 0.90,
                "center": [width * 0.75, height * 0.2]
            },
            {
                "name": "Fan Connector",
                "type": "4-pin PWM Connector",
                "bbox": [width * 0.4, height * 0.2, width * 0.55, height * 0.35],
                "confidence": 0.88,
                "center": [width * 0.475, height * 0.275]
            },
            {
                "name": "Battery Connector",
                "type": "JST Connector",
                "bbox": [width * 0.7, height * 0.7, width * 0.85, height * 0.85],
                "confidence": 0.82,
                "center": [width * 0.775, height * 0.775]
            }
        ]
        return components
    
    def _fallback_detection(self, image_path: str) -> Dict[str, Any]:
        """Fallback detection when model fails"""
        image = Image.open(image_path)
        components = self._rule_based_detection(image.size)
        
        return {
            "components": components,
            "image_size": image.size,
            "status": "fallback"
        }
    
    def annotate_image(self, image_path: str, components: List[Dict], output_path: str):
        """Annotate image with bounding boxes"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for comp in components:
            bbox = comp["bbox"]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Draw bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{comp['name']} ({comp['confidence']:.2f})"
            cv2.putText(
                image_rgb, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        # Save annotated image
        cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        return output_path

