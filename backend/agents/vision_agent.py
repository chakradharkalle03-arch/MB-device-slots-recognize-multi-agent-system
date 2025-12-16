"""
Vision Agent - Detects and localizes motherboard components using Hugging Face models
"""
import os
import torch
import warnings
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import numpy as np
from typing import List, Dict, Any

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")
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
            # Try loading without token first (if API key not needed)
            try:
                self.processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
                self.model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            except:
                # Fallback: try with token if provided
                if self.api_key and self.api_key != "your_huggingface_token_here":
                    self.processor = AutoImageProcessor.from_pretrained(
                        "facebook/detr-resnet-50",
                        token=self.api_key
                    )
                    self.model = AutoModelForObjectDetection.from_pretrained(
                        "facebook/detr-resnet-50",
                        token=self.api_key
                    )
                else:
                    raise Exception("DETR model requires API key or public access")
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ DETR model loaded successfully on {self.device}")
        except Exception as e:
            print(f"⚠️ DETR model not available, using computer vision only: {e}")
            # Fallback to computer vision only
            self.model = None
            self.processor = None
    
    def detect_components(self, image_path: str) -> Dict[str, Any]:
        """
        Detect motherboard components using vision model and image analysis
        Combines DETR model detection with computer vision techniques
        
        Returns:
            Dictionary with detected components and their bounding boxes
        """
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        width, height = image.size
        
        # Try to use DETR model if available
        detected_components = []
        if self.model is not None and self.processor is not None:
            try:
                detected_components = self._detect_with_detr(image, image_np)
            except Exception as e:
                print(f"DETR detection error: {e}")
        
        # Use computer vision to detect actual connectors/slots in image
        cv_detected = self._detect_with_computer_vision(image_path, image_np, width, height)
        
        # Combine detections and map to known components
        all_detections = detected_components + cv_detected
        
        # Map detections to known motherboard components
        mapped_components = self._map_detections_to_known_components(all_detections, width, height)
        
        # Ensure we have all expected connectors (fill missing ones with rule-based)
        final_components = self._ensure_all_connectors(mapped_components, width, height)
        
        return {
            "components": final_components,
            "image_size": image.size,
            "status": "success",
            "detection_method": "hybrid" if detected_components else "cv_based"
        }
    
    def _detect_with_detr(self, image: Image.Image, image_np: np.ndarray) -> List[Dict]:
        """Use DETR model to detect objects in image"""
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs, threshold=0.3, target_sizes=target_sizes
            )[0]
            
            detections = []
            for score, label, box in zip(
                results["scores"].cpu().numpy(),
                results["labels"].cpu().numpy(),
                results["boxes"].cpu().numpy()
            ):
                if score > 0.3:
                    x1, y1, x2, y2 = box
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(score),
                        "label": int(label),
                        "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                        "detection_source": "detr_real"  # Mark as real DETR detection
                    })
            return detections
        except Exception as e:
            print(f"DETR detection failed: {e}")
            return []
    
    def _detect_with_computer_vision(self, image_path: str, image_np: np.ndarray, width: int, height: int) -> List[Dict]:
        """Use OpenCV to detect actual connectors and slots in the image"""
        import cv2
        
        # Convert PIL image to OpenCV format
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        detections = []
        
        # Apply adaptive thresholding to better detect connectors
        # Connectors often have different brightness/contrast
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Detect rectangular connectors (slots, headers)
        # Use edge detection and contour analysis
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Also use adaptive threshold contours
        adaptive_contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours = list(contours) + list(adaptive_contours)
        
        seen_boxes = set()  # Avoid duplicates
        
        for contour in all_contours:
            area = cv2.contourArea(contour)
            # Filter by size - connectors are typically medium-sized rectangles
            # More selective range for better accuracy
            if 300 < area < 50000:  # More selective range
                x, y, w, h = cv2.boundingRect(contour)
                
                # Skip if too small or too large relative to image
                if w < 15 or h < 15 or w > width * 0.7 or h > height * 0.7:
                    continue
                
                # Create a unique key for this box to avoid duplicates (more precise)
                box_key = (x // 5, y // 5, w // 5, h // 5)
                if box_key in seen_boxes:
                    continue
                seen_boxes.add(box_key)
                
                aspect_ratio = w / h if h > 0 else 0
                
                # Connectors are typically rectangular (more selective range)
                if 0.3 <= aspect_ratio <= 6.0:  # More selective
                    # Check if it looks like a connector (rectangular shape)
                    rect_area = w * h
                    extent = area / rect_area if rect_area > 0 else 0
                    
                    # Connectors typically have 40-90% fill
                    if extent > 0.3:  # Lowered threshold for better detection
                        # Calculate confidence based on how well it matches connector characteristics
                        # Higher confidence for medium-sized, well-proportioned rectangles
                        size_score = min(1.0, area / 5000)  # Normalize by typical connector size
                        aspect_score = 1.0 - abs(aspect_ratio - 2.0) / 4.0  # Prefer aspect ratio around 2.0
                        aspect_score = max(0.3, aspect_score)  # Minimum score
                        confidence = 0.60 + (size_score * 0.2) + (aspect_score * 0.2)
                        confidence = min(0.95, confidence)  # Cap at 95%
                        
                        detections.append({
                            "bbox": [float(x), float(y), float(x + w), float(y + h)],
                            "confidence": confidence,
                            "type": "connector",
                            "center": [float(x + w/2), float(y + h/2)],
                            "detection_source": "cv_real",  # Mark as real CV detection
                            "area": area,
                            "aspect_ratio": aspect_ratio
                        })
        
        # Detect circular connectors (audio jacks, power connectors)
        # Use multiple methods for better detection
        circles1 = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=5, maxRadius=50
        )
        
        # Also try with different parameters
        circles2 = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=2, minDist=50,
            param1=100, param2=50, minRadius=3, maxRadius=80
        )
        
        all_circles = []
        if circles1 is not None:
            all_circles.extend(circles1[0])
        if circles2 is not None:
            all_circles.extend(circles2[0])
        
        if all_circles:
            circles = np.round(np.array(all_circles)).astype("int")
            for (x, y, r) in circles:
                # Avoid duplicates
                circle_key = (x // 10, y // 10, r // 5)
                if circle_key not in seen_boxes:
                    seen_boxes.add(circle_key)
                    detections.append({
                        "bbox": [float(x - r), float(y - r), float(x + r), float(y + r)],
                        "confidence": 0.70,
                        "type": "circular_connector",
                        "center": [float(x), float(y)],
                        "radius": float(r),
                        "detection_source": "cv_real"  # Mark as real CV detection
                    })
        
        return detections
    
    def _map_detections_to_known_components(self, detections: List[Dict], width: int, height: int) -> List[Dict]:
        """Map detected objects to known motherboard component types"""
        if not detections:
            return []
        
        # Known component locations and characteristics
        component_templates = {
            "RAM Slot": {
                "aspect_ratio_range": (3.0, 8.0),  # Long horizontal slot
                "area_range": (5000, 30000),
                "location_hint": "upper_right",
                "bbox_ratio": [(0.5, 1.0), (0.0, 0.4)]  # x: 50-100%, y: 0-40%
            },
            "Fan Connector": {
                "aspect_ratio_range": (0.8, 1.5),  # Square-ish
                "area_range": (200, 2000),
                "location_hint": "center_upper",
                "bbox_ratio": [(0.3, 0.7), (0.1, 0.4)]
            },
            "Display Connector": {
                "aspect_ratio_range": (2.0, 5.0),  # Long horizontal
                "area_range": (1000, 8000),
                "location_hint": "left_edge",
                "bbox_ratio": [(0.0, 0.3), (0.3, 0.7)]
            },
            "USB Connector": {
                "aspect_ratio_range": (0.3, 0.8),  # Vertical rectangle
                "area_range": (300, 2000),
                "location_hint": "right_edge",
                "bbox_ratio": [(0.7, 1.0), (0.3, 0.7)]
            },
            "SATA Connector": {
                "aspect_ratio_range": (1.5, 3.0),  # Medium horizontal
                "area_range": (400, 3000),
                "location_hint": "left_edge",
                "bbox_ratio": [(0.0, 0.3), (0.5, 0.8)]
            },
            "Keyboard Connector": {
                "aspect_ratio_range": (2.0, 4.0),  # Long horizontal
                "area_range": (800, 5000),
                "location_hint": "bottom_left",
                "bbox_ratio": [(0.0, 0.4), (0.7, 1.0)]
            },
            "Power Connector": {
                "aspect_ratio_range": (0.5, 1.5),  # Square to vertical
                "area_range": (200, 1500),
                "location_hint": "bottom_right",
                "bbox_ratio": [(0.8, 1.0), (0.7, 1.0)]
            },
            "Battery Connector": {
                "aspect_ratio_range": (0.8, 2.0),  # Square to horizontal
                "area_range": (300, 2000),
                "location_hint": "bottom_right",
                "bbox_ratio": [(0.6, 0.9), (0.6, 0.9)]
            },
            "LED Connector": {
                "aspect_ratio_range": (0.5, 1.5),  # Small square
                "area_range": (100, 800),
                "location_hint": "top_left",
                "bbox_ratio": [(0.0, 0.2), (0.0, 0.2)]
            },
            "Audio Connector": {
                "aspect_ratio_range": (0.8, 1.2),  # Circular/square
                "area_range": (50, 500),
                "location_hint": "top_right",
                "bbox_ratio": [(0.85, 1.0), (0.0, 0.3)]
            }
        }
        
        mapped = []
        used_detections = set()
        
        # Sort detections by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x.get("confidence", 0), reverse=True)
        
        for comp_name, template in component_templates.items():
            best_match = None
            best_score = 0
            
            for idx, det in enumerate(sorted_detections):
                if idx in used_detections:
                    continue
                
                bbox = det.get("bbox", [])
                if len(bbox) < 4:
                    continue
                
                x1, y1, x2, y2 = bbox
                det_width = x2 - x1
                det_height = y2 - y1
                area = det_width * det_height
                aspect_ratio = det_width / det_height if det_height > 0 else 0
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Check if detection matches template
                score = 0
                
                # Check aspect ratio
                if template["aspect_ratio_range"][0] <= aspect_ratio <= template["aspect_ratio_range"][1]:
                    score += 0.3
                
                # Check area
                if template["area_range"][0] <= area <= template["area_range"][1]:
                    score += 0.3
                
                # Check location
                width_norm = center_x / width if width > 0 else 0
                height_norm = center_y / height if height > 0 else 0
                bbox_x_range, bbox_y_range = template["bbox_ratio"]
                
                if bbox_x_range[0] <= width_norm <= bbox_x_range[1] and \
                   bbox_y_range[0] <= height_norm <= bbox_y_range[1]:
                    score += 0.4
                
                # Add confidence boost
                score += det.get("confidence", 0) * 0.1
                
                if score > best_score:
                    best_score = score
                    best_match = (idx, det)
            
            if best_match and best_score > 0.4:  # Minimum threshold
                idx, det = best_match
                used_detections.add(idx)
                
                # Get component type
                comp_type = self._get_component_subtype(comp_name)
                if comp_type == "Unknown":
                    comp_type = template.get("type", "Connector")
                
                mapped.append({
                    "name": comp_name,
                    "type": comp_type,
                    "bbox": det["bbox"],
                    "confidence": max(det.get("confidence", 0.7), best_score),
                    "center": det.get("center", [(det["bbox"][0] + det["bbox"][2])/2, (det["bbox"][1] + det["bbox"][3])/2]),
                    "location_description": self._get_location_description(comp_name, det["bbox"], width, height),
                    "detection_source": "real"  # Mark as real detection from image analysis
                })
        
        return mapped
    
    def _get_location_description(self, comp_name: str, bbox: List[float], width: int, height: int) -> str:
        """Generate location description based on bbox position"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Determine position
        x_pos = "left" if center_x < width * 0.33 else "right" if center_x > width * 0.67 else "center"
        y_pos = "top" if center_y < height * 0.33 else "bottom" if center_y > height * 0.67 else "middle"
        
        # Create description
        if x_pos == "left" and y_pos == "top":
            return "Top-left corner"
        elif x_pos == "right" and y_pos == "top":
            return "Top-right corner"
        elif x_pos == "left" and y_pos == "bottom":
            return "Bottom-left corner"
        elif x_pos == "right" and y_pos == "bottom":
            return "Bottom-right corner"
        elif x_pos == "left":
            return "Left edge, middle section"
        elif x_pos == "right":
            return "Right edge, middle section"
        elif y_pos == "top":
            return "Top edge, center section"
        elif y_pos == "bottom":
            return "Bottom edge, center section"
        else:
            return "Center area"
    
    def _ensure_all_connectors(self, mapped_components: List[Dict], width: int, height: int) -> List[Dict]:
        """
        Return ONLY real detections - NO rule-based fallbacks
        This ensures annotation shows only actual components detected from image
        """
        # Filter to only real detections (NO rule-based fallbacks)
        real_only_components = [
            comp for comp in mapped_components
            if comp.get("detection_source") in ["real", "cv_real", "detr_real"]
            and comp.get("detection_source") != "rule_based"
        ]
        
        # Log what was detected vs what would be rule-based (for debugging)
        expected_components = [
            "LED Connector", "USB Connector", "RAM Slot", "Fan Connector",
            "Display Connector", "Keyboard Connector", "Audio Connector",
            "SATA Connector", "Power Connector", "Battery Connector"
        ]
        found_names = {comp["name"] for comp in real_only_components}
        missing = [name for name in expected_components if name not in found_names]
        
        if missing:
            print(f"📊 Real detections: {len(real_only_components)} components")
            print(f"📊 Missing (not detected in image): {len(missing)} components")
            print(f"   Missing: {', '.join(missing)}")
            print(f"⚠️ These components will NOT appear (not detected in actual image)")
        else:
            print(f"✅ All expected components detected from image: {len(real_only_components)}")
        
        # Sort by installation order (Battery last) - but only real detections
        sorted_components = sorted(
            real_only_components,
            key=lambda x: 1 if "Battery" in x["name"] else 0
        )
        
        return sorted_components
    
    def _map_detections_to_components(self, results, image_size) -> List[Dict]:
        """Map generic detections to motherboard-specific components"""
        # This method is kept for backward compatibility
        width, height = image_size if isinstance(image_size, tuple) else (800, 600)
        return self._rule_based_detection((width, height))
    
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
        
        # Typical laptop motherboard component locations - ALL connectors detected
        # Ordered by installation sequence: Low-risk components first, Battery last (highest risk)
        components = [
            {
                "name": "LED Connector",
                "type": "2-pin LED Header",
                "bbox": [width * 0.05, height * 0.1, width * 0.15, height * 0.2],
                "confidence": 0.87,
                "center": [width * 0.1, height * 0.15],
                "location_description": "Top-left corner, near power indicator area"
            },
            {
                "name": "USB Connector",
                "type": "USB Type-A Port",
                "bbox": [width * 0.8, height * 0.4, width * 0.95, height * 0.6],
                "confidence": 0.89,
                "center": [width * 0.875, height * 0.5],
                "location_description": "Right edge, middle section, typically stacked vertically"
            },
            {
                "name": "RAM Slot",
                "type": "SO-DIMM Slot",
                "bbox": [width * 0.6, height * 0.1, width * 0.9, height * 0.3],
                "confidence": 0.90,
                "center": [width * 0.75, height * 0.2],
                "location_description": "Upper-right section, horizontal orientation"
            },
            {
                "name": "Fan Connector",
                "type": "4-pin PWM Connector",
                "bbox": [width * 0.4, height * 0.2, width * 0.55, height * 0.35],
                "confidence": 0.88,
                "center": [width * 0.475, height * 0.275],
                "location_description": "Center-upper area, near CPU/GPU heat sink"
            },
            {
                "name": "Display Connector",
                "type": "LVDS/eDP Connector",
                "bbox": [width * 0.1, height * 0.4, width * 0.25, height * 0.55],
                "confidence": 0.86,
                "center": [width * 0.175, height * 0.475],
                "location_description": "Left edge, middle section, for LCD panel connection"
            },
            {
                "name": "Keyboard Connector",
                "type": "ZIF Connector",
                "bbox": [width * 0.1, height * 0.8, width * 0.3, height * 0.95],
                "confidence": 0.85,
                "center": [width * 0.2, height * 0.875],
                "location_description": "Bottom-left edge, near keyboard area"
            },
            {
                "name": "Audio Connector",
                "type": "Audio Jack",
                "bbox": [width * 0.9, height * 0.1, width * 0.98, height * 0.25],
                "confidence": 0.83,
                "center": [width * 0.94, height * 0.175],
                "location_description": "Top-right corner, for headphone/microphone"
            },
            {
                "name": "SATA Connector",
                "type": "SATA Port",
                "bbox": [width * 0.05, height * 0.6, width * 0.2, height * 0.75],
                "confidence": 0.81,
                "center": [width * 0.125, height * 0.675],
                "location_description": "Left edge, lower-middle section, for storage drives"
            },
            {
                "name": "Power Connector",
                "type": "DC Jack",
                "bbox": [width * 0.85, height * 0.75, width * 0.98, height * 0.9],
                "confidence": 0.84,
                "center": [width * 0.915, height * 0.825],
                "location_description": "Right edge, bottom section, for AC adapter connection"
            },
            {
                "name": "Battery Connector",
                "type": "JST Connector",
                "bbox": [width * 0.7, height * 0.7, width * 0.85, height * 0.85],
                "confidence": 0.82,
                "center": [width * 0.775, height * 0.775],
                "location_description": "Bottom-right corner, power management area"
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
    
    def annotate_image(self, image_path: str, components: List[Dict], output_path: str) -> str:
        """
        Annotate image with bounding boxes - Only shows REAL detections from image analysis
        ALWAYS returns a valid image path (even if no components detected)
        """
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not load image from {image_path}")
            # Return output_path anyway so frontend can try to load it
            return output_path
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        # Color scheme: Green for real detections, Yellow for rule-based (if shown)
        real_detection_color = (0, 255, 0)  # Green
        rule_based_color = (255, 255, 0)    # Yellow (dashed)
        
        # Annotate ALL components that have valid bboxes (except explicitly rule-based)
        # Include components from DETR, Computer Vision, or any detection with bbox
        components_to_annotate = []
        rule_based_count = 0
        
        for comp in components:
            # Skip if explicitly marked as rule-based
            if comp.get("detection_source") == "rule_based":
                rule_based_count += 1
                continue
            
            # Include if it has a valid bbox
            if "bbox" in comp and len(comp.get("bbox", [])) == 4:
                # Check if bbox has valid coordinates
                bbox = comp["bbox"]
                if all(isinstance(coord, (int, float)) for coord in bbox):
                    components_to_annotate.append(comp)
        
        print(f"📊 Components to annotate: {len(components_to_annotate)}")
        print(f"📊 Rule-based components (excluded): {rule_based_count}")
        
        # If no components to annotate, show original image with message
        if not components_to_annotate:
            print(f"⚠️ No components with valid bounding boxes found")
            print(f"⚠️ Showing original image without annotations")
        
        # Draw bounding boxes and labels for all components
        for comp in components_to_annotate:
            try:
                bbox = comp["bbox"]
                if len(bbox) != 4:
                    print(f"⚠️ Component {comp.get('name', 'Unknown')} has invalid bbox: {bbox}")
                    continue
                    
                x1, y1, x2, y2 = [int(float(coord)) for coord in bbox]
                
                # Validate bbox coordinates are within image bounds
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(x1 + 1, min(x2, width))
                y2 = max(y1 + 1, min(y2, height))
                
                # Skip if bbox is invalid
                if x2 <= x1 or y2 <= y1:
                    print(f"⚠️ Component {comp.get('name', 'Unknown')} has invalid bbox dimensions")
                    continue
                
                # Use bright green for all real detections (highly visible)
                color = (0, 255, 0)  # Bright green in RGB
                line_thickness = 3  # Thicker lines for better visibility
                
                # Draw bounding box (thick, bright green)
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, line_thickness)
                
                # Add label with component name and confidence
                comp_name = comp.get("name", "Unknown")
                confidence = comp.get("confidence", 0)
                label = f"{comp_name}"
                if confidence > 0:
                    label += f" ({confidence:.0%})"
                
                # Use larger font for better visibility
                font_scale = 0.7
                font_thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )
                
                # Position label above bounding box (or inside if too close to top)
                label_y = max(y1 - 10, text_height + 10)
                label_x = x1
                
                # Ensure label doesn't go off screen
                if label_x + text_width > width:
                    label_x = width - text_width - 5
                
                # Draw semi-transparent background for label (better visibility)
                overlay = image_rgb.copy()
                cv2.rectangle(
                    overlay,
                    (label_x - 5, label_y - text_height - 5),
                    (label_x + text_width + 5, label_y + 5),
                    (0, 0, 0),  # Black background
                    -1
                )
                cv2.addWeighted(overlay, 0.7, image_rgb, 0.3, 0, image_rgb)
                
                # Draw text in bright white (high contrast)
                cv2.putText(
                    image_rgb, label, (label_x, label_y),
                    font, font_scale, (255, 255, 255), font_thickness
                )
                
                print(f"✅ Annotated: {comp_name} at ({x1},{y1})-({x2},{y2})")
                
            except Exception as e:
                print(f"❌ Error annotating component {comp.get('name', 'Unknown')}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save annotated image (always save original image, with real detections if found)
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # If no components detected, show original image with informative message
            if len(components_to_annotate) == 0:
                # Add informative text overlay
                text = "No components detected from image analysis"
                text2 = "Original image shown (no annotations)"
                
                # Get text size for centering
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                (text_width1, text_height1), _ = cv2.getTextSize(text, font, font_scale, thickness)
                (text_width2, text_height2), _ = cv2.getTextSize(text2, font, font_scale, thickness)
                
                # Center text
                x1 = (width - text_width1) // 2
                x2 = (width - text_width2) // 2
                y1 = height // 2 - 20
                y2 = height // 2 + 20
                
                # Add semi-transparent background for text
                overlay = image_rgb.copy()
                cv2.rectangle(overlay, (x1 - 10, y1 - text_height1 - 10), 
                            (x2 + text_width2 + 10, y2 + text_height2 + 10), 
                            (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, image_rgb, 0.4, 0, image_rgb)
                
                # Draw text
                cv2.putText(image_rgb, text, (x1, y1), font, font_scale, (255, 255, 255), thickness)
                cv2.putText(image_rgb, text2, (x2, y2), font, font_scale, (200, 200, 200), thickness)
                
                print(f"⚠️ No components detected - showing original image")
            else:
                print(f"✅ Annotated {len(components_to_annotate)} components with bounding boxes and labels")
            
            # Convert RGB to BGR for OpenCV saving
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Save as PNG for better quality
            success = cv2.imwrite(output_path, image_bgr)
            
            if success:
                print(f"✅ Annotated image saved successfully to {output_path}")
                
                # Verify file was created
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"✅ Image file verified: {file_size} bytes")
                    
                    # Additional verification: try to read it back
                    test_read = cv2.imread(output_path)
                    if test_read is not None:
                        h, w = test_read.shape[:2]
                        print(f"✅ Image verified: {w}x{h} pixels, {len(components_to_annotate)} components annotated")
                    else:
                        print(f"⚠️ Warning: Saved image could not be read back")
                else:
                    print(f"❌ ERROR: Image file was not created at {output_path}")
            else:
                print(f"❌ ERROR: cv2.imwrite() returned False - image not saved!")
                # Try alternative save method
                try:
                    from PIL import Image as PILImage
                    pil_image = PILImage.fromarray(image_rgb)
                    pil_image.save(output_path, 'PNG')
                    print(f"✅ Saved using PIL as fallback")
                except Exception as e2:
                    print(f"❌ PIL fallback also failed: {e2}")
                
        except Exception as e:
            print(f"❌ Error saving annotated image: {e}")
            import traceback
            traceback.print_exc()
            # Ensure we still return the path even if save failed
            # Frontend can try to load it
        
        return output_path

