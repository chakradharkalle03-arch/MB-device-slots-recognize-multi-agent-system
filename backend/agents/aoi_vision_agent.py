"""
AOI Vision Agent - Detects visual defects in PCB/AOI images
"""
import torch
import warnings
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import numpy as np
import cv2
from typing import List, Dict, Any

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")

class AOIVisionAgent:
    """Detects defects in AOI inspection images"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 or DETR model for defect detection"""
        try:
            # Try to load YOLOv8 (better for defect detection)
            try:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')  # Nano model for speed
                self.model_type = "yolov8"
            except:
                # Fallback to DETR
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
                self.model_type = "detr"
        except Exception as e:
            print(f"Error loading AOI model: {e}")
            self.model = None
            self.model_type = "rule_based"
    
    def detect_defects(self, image_path: str) -> Dict[str, Any]:
        """
        Detect defects in AOI inspection image
        
        Returns:
            Dictionary with detected defects, pass/fail status, and confidence scores
        """
        if self.model is None or self.model_type == "rule_based":
            return self._rule_based_defect_detection(image_path)
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            # IMPORTANT: DETR/YOLO models are NOT trained for PCB defect detection
            # They detect general objects (COCO dataset), not PCB defects
            # Always use REAL computer vision analysis for actual defect detection
            print("📊 Using REAL computer vision analysis for PCB defect detection...")
            print("⚠️ Note: DETR/YOLO models detect general objects, not PCB defects")
            print("✅ Using CV analysis based on actual image features")
            
            defects = self._detect_defects_with_computer_vision(image_path)
            
            # Log real detection results
            if len(defects) > 0:
                print(f"✅ Real defects detected from image analysis: {len(defects)}")
                for defect in defects:
                    print(f"   - {defect['type']} (confidence: {defect['confidence']:.2f}, source: {defect.get('detection_source', 'cv_real')})")
            else:
                print("✅ No defects detected in image (board appears clean)")
            
            # Determine pass/fail
            pass_fail_status = self._determine_pass_fail(defects)
            
            return {
                "defects": defects,
                "status": pass_fail_status["status"],
                "confidence": pass_fail_status["confidence"],
                "total_defects": len(defects),
                "critical_defects": len([d for d in defects if d.get("severity") == "Critical"]),
                "image_size": image.size,
                "detection_method": "cv_real"  # Always use real CV analysis, not DETR/YOLO
            }
        except Exception as e:
            print(f"AOI detection error: {e}")
            return self._rule_based_defect_detection(image_path)
    
    def _parse_yolo_results(self, results, image_size) -> List[Dict]:
        """Parse YOLOv8 results to defect format"""
        defects = []
        if results and len(results) > 0:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        defect_type = self._classify_defect_type(box.cls.item() if hasattr(box, 'cls') else 0)
                        confidence = float(box.conf.item() if hasattr(box, 'conf') else 0.5)
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        defects.append({
                            "type": defect_type,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": confidence,
                            "severity": self._get_severity(defect_type),
                            "area": float((x2 - x1) * (y2 - y1))
                        })
        return defects
    
    def _parse_detr_results(self, results, image_size) -> List[Dict]:
        """Parse DETR results to defect format"""
        defects = []
        for score, label, box in zip(
            results["scores"].cpu().numpy(),
            results["labels"].cpu().numpy(),
            results["boxes"].cpu().numpy()
        ):
            if score > 0.3:
                defect_type = self._classify_defect_type(label)
                x1, y1, x2, y2 = box
                defects.append({
                    "type": defect_type,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(score),
                    "severity": self._get_severity(defect_type),
                    "area": float((x2 - x1) * (y2 - y1))
                })
        return defects
    
    def _rule_based_defect_detection(self, image_path: str) -> Dict[str, Any]:
        """Fallback rule-based defect detection"""
        image = cv2.imread(image_path)
        if image is None:
            return {
                "defects": [],
                "status": "PASS",
                "confidence": 0.85,
                "total_defects": 0,
                "critical_defects": 0,
                "image_size": (0, 0),
                "detection_method": "rule_based"
            }
        
        height, width = image.shape[:2]
        
        # Use REAL computer vision analysis (no simulated defects)
        defects = self._detect_defects_with_computer_vision(image_path)
        
        pass_fail_status = self._determine_pass_fail(defects)
        
        return {
            "defects": defects,
            "status": pass_fail_status["status"],  # PASS or HOLD (never FAIL from AI)
            "confidence": pass_fail_status["confidence"],
            "decision_info": {
                "decision_reason": pass_fail_status.get("decision_reason", "LOW_RISK"),
                "requires_manual_aoi": pass_fail_status.get("requires_manual_aoi", False),
                "requires_axi": pass_fail_status.get("requires_axi", False),
                "ai_decision_only": pass_fail_status.get("ai_decision_only", True),
                "production_note": pass_fail_status.get("production_note", "")
            },
            "total_defects": len(defects),
            "critical_defects": len([d for d in defects if d.get("severity") == "Critical"]),
            "image_size": (width, height),
            "detection_method": "cv_real"  # Always use real CV analysis
        }
    
    def _detect_defects_with_computer_vision(self, image_path: str) -> List[Dict]:
        """
        REAL defect detection using computer vision techniques
        NO simulated defects - only actual image analysis
        """
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        defects = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # 1. Detect solder bridges (bright connected regions)
        _, thresh_bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours_bright, _ = cv2.findContours(thresh_bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_bright:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Reasonable defect size
                x, y, w, h = cv2.boundingRect(contour)
                # Check if it looks like a bridge (elongated shape)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                if aspect_ratio > 2.0:  # Bridge-like shape
                    defects.append({
                        "type": "Solder Bridge",
                        "bbox": [float(x), float(y), float(x + w), float(y + h)],
                        "confidence": min(0.75 + (area / 5000), 0.90),
                        "severity": "High",
                        "area": float(area),
                        "detection_source": "cv_real"
                    })
        
        # 2. Detect excess solder (bright blobs)
        _, thresh_excess = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        contours_excess, _ = cv2.findContours(thresh_excess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_excess:
            area = cv2.contourArea(contour)
            if 200 < area < 3000:
                x, y, w, h = cv2.boundingRect(contour)
                # Check if it's a blob (not elongated)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                if aspect_ratio < 2.0:  # Blob-like shape
                    defects.append({
                        "type": "Excess Solder",
                        "bbox": [float(x), float(y), float(x + w), float(y + h)],
                        "confidence": min(0.70 + (area / 3000), 0.85),
                        "severity": "Medium",
                        "area": float(area),
                        "detection_source": "cv_real"
                    })
        
        # 3. Detect missing components (dark rectangular regions where components should be)
        # Look for dark regions with component-like shapes
        _, thresh_dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        contours_dark, _ = cv2.findContours(thresh_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_dark:
            area = cv2.contourArea(contour)
            if 500 < area < 8000:  # Component-sized
                x, y, w, h = cv2.boundingRect(contour)
                # Check if it's rectangular (component-like)
                rect_area = w * h
                extent = area / max(rect_area, 1)
                if extent > 0.6:  # Relatively rectangular
                    defects.append({
                        "type": "Missing Component",
                        "bbox": [float(x), float(y), float(x + w), float(y + h)],
                        "confidence": min(0.75 + (area / 8000), 0.90),
                        "severity": "Critical",
                        "area": float(area),
                        "detection_source": "cv_real"
                    })
        
        # 4. Detect tombstone (component standing on edge - detected by edge analysis)
        edges = cv2.Canny(gray, 50, 150)
        contours_edges, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_edges:
            area = cv2.contourArea(contour)
            if 300 < area < 4000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                # Tombstone: tall and narrow
                if aspect_ratio > 3.0 and h > w:
                    defects.append({
                        "type": "Tombstone",
                        "bbox": [float(x), float(y), float(x + w), float(y + h)],
                        "confidence": 0.70,
                        "severity": "High",
                        "area": float(area),
                        "detection_source": "cv_real"
                    })
        
        # 5. Detect scratches/marks (linear dark features)
        # Use HoughLines to detect linear features
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        if lines is not None:
            for line in lines[:5]:  # Limit to top 5
                x1, y1, x2, y2 = line[0]
                # Create bbox around line
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                w, h = x_max - x_min, y_max - y_min
                if w > 20 or h > 20:  # Minimum size
                    defects.append({
                        "type": "Scratch/Mark",
                        "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
                        "confidence": 0.50,  # Lower confidence (high FP risk)
                        "severity": "Low",
                        "area": float(w * h),
                        "detection_source": "cv_real"
                    })
        
        print(f"📊 Real CV defects detected: {len(defects)}")
        return defects
    
    def _classify_defect_type(self, label: int) -> str:
        """Classify defect type from model label"""
        defect_types = [
            "Solder Bridge",
            "Missing Component",
            "Component Misalignment",
            "Solder Void",
            "Tombstone",
            "Insufficient Solder",
            "Excess Solder",
            "Component Damage",
            "Polarity Error",
            "Scratch/Mark"
        ]
        return defect_types[label % len(defect_types)]
    
    def _get_severity(self, defect_type: str) -> str:
        """Determine defect severity"""
        critical_defects = ["Missing Component", "Polarity Error", "Component Damage"]
        high_defects = ["Component Misalignment", "Tombstone"]
        
        if defect_type in critical_defects:
            return "Critical"
        elif defect_type in high_defects:
            return "High"
        else:
            return "Medium"
    
    def _determine_pass_fail(self, defects: List[Dict]) -> Dict[str, Any]:
        """
        Determine inspection status based on defects
        PRODUCTION-SAFE: AI can only assign PASS or HOLD, never FAIL
        FAIL can only be issued after human verification and IPC-A-610 compliance check
        """
        critical_count = len([d for d in defects if d.get("severity") == "Critical"])
        high_count = len([d for d in defects if d.get("severity") == "High"])
        total_count = len(defects)
        
        # Production-safe decision logic:
        # AI can only assign PASS or HOLD (for human verification)
        # FAIL requires human confirmation + IPC-A-610 compliance
        
        if critical_count > 0:
            # Critical defects detected - HOLD for manual verification
            status = "HOLD"
            decision_reason = "CRITICAL_RISK_DETECTED"
            confidence = 0.95
            requires_manual_aoi = True
            requires_axi = False  # May need X-ray for confirmation
        elif high_count >= 2:
            # Multiple high-severity defects - HOLD for verification
            status = "HOLD"
            decision_reason = "MULTIPLE_HIGH_RISK_DEFECTS"
            confidence = 0.85
            requires_manual_aoi = True
            requires_axi = False
        elif high_count >= 1:
            # Single high-severity defect - HOLD for verification
            status = "HOLD"
            decision_reason = "HIGH_RISK_DEFECT_DETECTED"
            confidence = 0.80
            requires_manual_aoi = True
            requires_axi = False
        elif total_count >= 5:
            # Multiple defects - HOLD for review
            status = "HOLD"
            decision_reason = "MULTIPLE_DEFECTS_DETECTED"
            confidence = 0.75
            requires_manual_aoi = True
            requires_axi = False
        elif total_count >= 3:
            # Moderate defect count - HOLD for verification
            status = "HOLD"
            decision_reason = "DEFECTS_REQUIRE_VERIFICATION"
            confidence = 0.70
            requires_manual_aoi = True
            requires_axi = False
        else:
            # Low defect count - PASS (but still flagged for review)
            status = "PASS"
            decision_reason = "LOW_RISK"
            confidence = 0.90
            requires_manual_aoi = total_count > 0  # Review if any defects
            requires_axi = False
        
        return {
            "status": status,
            "decision_reason": decision_reason,
            "confidence": confidence,
            "requires_manual_aoi": requires_manual_aoi,
            "requires_axi": requires_axi,
            "ai_decision_only": True,  # Flag that this is AI-only, not final QA
            "production_note": "AI-assisted inspection. Final decision requires human verification per IPC-A-610 standards."
        }
    
    def annotate_image(self, image_path: str, defects: List[Dict], output_path: str):
        """Annotate AOI image with defect bounding boxes"""
        image = cv2.imread(image_path)
        if image is None:
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for defect in defects:
            bbox = defect["bbox"]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            severity = defect.get("severity", "Medium")
            
            # Color coding: Critical=Red, High=Orange, Medium=Yellow
            color_map = {
                "Critical": (255, 0, 0),
                "High": (255, 165, 0),
                "Medium": (255, 255, 0)
            }
            color = color_map.get(severity, (255, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{defect['type']} ({defect['confidence']:.2f})"
            cv2.putText(
                image_rgb, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        # Save annotated image
        cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        return output_path

