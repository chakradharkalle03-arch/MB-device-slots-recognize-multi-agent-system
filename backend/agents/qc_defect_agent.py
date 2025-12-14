"""
QC Defect Classification Agent - Classifies and analyzes AOI defects
"""
from typing import Dict, Any, List

class QCDefectAgent:
    """Classifies defects and determines QC decisions"""
    
    def __init__(self):
        self.defect_database = self._initialize_defect_database()
    
    def _initialize_defect_database(self) -> Dict[str, Dict]:
        """Initialize defect knowledge database"""
        return {
            "Solder Bridge": {
                "description": "Unintended electrical connection between adjacent pads/components",
                "root_cause": ["Excess solder paste", "Stencil misalignment", "Reflow profile issue"],
                "action_required": "Remove bridge with soldering iron or rework",
                "false_positive_risk": "Medium",
                "acceptance_criteria": "No electrical continuity between pads"
            },
            "Missing Component": {
                "description": "Component not present at designated location",
                "root_cause": ["Pick and place error", "Component feed issue", "Nozzle problem"],
                "action_required": "CRITICAL: Install missing component",
                "false_positive_risk": "Low",
                "acceptance_criteria": "Component must be present and correctly oriented"
            },
            "Component Misalignment": {
                "description": "Component placed outside acceptable tolerance",
                "root_cause": ["Placement accuracy", "Board warpage", "Vision system error"],
                "action_required": "Rework if exceeds IPC-A-610 tolerance",
                "false_positive_risk": "High",
                "acceptance_criteria": "Within IPC-A-610 Class 2/3 standards"
            },
            "Solder Void": {
                "description": "Air pocket within solder joint",
                "root_cause": ["Moisture in component", "Insufficient flux", "Reflow profile"],
                "action_required": "Accept if <25% void area, rework if >25%",
                "false_positive_risk": "Medium",
                "acceptance_criteria": "Void area < 25% of joint"
            },
            "Tombstone": {
                "description": "Component standing on end due to uneven solder",
                "root_cause": ["Uneven pad heating", "Component size mismatch", "Reflow issue"],
                "action_required": "CRITICAL: Rework required",
                "false_positive_risk": "Low",
                "acceptance_criteria": "Component must be flat on board"
            },
            "Insufficient Solder": {
                "description": "Solder joint lacks adequate volume",
                "root_cause": ["Insufficient paste", "Stencil aperture issue", "Reflow incomplete"],
                "action_required": "Add solder if joint strength compromised",
                "false_positive_risk": "Medium",
                "acceptance_criteria": "Solder fillet visible on all sides"
            },
            "Excess Solder": {
                "description": "Too much solder causing potential shorts",
                "root_cause": ["Excess paste", "Stencil thickness", "Reflow profile"],
                "action_required": "Remove excess if bridging risk exists",
                "false_positive_risk": "High",
                "acceptance_criteria": "No bridging, acceptable fillet shape"
            },
            "Component Damage": {
                "description": "Physical damage to component body or leads",
                "root_cause": ["Handling damage", "Placement force", "ESD"],
                "action_required": "CRITICAL: Replace damaged component",
                "false_positive_risk": "Low",
                "acceptance_criteria": "Component intact, no cracks or missing parts"
            },
            "Polarity Error": {
                "description": "Component installed with reversed polarity",
                "root_cause": ["Placement error", "Vision misidentification"],
                "action_required": "CRITICAL: Correct polarity immediately",
                "false_positive_risk": "Low",
                "acceptance_criteria": "Polarity matches design specification"
            },
            "Scratch/Mark": {
                "description": "Surface damage or marking on PCB or component",
                "root_cause": ["Handling", "Tool contact", "Transport"],
                "action_required": "Evaluate if affects functionality",
                "false_positive_risk": "Very High",
                "acceptance_criteria": "No functional impact, cosmetic only"
            }
        }
    
    def classify_defects(self, defects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify and enrich each defect with QC information"""
        classified_defects = []
        
        for defect in defects:
            defect_type = defect.get("type", "Unknown")
            defect_info = self.defect_database.get(defect_type, {})
            
            enriched_defect = defect.copy()
            enriched_defect.update({
                "description": defect_info.get("description", "Unknown defect type"),
                "root_cause": defect_info.get("root_cause", []),
                "action_required": defect_info.get("action_required", "Review required"),
                "false_positive_risk": defect_info.get("false_positive_risk", "Medium"),
                "acceptance_criteria": defect_info.get("acceptance_criteria", "Per IPC standards"),
                "needs_rework": defect.get("severity") in ["Critical", "High"]
            })
            
            classified_defects.append(enriched_defect)
        
        return classified_defects
    
    def reduce_false_positives(self, defects: List[Dict[str, Any]], image_analysis: Dict = None) -> List[Dict[str, Any]]:
        """
        Reduce false positives using heuristics and rules
        
        Returns filtered defects with reduced false positive rate
        """
        filtered_defects = []
        
        for defect in defects:
            defect_type = defect.get("type", "")
            false_positive_risk = defect.get("false_positive_risk", "Medium")
            confidence = defect.get("confidence", 0.5)
            
            # Filter criteria
            should_include = True
            
            # High false positive risk + low confidence = likely false positive
            if false_positive_risk == "Very High" and confidence < 0.7:
                should_include = False
            
            # Very small defects might be noise
            area = defect.get("area", 0)
            if area < 20 and confidence < 0.6:
                should_include = False
            
            # Scratch/Mark with low confidence often false positive
            if defect_type == "Scratch/Mark" and confidence < 0.65:
                should_include = False
            
            if should_include:
                filtered_defects.append(defect)
        
        return filtered_defects
    
    def calculate_false_positive_reduction(self, original_count: int, filtered_count: int) -> Dict[str, Any]:
        """Calculate false positive reduction percentage"""
        if original_count == 0:
            return {
                "reduction_percentage": 0,
                "original_count": 0,
                "filtered_count": 0,
                "false_positives_removed": 0
            }
        
        reduction = ((original_count - filtered_count) / original_count) * 100
        
        return {
            "reduction_percentage": round(reduction, 1),
            "original_count": original_count,
            "filtered_count": filtered_count,
            "false_positives_removed": original_count - filtered_count
        }

