"""
SOP Explanation Agent - Provides detailed explanations for each SOP step
"""
from typing import Dict, Any, List

class ExplanationAgent:
    """Explains each step clearly for technicians/operators"""
    
    def __init__(self):
        self.explanation_templates = self._initialize_explanations()
    
    def _initialize_explanations(self) -> Dict[str, Dict[str, str]]:
        """Initialize explanation templates"""
        return {
            "power_off": {
                "why": "Power must be fully disconnected to prevent short circuits, component damage, and electrical shock hazards.",
                "how": "Unplug the AC adapter, remove the battery if accessible, and wait 30 seconds for capacitors to discharge.",
                "common_mistakes": "Forgetting to disconnect battery, not waiting for discharge, working on powered board."
            },
            "esd_protection": {
                "why": "Electrostatic discharge can instantly damage sensitive motherboard components, especially ICs and connectors.",
                "how": "Wear an ESD wrist strap connected to a grounded surface, work on an ESD mat, and avoid synthetic clothing.",
                "common_mistakes": "Not grounding properly, working on non-ESD surfaces, touching components directly."
            },
            "zif_connector": {
                "why": "ZIF (Zero Insertion Force) connectors use a locking tab mechanism to secure ribbon cables without applying insertion force that could damage pins.",
                "how": "Lift the tab to 90 degrees, insert cable fully, then press tab down firmly until it locks.",
                "common_mistakes": "Forcing cable without opening tab, not inserting fully, breaking the fragile tab mechanism."
            },
            "ram_insertion": {
                "why": "RAM modules must be inserted at an angle to align the notch, then pressed down to engage retention clips properly.",
                "how": "Insert at 30-degree angle, align notch, then press down evenly on both sides until clips engage.",
                "common_mistakes": "Wrong insertion angle, forcing without alignment, not engaging both retention clips."
            },
            "polarity_check": {
                "why": "Reversed polarity can cause immediate component failure, short circuits, or even fire hazards.",
                "how": "Match + and - markings on both connector and motherboard before insertion.",
                "common_mistakes": "Assuming orientation, not checking markings, forcing incorrect polarity."
            },
            "verification": {
                "why": "Verification ensures proper installation, prevents intermittent connections, and catches errors before final assembly.",
                "how": "Gently test connection security, visually inspect alignment, check for exposed pins or loose connections.",
                "common_mistakes": "Skipping verification, not checking all connection points, assuming it's correct."
            }
        }
    
    def generate_explanations(
        self,
        sop_steps: List[str],
        target_component: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate detailed explanations for each SOP step"""
        explanations = []
        connector_type = target_component.get("connector_type", "")
        risk_level = target_component.get("risk_level", "Medium")
        
        for idx, step in enumerate(sop_steps, 1):
            explanation = self._explain_step(step, idx, connector_type, risk_level)
            explanations.append({
                "step_number": idx,
                "step_text": step,
                "explanation": explanation["explanation"],
                "why_important": explanation["why"],
                "common_mistakes": explanation["mistakes"],
                "quality_check": explanation["quality_check"]
            })
        
        return explanations
    
    def _explain_step(
        self,
        step: str,
        step_num: int,
        connector_type: str,
        risk_level: str
    ) -> Dict[str, str]:
        """Generate explanation for a single step"""
        step_lower = step.lower()
        
        # Power off explanation
        if "power" in step_lower and ("off" in step_lower or "disconnect" in step_lower):
            template = self.explanation_templates["power_off"]
            return {
                "explanation": f"{template['how']} {template['why']}",
                "why": template["why"],
                "mistakes": template["common_mistakes"],
                "quality_check": "Verify no LED indicators are lit, use multimeter to confirm no voltage present."
            }
        
        # ESD explanation
        if "esd" in step_lower or "wrist strap" in step_lower:
            template = self.explanation_templates["esd_protection"]
            return {
                "explanation": f"{template['how']} {template['why']}",
                "why": template["why"],
                "mistakes": template["common_mistakes"],
                "quality_check": "Verify wrist strap continuity, check grounding connection."
            }
        
        # ZIF connector explanation
        if "zif" in step_lower or "locking tab" in step_lower:
            template = self.explanation_templates["zif_connector"]
            return {
                "explanation": f"{template['how']} {template['why']}",
                "why": template["why"],
                "mistakes": template["common_mistakes"],
                "quality_check": "Verify tab is fully locked, check cable alignment, ensure no pin exposure."
            }
        
        # RAM explanation
        if "ram" in step_lower or "memory" in step_lower:
            template = self.explanation_templates["ram_insertion"]
            return {
                "explanation": f"{template['how']} {template['why']}",
                "why": template["why"],
                "mistakes": template["common_mistakes"],
                "quality_check": "Verify retention clips engaged, check module alignment, ensure parallel seating."
            }
        
        # Polarity explanation
        if "polarity" in step_lower or ("+" in step and "-" in step):
            template = self.explanation_templates["polarity_check"]
            return {
                "explanation": f"{template['how']} {template['why']}",
                "why": template["why"],
                "mistakes": template["common_mistakes"],
                "quality_check": "Double-check polarity markings match, verify before power-on."
            }
        
        # Verification explanation
        if "verify" in step_lower or "inspect" in step_lower or "check" in step_lower:
            template = self.explanation_templates["verification"]
            return {
                "explanation": f"{template['how']} {template['why']}",
                "why": template["why"],
                "mistakes": template["common_mistakes"],
                "quality_check": "Perform visual and physical verification, document findings."
            }
        
        # Generic explanation
        return {
            "explanation": f"This step is critical for proper component installation. Follow the procedure carefully and verify each action.",
            "why": "Ensures correct installation and prevents damage to components.",
            "mistakes": "Rushing through steps, not following sequence, skipping verification.",
            "quality_check": "Verify step completion before proceeding to next step."
        }

