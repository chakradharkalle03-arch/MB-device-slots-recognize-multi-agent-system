"""
QA/Safety Agent - Validates SOP correctness and safety
"""
from typing import Dict, Any, List

class QAAgent:
    """Checks SOP correctness and detects unsafe or missing steps"""
    
    def __init__(self):
        self.required_elements = [
            "power_off",
            "esd_protection",
            "component_identification",
            "installation_procedure",
            "verification"
        ]
        
        self.safety_keywords = [
            "power off",
            "disconnect",
            "esd",
            "wrist strap",
            "verify",
            "check",
            "inspect"
        ]
    
    def validate_sop(
        self,
        sop_steps: List[str],
        target_component: Dict[str, Any],
        explanations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate SOP for completeness and safety"""
        issues = []
        warnings = []
        recommendations = []
        
        # Check for required safety elements
        sop_text = " ".join(sop_steps).lower()
        
        # Power off check
        if not any(keyword in sop_text for keyword in ["power off", "disconnect", "powered off"]):
            issues.append("Missing power-off instruction")
            recommendations.append("Add explicit power-off step at the beginning")
        
        # ESD protection check
        if not any(keyword in sop_text for keyword in ["esd", "wrist strap", "grounding"]):
            warnings.append("ESD protection not explicitly mentioned")
            recommendations.append("Recommend adding ESD wrist strap instruction")
        
        # Verification check
        if not any(keyword in sop_text for keyword in ["verify", "check", "inspect", "test"]):
            issues.append("Missing verification step")
            recommendations.append("Add verification step at the end")
        
        # Risk level specific checks
        risk_level = target_component.get("risk_level", "Medium")
        if risk_level == "High":
            if "polarity" not in sop_text:
                issues.append("High-risk component missing polarity check")
                recommendations.append("Add explicit polarity verification step")
            
            if "critical" not in sop_text.lower() and "warning" not in sop_text.lower():
                warnings.append("High-risk component should have explicit warnings")
        
        # Component-specific checks
        component_name = target_component.get("name", "")
        if "battery" in component_name.lower():
            if "polarity" not in sop_text:
                issues.append("Battery connector requires polarity check")
        
        if "zif" in component_name.lower() or "keyboard" in component_name.lower():
            if "locking tab" not in sop_text and "tab" not in sop_text:
                warnings.append("ZIF connector procedure should mention locking tab")
        
        # Determine overall status
        status = "Approved"
        if issues:
            status = "Needs Revision"
        elif warnings:
            status = "Approved with Warnings"
        
        return {
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "risk_level": risk_level,
            "safety_score": self._calculate_safety_score(sop_text, issues, warnings)
        }
    
    def _calculate_safety_score(self, sop_text: str, issues: List[str], warnings: List[str]) -> float:
        """Calculate safety score (0-100)"""
        score = 100.0
        
        # Deduct for issues
        score -= len(issues) * 20
        
        # Deduct for warnings
        score -= len(warnings) * 10
        
        # Bonus for safety keywords
        safety_count = sum(1 for keyword in self.safety_keywords if keyword in sop_text)
        score += min(safety_count * 2, 10)
        
        return max(0.0, min(100.0, score))

