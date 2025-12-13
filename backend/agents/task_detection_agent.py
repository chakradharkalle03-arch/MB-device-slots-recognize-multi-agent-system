"""
Task Detection Agent - Automatically determines task from motherboard image analysis
"""
from typing import Dict, Any, List
import re

class TaskDetectionAgent:
    """Analyzes motherboard image to automatically determine installation task"""
    
    def __init__(self):
        self.task_patterns = self._initialize_task_patterns()
    
    def _initialize_task_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for task detection based on component analysis"""
        return {
            "Install Keyboard": {
                "keywords": ["keyboard", "zif", "ribbon", "flat cable", "connector"],
                "component_types": ["Keyboard Connector", "ZIF Connector"],
                "location_hints": ["bottom", "edge", "flat"]
            },
            "Install RAM": {
                "keywords": ["ram", "memory", "dimm", "sodimm", "module"],
                "component_types": ["RAM Slot", "SO-DIMM Slot"],
                "location_hints": ["upper", "center", "slot"]
            },
            "Connect Fan": {
                "keywords": ["fan", "cooling", "pwm", "4-pin", "thermal"],
                "component_types": ["Fan Connector", "4-pin PWM Connector"],
                "location_hints": ["cpu", "near processor", "thermal"]
            },
            "Connect Battery": {
                "keywords": ["battery", "power", "jst", "dc", "voltage"],
                "component_types": ["Battery Connector", "JST Connector", "Power Connector"],
                "location_hints": ["side", "edge", "power"]
            },
            "Connect Display": {
                "keywords": ["display", "screen", "lvds", "edp", "lcd"],
                "component_types": ["Display Connector", "LVDS/eDP Connector"],
                "location_hints": ["side", "edge", "display"]
            }
        }
    
    def detect_task_from_components(
        self,
        components: List[Dict[str, Any]],
        image_analysis: str = ""
    ) -> Dict[str, Any]:
        """
        Automatically detect task from detected components
        
        Args:
            components: List of detected components
            image_analysis: Optional text analysis of image
        
        Returns:
            Detected task information
        """
        if not components:
            return {
                "task": "Install Keyboard",
                "confidence": 0.5,
                "reasoning": "Default task - no components detected"
            }
        
        # Analyze components to determine most likely task
        component_scores = {}
        
        for component in components:
            comp_name = component.get("name", "").lower()
            comp_type = component.get("type", "").lower()
            
            # Score each possible task
            for task_name, patterns in self.task_patterns.items():
                score = 0
                
                # Check component name matches
                for keyword in patterns["keywords"]:
                    if keyword in comp_name or keyword in comp_type:
                        score += 3
                
                # Check component type matches
                for comp_type_pattern in patterns["component_types"]:
                    if comp_type_pattern.lower() in comp_name or comp_type_pattern.lower() in comp_type:
                        score += 5
                
                # Check location hints
                location = component.get("center", [])
                if location:
                    # Simple heuristic: keyboard usually bottom, RAM usually top
                    if "keyboard" in task_name.lower() and location[1] > 0.6:
                        score += 2
                    elif "ram" in task_name.lower() and location[1] < 0.4:
                        score += 2
                
                component_scores[task_name] = component_scores.get(task_name, 0) + score
        
        # Find task with highest score
        if component_scores:
            best_task = max(component_scores.items(), key=lambda x: x[1])
            task_name = best_task[0]
            score = best_task[1]
            
            # Calculate confidence (normalize score)
            max_possible_score = len(components) * 8  # Max score per component
            confidence = min(0.95, max(0.6, score / max_possible_score))
            
            # Find matching component
            matching_component = None
            for comp in components:
                comp_lower = comp.get("name", "").lower()
                for keyword in self.task_patterns[task_name]["keywords"]:
                    if keyword in comp_lower:
                        matching_component = comp
                        break
                if matching_component:
                    break
            
            reasoning = f"Detected {matching_component.get('name', 'component')} - most likely task is {task_name}"
            
            return {
                "task": task_name,
                "confidence": confidence,
                "reasoning": reasoning,
                "detected_component": matching_component or components[0],
                "all_scores": component_scores
            }
        
        # Fallback: use first component
        first_component = components[0]
        comp_name = first_component.get("name", "").lower()
        
        # Simple keyword matching
        if "keyboard" in comp_name or "zif" in comp_name:
            task = "Install Keyboard"
        elif "ram" in comp_name or "memory" in comp_name:
            task = "Install RAM"
        elif "fan" in comp_name:
            task = "Connect Fan"
        elif "battery" in comp_name or "power" in comp_name:
            task = "Connect Battery"
        elif "display" in comp_name or "screen" in comp_name:
            task = "Connect Display"
        else:
            task = "Install Keyboard"  # Default
        
        return {
            "task": task,
            "confidence": 0.7,
            "reasoning": f"Based on detected component: {first_component.get('name')}",
            "detected_component": first_component
        }
    
    def generate_task_prompt(self, components: List[Dict[str, Any]]) -> str:
        """
        Generate a prompt for LLM-based task detection
        
        Args:
            components: List of detected components
        
        Returns:
            Prompt string for task detection
        """
        component_list = "\n".join([
            f"- {comp.get('name')} ({comp.get('type')}) at location {comp.get('center', 'unknown')}"
            for comp in components
        ])
        
        prompt = f"""Analyze this laptop motherboard and determine the most likely installation task based on the detected components.

Detected Components:
{component_list}

Based on the components detected, what is the most likely task a technician needs to perform?

Consider:
1. Which component appears to be missing or needs installation?
2. Which component is most commonly installed/replaced?
3. Which component has the highest priority for assembly?

Respond with ONLY the task name in this format: "Install [Component]" or "Connect [Component]"

Examples:
- Install Keyboard
- Install RAM
- Connect Fan
- Connect Battery
- Connect Display

Task:"""
        
        return prompt

