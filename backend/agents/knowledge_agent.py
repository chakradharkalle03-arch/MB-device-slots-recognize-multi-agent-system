"""
Hardware Knowledge Agent - Maps detected components to engineering knowledge
"""
from typing import Dict, Any, List

class KnowledgeAgent:
    """Converts vision output into engineering-level understanding"""
    
    def __init__(self):
        self.component_database = self._initialize_database()
    
    def _initialize_database(self) -> Dict[str, Dict]:
        """Initialize component knowledge database"""
        return {
            "Keyboard Connector": {
                "type": "ZIF Connector",
                "typical_location": "bottom edge",
                "orientation": "Flip-lock",
                "pin_count": "30-40 pins",
                "voltage": "3.3V",
                "risk_level": "Medium",
                "common_issues": [
                    "Ribbon cable misalignment",
                    "Locking tab damage",
                    "Cable insertion depth"
                ],
                "safety_notes": [
                    "ESD protection required",
                    "Handle locking tab gently",
                    "Verify cable orientation before locking"
                ]
            },
            "RAM Slot": {
                "type": "SO-DIMM Slot",
                "typical_location": "upper section",
                "orientation": "Angled insertion",
                "pin_count": "204 pins (DDR4)",
                "voltage": "1.2V",
                "risk_level": "Low",
                "common_issues": [
                    "Incorrect insertion angle",
                    "Not fully seated",
                    "Wrong DDR generation"
                ],
                "safety_notes": [
                    "ESD protection required",
                    "Insert at 30-degree angle",
                    "Verify notch alignment"
                ]
            },
            "Fan Connector": {
                "type": "4-pin PWM Connector",
                "typical_location": "near CPU area",
                "orientation": "Vertical insertion",
                "pin_count": "4 pins",
                "voltage": "12V",
                "risk_level": "Low",
                "common_issues": [
                    "Polarity reversal",
                    "Loose connection",
                    "Pin bending"
                ],
                "safety_notes": [
                    "Power off required",
                    "Check pin alignment",
                    "Verify connector orientation"
                ]
            },
            "Battery Connector": {
                "type": "JST Connector",
                "typical_location": "side edge",
                "orientation": "Horizontal insertion",
                "pin_count": "2-4 pins",
                "voltage": "7.4V-11.1V",
                "risk_level": "High",
                "common_issues": [
                    "Polarity reversal",
                    "Loose connection",
                    "Short circuit risk"
                ],
                "safety_notes": [
                    "CRITICAL: Power off required",
                    "Verify polarity before connection",
                    "Check for damaged pins",
                    "No power during installation"
                ]
            },
            "Display Connector": {
                "type": "LVDS/eDP Connector",
                "typical_location": "side edge",
                "orientation": "Horizontal insertion",
                "pin_count": "30-40 pins",
                "voltage": "3.3V-12V",
                "risk_level": "Medium",
                "common_issues": [
                    "Cable misalignment",
                    "Locking mechanism not engaged",
                    "Pin damage"
                ],
                "safety_notes": [
                    "ESD protection required",
                    "Handle connector carefully",
                    "Verify locking mechanism"
                ]
            },
            "LED Connector": {
                "type": "2-pin LED Header",
                "typical_location": "top-left corner",
                "orientation": "Vertical insertion",
                "pin_count": "2 pins",
                "voltage": "3.3V-5V",
                "risk_level": "Low",
                "common_issues": [
                    "Polarity reversal",
                    "Loose connection",
                    "LED not lighting"
                ],
                "safety_notes": [
                    "ESD protection recommended",
                    "Check polarity markings",
                    "Verify LED orientation"
                ]
            },
            "USB Connector": {
                "type": "USB Type-A Port",
                "typical_location": "right edge",
                "orientation": "Horizontal insertion",
                "pin_count": "4 pins (USB 2.0) or 9 pins (USB 3.0)",
                "voltage": "5V",
                "risk_level": "Low",
                "common_issues": [
                    "Port damage",
                    "Loose connection",
                    "USB device not recognized"
                ],
                "safety_notes": [
                    "Power off recommended",
                    "Check port alignment",
                    "Verify USB standard compatibility"
                ]
            },
            "Power Connector": {
                "type": "DC Jack",
                "typical_location": "right edge, bottom",
                "orientation": "Horizontal insertion",
                "pin_count": "1 pin (center positive)",
                "voltage": "19V (typical)",
                "risk_level": "High",
                "common_issues": [
                    "Polarity reversal",
                    "Loose connection",
                    "Jack damage"
                ],
                "safety_notes": [
                    "CRITICAL: Power off required",
                    "Verify voltage rating",
                    "Check polarity before connection",
                    "No power during installation"
                ]
            },
            "Audio Connector": {
                "type": "Audio Jack",
                "typical_location": "top-right corner",
                "orientation": "Vertical insertion",
                "pin_count": "3-4 pins (TRS/TRRS)",
                "voltage": "3.3V",
                "risk_level": "Low",
                "common_issues": [
                    "Jack damage",
                    "Loose connection",
                    "Audio not working"
                ],
                "safety_notes": [
                    "ESD protection recommended",
                    "Handle jack carefully",
                    "Verify audio functionality after installation"
                ]
            },
            "SATA Connector": {
                "type": "SATA Port",
                "typical_location": "left edge, lower-middle",
                "orientation": "Horizontal insertion",
                "pin_count": "7 pins (data) + 15 pins (power)",
                "voltage": "5V/12V",
                "risk_level": "Medium",
                "common_issues": [
                    "Cable misalignment",
                    "Loose connection",
                    "Drive not detected"
                ],
                "safety_notes": [
                    "Power off required",
                    "Check cable orientation",
                    "Verify SATA standard (SATA I/II/III)"
                ]
            }
        }
    
    def enrich_component(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich detected component with knowledge base information"""
        component_name = component.get("name", "")
        # Preserve location_description if it exists from vision agent
        location_desc = component.get("location_description", "")
        
        if component_name in self.component_database:
            knowledge = self.component_database[component_name]
            component.update({
                "connector_type": knowledge["type"],
                "orientation": knowledge["orientation"],
                "risk_level": knowledge["risk_level"],
                "voltage": knowledge["voltage"],
                "pin_count": knowledge["pin_count"],
                "common_issues": knowledge["common_issues"],
                "safety_notes": knowledge["safety_notes"],
                "typical_location": knowledge.get("typical_location", "")
            })
            # Use knowledge base location if vision agent didn't provide one
            if not location_desc:
                component["location_description"] = knowledge.get("typical_location", "")
        else:
            # Default values for unknown components
            component.update({
                "connector_type": "Unknown",
                "orientation": "Unknown",
                "risk_level": "Medium",
                "voltage": "Unknown",
                "pin_count": "Unknown",
                "common_issues": [],
                "safety_notes": ["ESD protection recommended"],
                "typical_location": location_desc if location_desc else "Unknown location"
            })
            if not location_desc:
                component["location_description"] = "Unknown location"
        
        return component
    
    def find_target_component(self, task: str, components: List[Dict]) -> Dict[str, Any]:
        """Find the target component for a given task"""
        task_lower = task.lower()
        
        # Task to component mapping
        task_mapping = {
            "keyboard": "Keyboard Connector",
            "ram": "RAM Slot",
            "memory": "RAM Slot",
            "fan": "Fan Connector",
            "cooling": "Fan Connector",
            "battery": "Battery Connector",
            "power": "Battery Connector",
            "display": "Display Connector",
            "screen": "Display Connector"
        }
        
        target_name = None
        for keyword, component_name in task_mapping.items():
            if keyword in task_lower:
                target_name = component_name
                break
        
        # Find matching component
        for component in components:
            if target_name and component["name"] == target_name:
                enriched = self.enrich_component(component.copy())
                return {
                    "task_target": component["name"],
                    "target_component": enriched,
                    "status": "found"
                }
        
        # If not found, return first component with enrichment
        if components:
            enriched = self.enrich_component(components[0].copy())
            return {
                "task_target": components[0]["name"],
                "target_component": enriched,
                "status": "default"
            }
        
        return {
            "task_target": "Unknown",
            "target_component": None,
            "status": "not_found"
        }

