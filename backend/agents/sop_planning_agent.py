"""
SOP Planning Agent - Generates manufacturing-ready SOP steps
"""
from typing import Dict, Any, List

class SOPPlanningAgent:
    """Generates clean, manufacturing-ready SOP steps"""
    
    def __init__(self):
        self.step_templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, List[str]]:
        """Initialize SOP step templates by component type"""
        return {
            "Keyboard Connector": [
                "Ensure the motherboard is powered off and disconnected from all power sources.",
                "Locate the keyboard ZIF connector on the motherboard (typically near the bottom edge).",
                "Identify the locking tab mechanism on the ZIF connector.",
                "Gently lift the ZIF connector locking tab to the open position (approximately 90 degrees).",
                "Align the keyboard ribbon cable with the connector, ensuring proper orientation (check cable markings).",
                "Insert the keyboard ribbon cable fully into the connector slot, ensuring even insertion depth.",
                "Press down the locking tab firmly until it clicks into the locked position.",
                "Verify the cable is securely seated by gently tugging on the cable (should not dislodge).",
                "Inspect the connector for proper alignment and ensure no exposed pins are visible."
            ],
            "RAM Slot": [
                "Power off the motherboard and disconnect all power sources.",
                "Locate the RAM slot(s) on the motherboard (typically in the upper section).",
                "Identify the notch alignment on both the RAM module and slot.",
                "Release any existing retention clips on the RAM slot if present.",
                "Hold the RAM module at a 30-degree angle, aligning the notch with the slot key.",
                "Insert the RAM module into the slot, applying even pressure on both sides.",
                "Press down firmly until the retention clips automatically engage and click into place.",
                "Verify the RAM module is fully seated and the clips are locked.",
                "Perform a visual inspection to ensure the module is parallel to the motherboard."
            ],
            "Fan Connector": [
                "Ensure the motherboard is powered off and disconnected.",
                "Locate the fan connector header on the motherboard (typically labeled 'FAN' or 'CPU_FAN').",
                "Identify the 4-pin connector orientation (check for keying/notch).",
                "Align the fan connector with the header, matching the keying mechanism.",
                "Insert the connector vertically onto the header pins, applying gentle downward pressure.",
                "Push until the connector is fully seated (connector should sit flush with the header).",
                "Verify the connection is secure by gently pulling upward (should not disconnect).",
                "Check that all 4 pins are properly engaged and no pins are bent or exposed."
            ],
            "Battery Connector": [
                "CRITICAL: Ensure the motherboard is completely powered off and disconnected from all power sources.",
                "Locate the battery connector on the motherboard (typically on the side edge).",
                "Identify the polarity markings on both the connector and motherboard (usually marked + and -).",
                "Verify the connector orientation matches the motherboard markings before insertion.",
                "Align the connector with the header, ensuring correct polarity alignment.",
                "Insert the connector horizontally, applying even pressure on both sides.",
                "Push until the connector clicks into place and is fully seated.",
                "Verify the connection is secure and check for any exposed pins or loose connections.",
                "Perform a final safety check: ensure no short circuits are possible and polarity is correct."
            ],
            "Display Connector": [
                "Power off the motherboard and disconnect all power sources.",
                "Locate the display connector on the motherboard (typically LVDS or eDP connector).",
                "Identify the locking mechanism on the connector (usually a sliding lock or flip tab).",
                "Release the locking mechanism to the open position.",
                "Align the display cable connector with the motherboard connector, checking orientation markers.",
                "Insert the connector horizontally, ensuring even insertion depth.",
                "Engage the locking mechanism until it clicks into the locked position.",
                "Verify the connector is securely locked and the cable cannot be easily removed.",
                "Inspect for proper alignment and ensure all pins are properly engaged."
            ]
        }
    
    def generate_sop(self, target_component: Dict[str, Any], task: str, all_components: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate SOP steps for the target component"""
        component_name = target_component.get("name", "Unknown")
        connector_type = target_component.get("connector_type", "Unknown")
        risk_level = target_component.get("risk_level", "Medium")
        
        # Get base steps from template
        base_steps = self.step_templates.get(
            component_name,
            self._generate_generic_steps(component_name, connector_type)
        )
        
        # Customize steps based on risk level
        if risk_level == "High":
            base_steps.insert(0, "WARNING: High-risk component. Double-check all safety procedures before proceeding.")
        
        # Add ESD protection step if not already present
        if not any("ESD" in step.upper() for step in base_steps):
            base_steps.insert(1, "Put on ESD wrist strap and ensure proper grounding.")
        
        # Battery connector MUST always be the final step if it exists
        if all_components:
            battery_component = None
            for comp in all_components:
                if "battery" in comp.get("name", "").lower():
                    battery_component = comp
                    break
            
            if battery_component:
                # If battery is NOT the target, add it as final step
                if component_name != "Battery Connector":
                    battery_steps = self.step_templates.get(
                        "Connect Battery",
                        self._generate_generic_steps("Battery Connector", "JST Connector")
                    )
                    # Remove power off and ESD from battery steps (already done)
                    battery_steps = [s for s in battery_steps if "power off" not in s.lower() and "esd" not in s.lower() and "disconnect" not in s.lower()]
                    # Add battery steps at the end
                    base_steps.append("")
                    base_steps.append("=== FINAL STEP: Connect Battery ===")
                    base_steps.extend(battery_steps)
                else:
                    # Battery IS the target - ensure it's marked as final
                    if not any("FINAL" in step.upper() for step in base_steps):
                        # Add marker that this is the final step
                        base_steps.append("")
                        base_steps.append("=== NOTE: This is the FINAL step - Battery connection must be done last ===")
        
        return {
            "sop_steps": base_steps,
            "component_name": component_name,
            "connector_type": connector_type,
            "risk_level": risk_level,
            "total_steps": len(base_steps)
        }
    
    def _generate_generic_steps(self, component_name: str, connector_type: str) -> List[str]:
        """Generate generic SOP steps for unknown components"""
        return [
            f"Ensure the motherboard is powered off and disconnected.",
            f"Locate the {component_name} ({connector_type}) on the motherboard.",
            f"Identify the connection mechanism and orientation.",
            f"Align the connector/cable with the motherboard header.",
            f"Insert the connector following the proper orientation.",
            f"Secure the connection using the locking mechanism if present.",
            f"Verify the connection is secure and properly seated.",
            f"Perform a visual inspection to ensure proper installation."
        ]

