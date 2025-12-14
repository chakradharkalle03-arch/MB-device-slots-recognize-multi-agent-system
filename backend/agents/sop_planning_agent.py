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
            ],
            "LED Connector": [
                "Ensure the motherboard is powered off and disconnected from all power sources.",
                "Locate the LED connector header on the motherboard (typically in the top-left corner, near power indicator area).",
                "Identify the 2-pin header and check for polarity markings (+ and -).",
                "Verify the LED cable polarity matches the header markings (positive to positive, negative to negative).",
                "Align the LED connector with the header pins, ensuring correct polarity orientation.",
                "Insert the connector vertically onto the header pins, applying gentle downward pressure.",
                "Push until the connector is fully seated (connector should sit flush with the header).",
                "Verify the connection is secure and check that both pins are properly engaged.",
                "Test the LED functionality by powering on the system (LED should light up if installed correctly)."
            ],
            "USB Connector": [
                "Power off the motherboard and disconnect all power sources.",
                "Locate the USB connector port on the motherboard (typically on the right edge, middle section, stacked vertically).",
                "Identify the USB port type (USB 2.0 has 4 pins, USB 3.0 has 9 pins) and check for port orientation.",
                "Verify the USB connector matches the port standard (USB 2.0 or USB 3.0).",
                "Align the USB connector with the port, ensuring proper orientation (check for keying/notch).",
                "Insert the connector horizontally into the port, applying even pressure.",
                "Push until the connector is fully seated and sits flush with the motherboard edge.",
                "Verify the connection is secure by gently pulling outward (should not disconnect easily).",
                "Check that all pins are properly engaged and the port is not damaged.",
                "Test USB functionality by connecting a USB device after power-on."
            ],
            "Power Connector": [
                "CRITICAL: Ensure the motherboard is completely powered off and disconnected from all power sources.",
                "Locate the DC power jack on the motherboard (typically on the right edge, bottom section, for AC adapter connection).",
                "Identify the power jack orientation and verify the voltage rating (typically 19V for laptops).",
                "Check the AC adapter voltage matches the motherboard requirement before connection.",
                "Verify the power jack polarity (center positive, outer negative is standard).",
                "Align the DC plug with the power jack, ensuring correct orientation.",
                "Insert the DC plug horizontally into the jack until it clicks into place.",
                "Verify the connection is secure and the plug cannot be easily removed.",
                "Perform a final safety check: ensure no exposed wires and verify voltage rating.",
                "WARNING: Do not power on until all other components are properly installed."
            ],
            "Audio Connector": [
                "Ensure the motherboard is powered off and disconnected from all power sources.",
                "Locate the audio jack connector on the motherboard (typically in the top-right corner, for headphone/microphone).",
                "Identify the audio jack type (3-pin TRS for headphones, 4-pin TRRS for headset with mic).",
                "Check the audio jack orientation and verify it matches the connector type.",
                "Align the audio connector with the jack, ensuring proper orientation.",
                "Insert the connector vertically into the jack until it clicks into place.",
                "Verify the connection is secure and the connector is fully seated.",
                "Check that all pins are properly engaged and the jack is not damaged.",
                "Test audio functionality by connecting headphones/headset after power-on."
            ],
            "SATA Connector": [
                "Power off the motherboard and disconnect all power sources.",
                "Locate the SATA connector port on the motherboard (typically on the left edge, lower-middle section, for storage drives).",
                "Identify the SATA port type (SATA I/II/III) and check for port orientation.",
                "Verify the SATA cable matches the port standard and check cable orientation (L-shaped key).",
                "Align the SATA data cable connector with the port, ensuring the L-shaped key matches.",
                "Insert the data connector horizontally into the port until it clicks into place.",
                "Connect the SATA power cable to the drive (15-pin power connector).",
                "Verify both data and power connections are secure and properly seated.",
                "Check that all pins are properly engaged and cables are not loose.",
                "Test drive detection by checking BIOS/UEFI after power-on."
            ]
        }
    
    def generate_sop(self, target_component: Dict[str, Any], task: str, all_components: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate SOP steps for the target component"""
        component_name = target_component.get("name", "Unknown")
        connector_type = target_component.get("connector_type", "Unknown")
        risk_level = target_component.get("risk_level", "Medium")
        location_desc = target_component.get("location_description", target_component.get("typical_location", ""))
        
        # Get base steps from template
        base_steps = self.step_templates.get(
            component_name,
            self._generate_generic_steps(component_name, connector_type, location_desc)
        )
        
        # Update location information in steps if available (avoid duplication)
        if location_desc:
            for i, step in enumerate(base_steps):
                if "locate" in step.lower() or "find" in step.lower():
                    # Only add location if not already present in step
                    if location_desc and location_desc not in step:
                        # Simple replacement: add location after "on the motherboard"
                        if "on the motherboard" in step and "(" not in step:
                            base_steps[i] = step.replace(
                                "on the motherboard",
                                f"on the motherboard ({location_desc})"
                            )
                        elif "typically" in step and location_desc not in step and "(" not in step:
                            # Replace "typically" with location + "typically"
                            base_steps[i] = step.replace(
                                "typically",
                                f"{location_desc}, typically"
                            )
        
        # Customize steps based on risk level
        if risk_level == "High":
            base_steps.insert(0, "WARNING: High-risk component. Double-check all safety procedures before proceeding.")
        
        # Add ESD protection step if not already present
        if not any("ESD" in step.upper() for step in base_steps):
            base_steps.insert(1, "Put on ESD wrist strap and ensure proper grounding.")
        
        # Note: Battery connector will be handled separately in the orchestrator
        # to ensure it only appears once at the end of the complete sequence
        
        return {
            "sop_steps": base_steps,
            "component_name": component_name,
            "connector_type": connector_type,
            "risk_level": risk_level,
            "total_steps": len(base_steps)
        }
    
    def _generate_generic_steps(self, component_name: str, connector_type: str, location_desc: str = "") -> List[str]:
        """Generate generic SOP steps for unknown components"""
        location_info = f" ({location_desc})" if location_desc else ""
        return [
            f"Ensure the motherboard is powered off and disconnected.",
            f"Locate the {component_name} ({connector_type}) on the motherboard{location_info}.",
            f"Identify the connection mechanism and orientation.",
            f"Align the connector/cable with the motherboard header.",
            f"Insert the connector following the proper orientation.",
            f"Secure the connection using the locking mechanism if present.",
            f"Verify the connection is secure and properly seated.",
            f"Perform a visual inspection to ensure proper installation."
        ]

