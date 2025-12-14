"""
Equipment Monitoring Agent - Monitors equipment health and performance
"""
from typing import Dict, Any, List
from datetime import datetime, timedelta
import random

class EquipmentMonitoringAgent:
    """Monitors equipment status and collects metrics"""
    
    def __init__(self):
        self.equipment_types = [
            "Pick and Place Machine",
            "Reflow Oven",
            "AOI Inspection System",
            "Solder Paste Printer",
            "Wave Soldering Machine",
            "ICT Tester",
            "X-Ray Inspection",
            "Conveyor System"
        ]
    
    def collect_equipment_data(self, equipment_id: str, equipment_type: str) -> Dict[str, Any]:
        """
        Collect current equipment metrics
        
        In production, this would connect to MES/SCADA systems
        For PoC, we simulate realistic equipment data with some anomalies for demo
        """
        # Simulate equipment metrics with some anomalies for demo
        # EQ-002 (Reflow Oven) - simulate temperature anomaly
        # EQ-005 (Wave Soldering) - simulate vibration anomaly
        # EQ-007 (X-Ray) - simulate maintenance due
        
        if equipment_id == "EQ-002":
            # Reflow Oven - temperature trending high
            base_metrics = {
                "temperature": random.uniform(88, 92),  # Above normal (85 max)
                "vibration": random.uniform(0.1, 1.5),
                "pressure": random.uniform(0.8, 1.2),
                "cycle_count": random.randint(1000, 50000),
                "uptime_hours": random.uniform(100, 2000),
                "error_count_24h": random.randint(3, 8),  # Elevated errors
                "maintenance_due_hours": random.uniform(50, 150)
            }
        elif equipment_id == "EQ-005":
            # Wave Soldering - vibration anomaly
            base_metrics = {
                "temperature": random.uniform(20, 80),
                "vibration": random.uniform(3.2, 3.8),  # Above normal (3.0 max)
                "pressure": random.uniform(0.8, 1.2),
                "cycle_count": random.randint(1000, 50000),
                "uptime_hours": random.uniform(100, 2000),
                "error_count_24h": random.randint(2, 6),
                "maintenance_due_hours": random.uniform(100, 300)
            }
        elif equipment_id == "EQ-007":
            # X-Ray - maintenance due soon
            base_metrics = {
                "temperature": random.uniform(20, 80),
                "vibration": random.uniform(0.1, 2.0),
                "pressure": random.uniform(0.8, 1.2),
                "cycle_count": random.randint(1000, 50000),
                "uptime_hours": random.uniform(100, 2000),
                "error_count_24h": random.randint(0, 3),
                "maintenance_due_hours": random.uniform(-50, 80)  # Due soon or overdue
            }
        else:
            # Normal equipment
            base_metrics = {
                "temperature": random.uniform(20, 80),
                "vibration": random.uniform(0.1, 2.0),
                "pressure": random.uniform(0.8, 1.2),
                "cycle_count": random.randint(1000, 50000),
                "uptime_hours": random.uniform(100, 2000),
                "error_count_24h": random.randint(0, 2),
                "maintenance_due_hours": random.uniform(200, 500)
            }
        
        # Add equipment-specific metrics
        if "Oven" in equipment_type:
            base_metrics["zone_temperatures"] = [random.uniform(150, 250) for _ in range(8)]
            base_metrics["conveyor_speed"] = random.uniform(0.5, 1.5)
        
        elif "Printer" in equipment_type:
            base_metrics["squeegee_pressure"] = random.uniform(1.0, 3.0)
            base_metrics["print_cycles"] = random.randint(500, 10000)
        
        elif "AOI" in equipment_type:
            base_metrics["inspection_rate"] = random.uniform(0.8, 1.2)
            base_metrics["camera_status"] = "OK"
        
        return {
            "equipment_id": equipment_id,
            "equipment_type": equipment_type,
            "timestamp": datetime.now().isoformat(),
            "status": "RUNNING",
            "metrics": base_metrics
        }
    
    def get_equipment_list(self) -> List[Dict[str, Any]]:
        """Get list of all monitored equipment"""
        equipment_list = []
        for i, eq_type in enumerate(self.equipment_types, 1):
            equipment_list.append({
                "equipment_id": f"EQ-{i:03d}",
                "equipment_type": eq_type,
                "status": "RUNNING",
                "last_update": datetime.now().isoformat()
            })
        return equipment_list

