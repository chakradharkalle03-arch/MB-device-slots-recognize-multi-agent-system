"""
Anomaly Detection Agent - Detects equipment anomalies and predicts failures
"""
from typing import Dict, Any, List
from datetime import datetime, timedelta
import numpy as np

class AnomalyDetectionAgent:
    """Detects anomalies in equipment data and predicts potential failures"""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_thresholds = {
            "temperature": {"min": 15, "max": 85, "std_threshold": 2.0},
            "vibration": {"min": 0.0, "max": 3.0, "std_threshold": 2.0},
            "pressure": {"min": 0.7, "max": 1.5, "std_threshold": 2.0},
            "error_count_24h": {"min": 0, "max": 10, "std_threshold": 2.5}
        }
    
    def detect_anomalies(
        self,
        current_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect anomalies in equipment metrics
        
        Returns anomaly detection results with severity and predictions
        """
        equipment_id = current_data.get("equipment_id", "UNKNOWN")
        metrics = current_data.get("metrics", {})
        
        anomalies = []
        warnings = []
        predictions = []
        
        # Check each metric against thresholds
        for metric_name, value in metrics.items():
            if metric_name in self.anomaly_thresholds:
                threshold = self.anomaly_thresholds[metric_name]
                
                # Check if value is outside normal range
                if value < threshold["min"] or value > threshold["max"]:
                    severity = "Critical" if abs(value - (threshold["min"] + threshold["max"])/2) > threshold["max"] * 0.3 else "High"
                    anomalies.append({
                        "metric": metric_name,
                        "value": value,
                        "normal_range": f"{threshold['min']}-{threshold['max']}",
                        "severity": severity,
                        "message": f"{metric_name} is outside normal range: {value:.2f}"
                    })
        
        # Check for trend anomalies using historical data
        if historical_data and len(historical_data) > 5:
            trend_anomalies = self._detect_trend_anomalies(metrics, historical_data)
            anomalies.extend(trend_anomalies)
        
        # Generate failure predictions
        if anomalies:
            predictions = self._predict_failure_risk(anomalies, current_data)
        
        # Determine overall status
        critical_count = len([a for a in anomalies if a.get("severity") == "Critical"])
        high_count = len([a for a in anomalies if a.get("severity") == "High"])
        
        if critical_count > 0:
            overall_status = "CRITICAL"
            alert_level = "IMMEDIATE_ACTION"
        elif high_count >= 2:
            overall_status = "WARNING"
            alert_level = "REVIEW_REQUIRED"
        elif len(anomalies) > 0:
            overall_status = "MONITOR"
            alert_level = "INFORMATIONAL"
        else:
            overall_status = "NORMAL"
            alert_level = "NONE"
        
        return {
            "equipment_id": equipment_id,
            "timestamp": datetime.now().isoformat(),
            "status": overall_status,
            "alert_level": alert_level,
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "critical_anomalies": critical_count,
            "predictions": predictions,
            "recommended_actions": self._generate_actions(anomalies, overall_status)
        }
    
    def _detect_trend_anomalies(
        self,
        current_metrics: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies based on trends"""
        trend_anomalies = []
        
        # Analyze temperature trends
        if "temperature" in current_metrics:
            temps = [d.get("metrics", {}).get("temperature", 0) for d in historical_data[-10:]]
            if len(temps) > 3:
                avg_temp = np.mean(temps)
                std_temp = np.std(temps)
                current_temp = current_metrics["temperature"]
                
                # Detect sudden spike or drop
                if abs(current_temp - avg_temp) > 2 * std_temp:
                    trend_anomalies.append({
                        "metric": "temperature",
                        "value": current_temp,
                        "trend": "increasing" if current_temp > avg_temp else "decreasing",
                        "severity": "High",
                        "message": f"Temperature trend anomaly detected: {current_temp:.2f}°C (avg: {avg_temp:.2f}°C)"
                    })
        
        # Analyze vibration trends
        if "vibration" in current_metrics:
            vibrations = [d.get("metrics", {}).get("vibration", 0) for d in historical_data[-10:]]
            if len(vibrations) > 3:
                avg_vib = np.mean(vibrations)
                current_vib = current_metrics["vibration"]
                
                if current_vib > avg_vib * 1.5:
                    trend_anomalies.append({
                        "metric": "vibration",
                        "value": current_vib,
                        "trend": "increasing",
                        "severity": "High",
                        "message": f"Vibration increasing: {current_vib:.2f} (avg: {avg_vib:.2f})"
                    })
        
        return trend_anomalies
    
    def _predict_failure_risk(
        self,
        anomalies: List[Dict[str, Any]],
        current_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Predict potential failure based on anomalies"""
        predictions = []
        
        critical_anomalies = [a for a in anomalies if a.get("severity") == "Critical"]
        
        if critical_anomalies:
            # High risk of failure within days
            predictions.append({
                "type": "Equipment Failure Risk",
                "probability": "High",
                "timeframe": "3-7 days",
                "confidence": 0.75,
                "message": f"{len(critical_anomalies)} critical anomalies detected. Equipment failure likely within week."
            })
        
        # Check error count trend
        error_count = current_data.get("metrics", {}).get("error_count_24h", 0)
        if error_count > 5:
            predictions.append({
                "type": "Error Rate Increase",
                "probability": "Medium",
                "timeframe": "1-2 weeks",
                "confidence": 0.65,
                "message": f"Error count elevated: {error_count} errors in 24h. Monitor closely."
            })
        
        # Check maintenance due
        maintenance_due = current_data.get("metrics", {}).get("maintenance_due_hours", 500)
        if maintenance_due < 100:
            predictions.append({
                "type": "Maintenance Due",
                "probability": "High",
                "timeframe": "Immediate",
                "confidence": 0.90,
                "message": f"Maintenance due in {maintenance_due:.0f} hours. Schedule preventive maintenance."
            })
        
        return predictions
    
    def _generate_actions(
        self,
        anomalies: List[Dict[str, Any]],
        status: str
    ) -> List[str]:
        """Generate recommended actions"""
        actions = []
        
        if status == "CRITICAL":
            actions.append("IMMEDIATE: Stop equipment and inspect")
            actions.append("Notify maintenance team immediately")
            actions.append("Review recent production for quality issues")
        
        elif status == "WARNING":
            actions.append("Schedule equipment inspection within 24 hours")
            actions.append("Monitor metrics closely")
            actions.append("Prepare maintenance plan")
        
        elif status == "MONITOR":
            actions.append("Continue monitoring")
            actions.append("Document anomaly trends")
            actions.append("Schedule routine inspection")
        
        # Metric-specific actions
        for anomaly in anomalies:
            metric = anomaly.get("metric", "")
            if metric == "temperature":
                actions.append("Check cooling system and thermal management")
            elif metric == "vibration":
                actions.append("Inspect mechanical components and bearings")
            elif metric == "pressure":
                actions.append("Verify pressure sensors and calibration")
        
        return list(set(actions))  # Remove duplicates

