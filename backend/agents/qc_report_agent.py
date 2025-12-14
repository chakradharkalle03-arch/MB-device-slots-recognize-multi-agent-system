"""
QC Report Agent - Generates automated QC inspection reports
"""
from typing import Dict, Any, List
from datetime import datetime

class QCReportAgent:
    """Generates automated QC inspection reports"""
    
    def generate_qc_report(
        self,
        defects: List[Dict[str, Any]],
        pass_fail_status: str,
        part_number: str = "N/A",
        lot_number: str = "N/A",
        inspection_date: str = None
    ) -> Dict[str, Any]:
        """Generate comprehensive QC report"""
        
        if inspection_date is None:
            inspection_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate statistics
        total_defects = len(defects)
        critical_defects = len([d for d in defects if d.get("severity") == "Critical"])
        high_defects = len([d for d in defects if d.get("severity") == "High"])
        medium_defects = len([d for d in defects if d.get("severity") == "Medium"])
        
        # Defect type breakdown
        defect_types = {}
        for defect in defects:
            defect_type = defect.get("type", "Unknown")
            defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
        
        # Generate recommendations
        recommendations = self._generate_recommendations(defects, pass_fail_status)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(defects, pass_fail_status)
        
        report = {
            "part_number": part_number,
            "lot_number": lot_number,
            "inspection_date": inspection_date,
            "status": pass_fail_status,
            "quality_score": quality_score,
            "summary": {
                "total_defects": total_defects,
                "critical_defects": critical_defects,
                "high_defects": high_defects,
                "medium_defects": medium_defects,
                "defect_types": defect_types
            },
            "defects": defects,
            "recommendations": recommendations,
            "next_actions": self._determine_next_actions(pass_fail_status, critical_defects),
            "report_generated_time": datetime.now().isoformat()
        }
        
        return report
    
    def _generate_recommendations(
        self,
        defects: List[Dict[str, Any]],
        status: str
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if status == "FAIL":
            recommendations.append("REJECT: Board does not meet quality standards")
            recommendations.append("Investigate root causes in manufacturing process")
            
            critical_defects = [d for d in defects if d.get("severity") == "Critical"]
            if critical_defects:
                recommendations.append(f"CRITICAL: {len(critical_defects)} critical defects require immediate attention")
        
        elif status == "REVIEW":
            recommendations.append("Manual review recommended before acceptance")
            recommendations.append("Verify defect severity matches visual inspection")
        
        else:  # PASS
            recommendations.append("Board meets quality standards")
            if defects:
                recommendations.append("Monitor defect trends for process improvement")
        
        # Process-specific recommendations
        defect_types = [d.get("type") for d in defects]
        
        if "Solder Bridge" in defect_types:
            recommendations.append("Review stencil alignment and paste volume")
        
        if "Missing Component" in defect_types:
            recommendations.append("Check pick-and-place machine calibration")
        
        if "Component Misalignment" in defect_types:
            recommendations.append("Verify placement accuracy and board flatness")
        
        return recommendations
    
    def _calculate_quality_score(
        self,
        defects: List[Dict[str, Any]],
        status: str
    ) -> float:
        """Calculate quality score (0-100)"""
        base_score = 100.0
        
        # Deduct points for defects
        for defect in defects:
            severity = defect.get("severity", "Medium")
            if severity == "Critical":
                base_score -= 15
            elif severity == "High":
                base_score -= 8
            else:
                base_score -= 3
        
        # Status-based adjustment
        if status == "FAIL":
            base_score = max(0, base_score - 20)
        elif status == "REVIEW":
            base_score = max(0, base_score - 10)
        
        return max(0, min(100, round(base_score, 1)))
    
    def _determine_next_actions(
        self,
        status: str,
        critical_count: int
    ) -> List[str]:
        """Determine next actions based on inspection results"""
        actions = []
        
        if status == "FAIL":
            actions.append("1. Reject board and document defect locations")
            actions.append("2. Notify production supervisor")
            if critical_count > 0:
                actions.append("3. Stop production line if critical defects exceed threshold")
            actions.append("4. Initiate root cause analysis")
        elif status == "REVIEW":
            actions.append("1. Forward to QC supervisor for manual review")
            actions.append("2. Document findings for trend analysis")
        else:  # PASS
            actions.append("1. Approve board for next assembly stage")
            actions.append("2. Update quality records")
        
        return actions
    
    def generate_report_summary(self, report: Dict[str, Any]) -> str:
        """Generate human-readable report summary"""
        summary_lines = [
            f"QC Inspection Report - {report.get('part_number', 'N/A')}",
            f"Inspection Date: {report.get('inspection_date', 'N/A')}",
            f"Status: {report.get('status', 'UNKNOWN')}",
            f"Quality Score: {report.get('quality_score', 0)}/100",
            "",
            f"Total Defects: {report.get('summary', {}).get('total_defects', 0)}",
            f"Critical: {report.get('summary', {}).get('critical_defects', 0)}",
            f"High: {report.get('summary', {}).get('high_defects', 0)}",
            f"Medium: {report.get('summary', {}).get('medium_defects', 0)}",
            "",
            "Recommendations:"
        ]
        
        for rec in report.get('recommendations', []):
            summary_lines.append(f"  • {rec}")
        
        return "\n".join(summary_lines)

