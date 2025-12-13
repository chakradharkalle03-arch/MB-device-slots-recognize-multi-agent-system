"""
LangGraph Orchestrator - Coordinates multi-agent workflow
"""
from typing import Dict, Any, TypedDict, List
from agents import (
    VisionAgent,
    KnowledgeAgent,
    TaskDetectionAgent,
    SOPPlanningAgent,
    ExplanationAgent,
    QAAgent,
    PDFGeneratorAgent
)
import os
from config import HUGGINGFACE_API_KEY

class AgentState(TypedDict):
    """State passed between agents"""
    image_path: str
    task: str
    components: List[Dict[str, Any]]
    target_component: Dict[str, Any]
    all_components_enriched: List[Dict[str, Any]]
    sop_data: Dict[str, Any]
    all_sop_data: List[Dict[str, Any]]  # SOPs for all components
    explanations: List[Dict[str, Any]]
    all_explanations: List[Dict[str, Any]]  # Explanations for all components
    qa_result: Dict[str, Any]
    all_qa_results: List[Dict[str, Any]]  # QA results for all components
    annotated_image_path: str
    pdf_path: str
    status: str
    error: str

class Orchestrator:
    """Orchestrates multi-agent workflow using LangGraph-style state machine"""
    
    def __init__(self):
        self.vision_agent = VisionAgent(HUGGINGFACE_API_KEY)
        self.knowledge_agent = KnowledgeAgent()
        self.task_detection_agent = TaskDetectionAgent()
        self.sop_planning_agent = SOPPlanningAgent()
        self.explanation_agent = ExplanationAgent()
        self.qa_agent = QAAgent()
        self.pdf_generator = PDFGeneratorAgent()
        
        # Create necessary directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(base_dir, "uploads"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "outputs"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "outputs", "annotated"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "outputs", "pdfs"), exist_ok=True)
    
    def execute_workflow(self, image_path: str, task: str = None) -> Dict[str, Any]:
        """
        Execute complete multi-agent workflow
        
        Args:
            image_path: Path to uploaded motherboard image
            task: Task description (e.g., "Install Keyboard")
        
        Returns:
            Complete workflow result with PDF path
        """
        state: AgentState = {
            "image_path": image_path,
            "task": task or "Analyze All Connectors",
            "components": [],
            "target_component": {},
            "all_components_enriched": [],
            "sop_data": {},
            "all_sop_data": [],
            "explanations": [],
            "all_explanations": [],
            "qa_result": {},
            "all_qa_results": [],
            "annotated_image_path": "",
            "pdf_path": "",
            "status": "processing",
            "error": ""
        }
        
        try:
            # Step 1: Vision Agent - Detect components
            state = self._run_vision_agent(state)
            if state["status"] == "error":
                return state
            
            # Step 2: Task Detection Agent - Automatically detect task if not provided
            if not task:
                state = self._run_task_detection_agent(state)
                if state["status"] == "error":
                    return state
            
            # Step 3: Knowledge Agent - Enrich ALL components
            state = self._run_knowledge_agent(state)
            if state["status"] == "error":
                return state
            
            # Step 4: Generate SOPs for ALL components
            state = self._run_all_sop_planning(state)
            if state["status"] == "error":
                return state
            
            # Step 5: Generate explanations for ALL components
            state = self._run_all_explanations(state)
            if state["status"] == "error":
                return state
            
            # Step 6: QA Agent - Validate all SOPs
            state = self._run_all_qa_validation(state)
            
            # Step 7: PDF Generator - Create report
            state = self._run_pdf_generator(state)
            if state["status"] == "error":
                return state
            
            state["status"] = "completed"
            return state
            
        except Exception as e:
            state["status"] = "error"
            state["error"] = str(e)
            return state
    
    def _run_vision_agent(self, state: AgentState) -> AgentState:
        """Run vision agent to detect components"""
        try:
            result = self.vision_agent.detect_components(state["image_path"])
            state["components"] = result.get("components", [])
            
            # Annotate image
            base_dir = os.path.dirname(os.path.abspath(__file__))
            annotated_path = os.path.join(
                base_dir,
                "outputs",
                "annotated",
                f"annotated_{os.path.basename(state['image_path'])}"
            )
            self.vision_agent.annotate_image(
                state["image_path"],
                state["components"],
                annotated_path
            )
            state["annotated_image_path"] = annotated_path
            
            return state
        except Exception as e:
            state["status"] = "error"
            state["error"] = f"Vision agent error: {str(e)}"
            return state
    
    def _run_task_detection_agent(self, state: AgentState) -> AgentState:
        """Run task detection agent to automatically determine task"""
        try:
            result = self.task_detection_agent.detect_task_from_components(
                state["components"]
            )
            state["task"] = result.get("task", "Install Keyboard")
            # Store detection info for reference
            state["task_detection"] = result
            return state
        except Exception as e:
            state["status"] = "error"
            state["error"] = f"Task detection agent error: {str(e)}"
            return state
    
    def _run_knowledge_agent(self, state: AgentState) -> AgentState:
        """Run knowledge agent to find and enrich target component and all components"""
        try:
            # Find target component
            result = self.knowledge_agent.find_target_component(
                state["task"],
                state["components"]
            )
            state["target_component"] = result.get("target_component", {})
            
            # Enrich ALL components for display
            all_enriched = []
            for comp in state["components"]:
                enriched = self.knowledge_agent.enrich_component(comp.copy())
                all_enriched.append(enriched)
            state["all_components_enriched"] = all_enriched
            
            return state
        except Exception as e:
            state["status"] = "error"
            state["error"] = f"Knowledge agent error: {str(e)}"
            return state
    
    def _run_all_sop_planning(self, state: AgentState) -> AgentState:
        """Generate SOPs for ALL detected components"""
        try:
            all_sop_data = []
            
            # Generate SOP for each enriched component
            for component in state["all_components_enriched"]:
                task_for_component = f"Install {component.get('name', 'Component')}"
                sop_data = self.sop_planning_agent.generate_sop(
                    component,
                    task_for_component,
                    state["all_components_enriched"]
                )
                all_sop_data.append({
                    "component": component,
                    "sop_data": sop_data
                })
            
            # Set first component's SOP as primary (for backward compatibility)
            if all_sop_data:
                state["target_component"] = all_sop_data[0]["component"]
                state["sop_data"] = all_sop_data[0]["sop_data"]
            
            state["all_sop_data"] = all_sop_data
            return state
        except Exception as e:
            state["status"] = "error"
            state["error"] = f"SOP planning agent error: {str(e)}"
            return state
    
    def _run_all_explanations(self, state: AgentState) -> AgentState:
        """Generate explanations for ALL components"""
        try:
            all_explanations = []
            
            for sop_item in state["all_sop_data"]:
                component = sop_item["component"]
                sop_data = sop_item["sop_data"]
                explanations = self.explanation_agent.generate_explanations(
                    sop_data.get("sop_steps", []),
                    component
                )
                all_explanations.append({
                    "component_name": component.get("name", "Unknown"),
                    "explanations": explanations
                })
            
            # Set first component's explanations as primary (for backward compatibility)
            if all_explanations:
                state["explanations"] = all_explanations[0]["explanations"]
            
            state["all_explanations"] = all_explanations
            return state
        except Exception as e:
            state["status"] = "error"
            state["error"] = f"Explanation agent error: {str(e)}"
            return state
    
    def _run_all_qa_validation(self, state: AgentState) -> AgentState:
        """Validate SOPs for ALL components"""
        try:
            all_qa_results = []
            
            for i, sop_item in enumerate(state["all_sop_data"]):
                component = sop_item["component"]
                sop_data = sop_item["sop_data"]
                explanations = state["all_explanations"][i]["explanations"] if i < len(state["all_explanations"]) else []
                
                qa_result = self.qa_agent.validate_sop(
                    sop_data.get("sop_steps", []),
                    component,
                    explanations
                )
                all_qa_results.append({
                    "component_name": component.get("name", "Unknown"),
                    "qa_result": qa_result
                })
            
            # Set first component's QA as primary (for backward compatibility)
            if all_qa_results:
                state["qa_result"] = all_qa_results[0]["qa_result"]
            
            state["all_qa_results"] = all_qa_results
            return state
        except Exception as e:
            state["status"] = "error"
            state["error"] = f"QA agent error: {str(e)}"
            return state
    
    def _run_pdf_generator(self, state: AgentState) -> AgentState:
        """Run PDF generator"""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            pdf_filename = f"sop_report_{os.path.splitext(os.path.basename(state['image_path']))[0]}.pdf"
            pdf_path = os.path.join(base_dir, "outputs", "pdfs", pdf_filename)
            
            self.pdf_generator.generate_pdf(
                pdf_path,
                state["image_path"],
                state["annotated_image_path"],
                state["task"],
                state["target_component"],
                state["sop_data"],
                state["explanations"],
                state["qa_result"],
                state.get("all_components_enriched", []),
                state.get("all_sop_data", []),
                state.get("all_explanations", []),
                state.get("all_qa_results", [])
            )
            
            # Store relative path for API response
            state["pdf_path"] = pdf_filename
            return state
        except Exception as e:
            state["status"] = "error"
            state["error"] = f"PDF generator error: {str(e)}"
            return state

