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
    sop_data: Dict[str, Any]
    explanations: List[Dict[str, Any]]
    qa_result: Dict[str, Any]
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
            "task": task or "",
            "components": [],
            "target_component": {},
            "all_components_enriched": [],
            "sop_data": {},
            "explanations": [],
            "qa_result": {},
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
            
            # Step 3: Knowledge Agent - Enrich components
            state = self._run_knowledge_agent(state)
            if state["status"] == "error":
                return state
            
            # Step 4: SOP Planning Agent - Generate steps
            state = self._run_sop_planning_agent(state)
            if state["status"] == "error":
                return state
            
            # Step 5: Explanation Agent - Generate explanations
            state = self._run_explanation_agent(state)
            if state["status"] == "error":
                return state
            
            # Step 6: QA Agent - Validate SOP
            state = self._run_qa_agent(state)
            
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
    
    def _run_sop_planning_agent(self, state: AgentState) -> AgentState:
        """Run SOP planning agent"""
        try:
            sop_data = self.sop_planning_agent.generate_sop(
                state["target_component"],
                state["task"]
            )
            state["sop_data"] = sop_data
            return state
        except Exception as e:
            state["status"] = "error"
            state["error"] = f"SOP planning agent error: {str(e)}"
            return state
    
    def _run_explanation_agent(self, state: AgentState) -> AgentState:
        """Run explanation agent"""
        try:
            explanations = self.explanation_agent.generate_explanations(
                state["sop_data"].get("sop_steps", []),
                state["target_component"]
            )
            state["explanations"] = explanations
            return state
        except Exception as e:
            state["status"] = "error"
            state["error"] = f"Explanation agent error: {str(e)}"
            return state
    
    def _run_qa_agent(self, state: AgentState) -> AgentState:
        """Run QA agent"""
        try:
            qa_result = self.qa_agent.validate_sop(
                state["sop_data"].get("sop_steps", []),
                state["target_component"],
                state["explanations"]
            )
            state["qa_result"] = qa_result
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
                state.get("all_components_enriched", [])
            )
            
            # Store relative path for API response
            state["pdf_path"] = pdf_filename
            return state
        except Exception as e:
            state["status"] = "error"
            state["error"] = f"PDF generator error: {str(e)}"
            return state

