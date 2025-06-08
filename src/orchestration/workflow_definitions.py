"""Predefined workflow definitions for common CV generation scenarios."""

import asyncio
from datetime import timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..models.data_models import ContentType, ProcessingStatus
from ..agents.agent_base import AgentExecutionContext
from .agent_orchestrator import (
    AgentOrchestrator, AgentTask, OrchestrationPlan, OrchestrationStrategy,
    AgentPriority, OrchestrationResult
)


class WorkflowType(Enum):
    """Types of predefined workflows."""
    BASIC_CV_GENERATION = "basic_cv_generation"
    JOB_TAILORED_CV = "job_tailored_cv"
    CV_OPTIMIZATION = "cv_optimization"
    QUALITY_ASSURANCE = "quality_assurance"
    COMPREHENSIVE_CV = "comprehensive_cv"
    QUICK_UPDATE = "quick_update"
    MULTI_LANGUAGE_CV = "multi_language_cv"
    INDUSTRY_SPECIFIC = "industry_specific"


@dataclass
class WorkflowDefinition:
    """Definition of a workflow with its steps and configuration."""
    name: str
    description: str
    workflow_type: WorkflowType
    strategy: OrchestrationStrategy
    steps: List[Dict[str, Any]]  # Agent configurations
    estimated_duration: timedelta
    required_inputs: List[str]
    optional_inputs: List[str]
    output_types: List[ContentType]
    metadata: Dict[str, Any]


class WorkflowBuilder:
    """Builder for creating workflow definitions and executing them."""
    
    def __init__(self, orchestrator: Optional[AgentOrchestrator] = None):
        self.orchestrator = orchestrator
        self.workflows: Dict[WorkflowType, WorkflowDefinition] = {}
        self._register_default_workflows()
    
    def _register_default_workflows(self):
        """Register all default workflow definitions."""
        
        # Basic CV Generation Workflow
        self.workflows[WorkflowType.BASIC_CV_GENERATION] = WorkflowDefinition(
            name="Basic CV Generation",
            description="Generate a basic CV with essential sections",
            workflow_type=WorkflowType.BASIC_CV_GENERATION,
            strategy=OrchestrationStrategy.SEQUENTIAL,
            steps=[
                {
                    "agent_type": "content_writer",
                    "priority": AgentPriority.NORMAL,
                    "content_types": [ContentType.EXPERIENCE, ContentType.EDUCATION],
                    "timeout": timedelta(minutes=5),
                    "dependencies": []
                },
                {
                    "agent_type": "quality_assurance",
                    "priority": AgentPriority.NORMAL,
                    "content_types": [ContentType.QUALITY_CHECK],
                    "timeout": timedelta(minutes=2),
                    "dependencies": [0]
                }
            ],
            estimated_duration=timedelta(minutes=7),
            required_inputs=["personal_info", "experience", "education"],
            optional_inputs=["skills", "projects"],
            output_types=[ContentType.EXPERIENCE, ContentType.EDUCATION, ContentType.QUALITY_CHECK],
            metadata={"complexity": "low", "target_audience": "general"}
        )
        
        # Job-Tailored CV Workflow
        self.workflows[WorkflowType.JOB_TAILORED_CV] = WorkflowDefinition(
            name="Job-Tailored CV Generation",
            description="Generate a CV tailored to specific job requirements",
            workflow_type=WorkflowType.JOB_TAILORED_CV,
            strategy=OrchestrationStrategy.PIPELINE,
            steps=[
                {
                    "agent_type": "cv_analysis",
                    "priority": AgentPriority.HIGH,
                    "content_types": [ContentType.ANALYSIS],
                    "timeout": timedelta(minutes=3),
                    "dependencies": [],
                    "options": {"analyze_job_fit": True, "suggest_improvements": True}
                },
                {
                    "agent_type": "content_writer",
                    "priority": AgentPriority.HIGH,
                    "content_types": [ContentType.EXPERIENCE, ContentType.SKILLS],
                    "timeout": timedelta(minutes=8),
                    "dependencies": [0],
                    "options": {"optimize_for_job": True, "use_keywords": True}
                },
                {
                    "agent_type": "content_optimization",
                    "priority": AgentPriority.NORMAL,
                    "content_types": [ContentType.OPTIMIZATION],
                    "timeout": timedelta(minutes=4),
                    "dependencies": [1],
                    "options": {"enhance_relevance": True}
                },
                {
                    "agent_type": "quality_assurance",
                    "priority": AgentPriority.NORMAL,
                    "content_types": [ContentType.QUALITY_CHECK],
                    "timeout": timedelta(minutes=3),
                    "dependencies": [2]
                }
            ],
            estimated_duration=timedelta(minutes=18),
            required_inputs=["personal_info", "experience", "job_description"],
            optional_inputs=["skills", "education", "projects", "preferences"],
            output_types=[
                ContentType.ANALYSIS, ContentType.EXPERIENCE, ContentType.SKILLS,
                ContentType.OPTIMIZATION, ContentType.QUALITY_CHECK
            ],
            metadata={"complexity": "high", "target_audience": "job_specific"}
        )
        
        # CV Optimization Workflow
        self.workflows[WorkflowType.CV_OPTIMIZATION] = WorkflowDefinition(
            name="CV Optimization",
            description="Optimize existing CV content for better impact",
            workflow_type=WorkflowType.CV_OPTIMIZATION,
            strategy=OrchestrationStrategy.PARALLEL,
            steps=[
                {
                    "agent_type": "cv_analysis",
                    "priority": AgentPriority.HIGH,
                    "content_types": [ContentType.ANALYSIS],
                    "timeout": timedelta(minutes=3),
                    "dependencies": [],
                    "options": {"identify_weaknesses": True, "suggest_improvements": True}
                },
                {
                    "agent_type": "content_optimization",
                    "priority": AgentPriority.HIGH,
                    "content_types": [ContentType.OPTIMIZATION],
                    "timeout": timedelta(minutes=6),
                    "dependencies": [],
                    "options": {"enhance_impact": True, "improve_clarity": True}
                },
                {
                    "agent_type": "quality_assurance",
                    "priority": AgentPriority.NORMAL,
                    "content_types": [ContentType.QUALITY_CHECK],
                    "timeout": timedelta(minutes=3),
                    "dependencies": [0, 1]
                }
            ],
            estimated_duration=timedelta(minutes=12),
            required_inputs=["existing_cv"],
            optional_inputs=["target_role", "industry", "preferences"],
            output_types=[ContentType.ANALYSIS, ContentType.OPTIMIZATION, ContentType.QUALITY_CHECK],
            metadata={"complexity": "medium", "target_audience": "existing_cv"}
        )
        
        # Comprehensive CV Workflow
        self.workflows[WorkflowType.COMPREHENSIVE_CV] = WorkflowDefinition(
            name="Comprehensive CV Generation",
            description="Generate a complete, comprehensive CV with all sections",
            workflow_type=WorkflowType.COMPREHENSIVE_CV,
            strategy=OrchestrationStrategy.ADAPTIVE,
            steps=[
                {
                    "agent_type": "cv_analysis",
                    "priority": AgentPriority.HIGH,
                    "content_types": [ContentType.ANALYSIS],
                    "timeout": timedelta(minutes=4),
                    "dependencies": []
                },
                {
                    "agent_type": "content_writer",
                    "priority": AgentPriority.HIGH,
                    "content_types": [
                        ContentType.EXPERIENCE, ContentType.EDUCATION,
                        ContentType.SKILLS, ContentType.PROJECTS
                    ],
                    "timeout": timedelta(minutes=12),
                    "dependencies": [0]
                },
                {
                    "agent_type": "content_optimization",
                    "priority": AgentPriority.NORMAL,
                    "content_types": [ContentType.OPTIMIZATION],
                    "timeout": timedelta(minutes=6),
                    "dependencies": [1]
                },
                {
                    "agent_type": "quality_assurance",
                    "priority": AgentPriority.NORMAL,
                    "content_types": [ContentType.QUALITY_CHECK],
                    "timeout": timedelta(minutes=4),
                    "dependencies": [2]
                }
            ],
            estimated_duration=timedelta(minutes=26),
            required_inputs=["personal_info", "experience", "education"],
            optional_inputs=[
                "skills", "projects", "certifications", "languages",
                "job_description", "preferences"
            ],
            output_types=[
                ContentType.ANALYSIS, ContentType.EXPERIENCE, ContentType.EDUCATION,
                ContentType.SKILLS, ContentType.PROJECTS, ContentType.OPTIMIZATION,
                ContentType.QUALITY_CHECK
            ],
            metadata={"complexity": "very_high", "target_audience": "comprehensive"}
        )
        
        # Quick Update Workflow
        self.workflows[WorkflowType.QUICK_UPDATE] = WorkflowDefinition(
            name="Quick CV Update",
            description="Quick update of specific CV sections",
            workflow_type=WorkflowType.QUICK_UPDATE,
            strategy=OrchestrationStrategy.SEQUENTIAL,
            steps=[
                {
                    "agent_type": "content_writer",
                    "priority": AgentPriority.HIGH,
                    "content_types": [ContentType.EXPERIENCE],
                    "timeout": timedelta(minutes=3),
                    "dependencies": [],
                    "options": {"quick_mode": True}
                },
                {
                    "agent_type": "quality_assurance",
                    "priority": AgentPriority.NORMAL,
                    "content_types": [ContentType.QUALITY_CHECK],
                    "timeout": timedelta(minutes=1),
                    "dependencies": [0],
                    "options": {"quick_check": True}
                }
            ],
            estimated_duration=timedelta(minutes=4),
            required_inputs=["update_sections"],
            optional_inputs=["new_experience", "new_skills"],
            output_types=[ContentType.EXPERIENCE, ContentType.QUALITY_CHECK],
            metadata={"complexity": "low", "target_audience": "quick_update"}
        )
        
        # Quality Assurance Only Workflow
        self.workflows[WorkflowType.QUALITY_ASSURANCE] = WorkflowDefinition(
            name="Quality Assurance Check",
            description="Comprehensive quality check of existing CV",
            workflow_type=WorkflowType.QUALITY_ASSURANCE,
            strategy=OrchestrationStrategy.SEQUENTIAL,
            steps=[
                {
                    "agent_type": "quality_assurance",
                    "priority": AgentPriority.HIGH,
                    "content_types": [ContentType.QUALITY_CHECK],
                    "timeout": timedelta(minutes=5),
                    "dependencies": [],
                    "options": {
                        "check_grammar": True,
                        "check_consistency": True,
                        "check_formatting": True,
                        "check_completeness": True
                    }
                }
            ],
            estimated_duration=timedelta(minutes=5),
            required_inputs=["cv_content"],
            optional_inputs=["quality_criteria"],
            output_types=[ContentType.QUALITY_CHECK],
            metadata={"complexity": "low", "target_audience": "quality_check"}
        )
    
    def get_workflow(self, workflow_type: WorkflowType) -> Optional[WorkflowDefinition]:
        """Get a workflow definition by type."""
        return self.workflows.get(workflow_type)
    
    def list_workflows(self) -> List[WorkflowDefinition]:
        """List all available workflows."""
        return list(self.workflows.values())
    
    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a custom workflow."""
        self.workflows[workflow.workflow_type] = workflow
    
    async def execute_workflow(
        self,
        workflow_type: WorkflowType,
        input_data: Dict[str, Any],
        session_id: Optional[str] = None,
        custom_options: Optional[Dict[str, Any]] = None
    ) -> OrchestrationResult:
        """Execute a predefined workflow."""
        
        workflow = self.get_workflow(workflow_type)
        if not workflow:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        # Validate required inputs
        missing_inputs = [
            inp for inp in workflow.required_inputs
            if inp not in input_data
        ]
        if missing_inputs:
            raise ValueError(f"Missing required inputs: {missing_inputs}")
        
        # Get or create orchestrator
        if not self.orchestrator:
            from .agent_orchestrator import get_agent_orchestrator
            self.orchestrator = get_agent_orchestrator(session_id)
        
        # Create tasks from workflow steps
        tasks = []
        for i, step in enumerate(workflow.steps):
            # Create execution context
            context = AgentExecutionContext(
                session_id=session_id,
                input_data=input_data,
                content_type=step["content_types"][0],  # Primary content type
                processing_options={
                    **step.get("options", {}),
                    **(custom_options or {}),
                    "workflow_type": workflow_type.value,
                    "step_index": i
                }
            )
            
            # Create task
            task = self.orchestrator.create_task(
                agent_type=step["agent_type"],
                context=context,
                priority=step["priority"],
                dependencies=[tasks[dep_idx].id for dep_idx in step["dependencies"]],
                timeout=step.get("timeout"),
                metadata={
                    "workflow_type": workflow_type.value,
                    "step_name": step["agent_type"],
                    "step_index": i,
                    "content_types": step["content_types"]
                }
            )
            tasks.append(task)
        
        # Create orchestration plan
        plan = self.orchestrator.create_plan(
            strategy=workflow.strategy,
            tasks=tasks,
            timeout=workflow.estimated_duration * 2,  # Allow 2x estimated time
            metadata={
                "workflow_type": workflow_type.value,
                "workflow_name": workflow.name,
                "estimated_duration": workflow.estimated_duration.total_seconds()
            }
        )
        
        # Execute the plan
        return await self.orchestrator.execute_plan(plan)
    
    def create_custom_workflow(
        self,
        name: str,
        description: str,
        workflow_type: WorkflowType,
        agent_steps: List[Dict[str, Any]],
        strategy: OrchestrationStrategy = OrchestrationStrategy.SEQUENTIAL,
        estimated_duration: Optional[timedelta] = None,
        required_inputs: Optional[List[str]] = None,
        optional_inputs: Optional[List[str]] = None,
        output_types: Optional[List[ContentType]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowDefinition:
        """Create a custom workflow definition."""
        
        # Calculate estimated duration if not provided
        if estimated_duration is None:
            total_minutes = sum(
                step.get("timeout", timedelta(minutes=5)).total_seconds() / 60
                for step in agent_steps
            )
            estimated_duration = timedelta(minutes=int(total_minutes))
        
        # Extract output types from steps if not provided
        if output_types is None:
            output_types = []
            for step in agent_steps:
                output_types.extend(step.get("content_types", []))
            output_types = list(set(output_types))  # Remove duplicates
        
        workflow = WorkflowDefinition(
            name=name,
            description=description,
            workflow_type=workflow_type,
            strategy=strategy,
            steps=agent_steps,
            estimated_duration=estimated_duration,
            required_inputs=required_inputs or [],
            optional_inputs=optional_inputs or [],
            output_types=output_types,
            metadata=metadata or {}
        )
        
        self.register_workflow(workflow)
        return workflow


# Global workflow builder instance
_workflow_builder = None


def get_workflow_builder(orchestrator: Optional[AgentOrchestrator] = None) -> WorkflowBuilder:
    """Get the global workflow builder instance."""
    global _workflow_builder
    if _workflow_builder is None:
        _workflow_builder = WorkflowBuilder(orchestrator)
    return _workflow_builder


# Convenience functions for common workflows
async def execute_basic_cv_generation(
    personal_info: Dict[str, Any],
    experience: List[Dict[str, Any]],
    education: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    **kwargs
) -> OrchestrationResult:
    """Execute basic CV generation workflow."""
    builder = get_workflow_builder()
    input_data = {
        "personal_info": personal_info,
        "experience": experience,
        "education": education,
        **kwargs
    }
    return await builder.execute_workflow(
        WorkflowType.BASIC_CV_GENERATION,
        input_data,
        session_id
    )


async def execute_job_tailored_cv(
    personal_info: Dict[str, Any],
    experience: List[Dict[str, Any]],
    job_description: Dict[str, Any],
    session_id: Optional[str] = None,
    **kwargs
) -> OrchestrationResult:
    """Execute job-tailored CV generation workflow."""
    builder = get_workflow_builder()
    input_data = {
        "personal_info": personal_info,
        "experience": experience,
        "job_description": job_description,
        **kwargs
    }
    return await builder.execute_workflow(
        WorkflowType.JOB_TAILORED_CV,
        input_data,
        session_id
    )


async def execute_cv_optimization(
    existing_cv: Dict[str, Any],
    session_id: Optional[str] = None,
    **kwargs
) -> OrchestrationResult:
    """Execute CV optimization workflow."""
    builder = get_workflow_builder()
    input_data = {
        "existing_cv": existing_cv,
        **kwargs
    }
    return await builder.execute_workflow(
        WorkflowType.CV_OPTIMIZATION,
        input_data,
        session_id
    )


async def execute_quality_assurance(
    cv_content: Dict[str, Any],
    session_id: Optional[str] = None,
    **kwargs
) -> OrchestrationResult:
    """Execute quality assurance workflow."""
    builder = get_workflow_builder()
    input_data = {
        "cv_content": cv_content,
        **kwargs
    }
    return await builder.execute_workflow(
        WorkflowType.QUALITY_ASSURANCE,
        input_data,
        session_id
    )


async def execute_comprehensive_cv(
    personal_info: Dict[str, Any],
    experience: List[Dict[str, Any]],
    education: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    **kwargs
) -> OrchestrationResult:
    """Execute comprehensive CV generation workflow."""
    builder = get_workflow_builder()
    input_data = {
        "personal_info": personal_info,
        "experience": experience,
        "education": education,
        **kwargs
    }
    return await builder.execute_workflow(
        WorkflowType.COMPREHENSIVE_CV,
        input_data,
        session_id
    )


async def execute_quick_update(
    update_sections: List[str],
    session_id: Optional[str] = None,
    **kwargs
) -> OrchestrationResult:
    """Execute quick CV update workflow."""
    builder = get_workflow_builder()
    input_data = {
        "update_sections": update_sections,
        **kwargs
    }
    return await builder.execute_workflow(
        WorkflowType.QUICK_UPDATE,
        input_data,
        session_id
    )