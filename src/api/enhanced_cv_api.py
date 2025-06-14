"""API endpoints for enhanced CV generation features.

This module provides REST API endpoints that expose the enhanced CV generation functionality
including enhanced agents, orchestration, templates, and vector database operations.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..models.data_models import ContentType, ProcessingStatus, WorkflowType
from ..config.logging_config import get_structured_logger
from ..integration.enhanced_cv_system import (
    get_enhanced_cv_integration, EnhancedCVIntegration, EnhancedCVConfig, IntegrationMode
)


# Pydantic models for API requests/responses
class PersonalInfo(BaseModel):
    """Personal information model."""
    name: str
    email: str
    phone: Optional[str] = None
    address: Optional[str] = None
    linkedin: Optional[str] = None
    website: Optional[str] = None
    summary: Optional[str] = None


class Experience(BaseModel):
    """Work experience model."""
    title: str
    company: str
    location: Optional[str] = None
    start_date: str
    end_date: Optional[str] = None
    current: bool = False
    description: Optional[str] = None
    achievements: Optional[List[str]] = None
    technologies: Optional[List[str]] = None


class Education(BaseModel):
    """Education model."""
    degree: str
    institution: str
    location: Optional[str] = None
    graduation_date: Optional[str] = None
    gpa: Optional[str] = None
    honors: Optional[List[str]] = None
    relevant_coursework: Optional[List[str]] = None


class Project(BaseModel):
    """Project model."""
    name: str
    description: str
    technologies: Optional[List[str]] = None
    url: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    achievements: Optional[List[str]] = None


class JobDescription(BaseModel):
    """Job description model."""
    title: str
    company: str
    description: str
    requirements: List[str]
    preferred_qualifications: Optional[List[str]] = None
    location: Optional[str] = None
    salary_range: Optional[str] = None
    industry: Optional[str] = None


class CVGenerationRequest(BaseModel):
    """Base CV generation request."""
    personal_info: PersonalInfo
    experience: List[Experience]
    education: List[Education]
    skills: Optional[List[str]] = None
    projects: Optional[List[Project]] = None
    certifications: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    session_id: Optional[str] = None
    custom_options: Optional[Dict[str, Any]] = None


class JobTailoredCVRequest(CVGenerationRequest):
    """Job-tailored CV generation request."""
    job_description: JobDescription


class CVOptimizationRequest(BaseModel):
    """CV optimization request."""
    existing_cv: Dict[str, Any]
    target_role: Optional[str] = None
    industry: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    custom_options: Optional[Dict[str, Any]] = None


class QualityCheckRequest(BaseModel):
    """Quality check request."""
    cv_content: Dict[str, Any]
    quality_criteria: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class ContentSearchRequest(BaseModel):
    """Content search request."""
    query: str
    content_type: Optional[ContentType] = None
    limit: int = Field(default=5, ge=1, le=20)


class ContentStoreRequest(BaseModel):
    """Content store request."""
    content: str
    content_type: ContentType
    metadata: Optional[Dict[str, Any]] = None


class TemplateFormatRequest(BaseModel):
    """Template format request."""
    template_id: str
    variables: Dict[str, Any]
    category: Optional[str] = None


class WorkflowExecutionRequest(BaseModel):
    """Generic workflow execution request."""
    workflow_type: WorkflowType
    input_data: Dict[str, Any]
    session_id: Optional[str] = None
    custom_options: Optional[Dict[str, Any]] = None


class APIResponse(BaseModel):
    """Standard API response model."""
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    errors: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None


# Router setup
router = APIRouter(prefix="/api/v2", tags=["Enhanced CV Generation"])
logger = get_structured_logger(__name__)


# Dependency to get enhanced CV system integration
def get_integration() -> EnhancedCVIntegration:
    """Get enhanced CV system integration instance."""
    return get_enhanced_cv_integration()


# Utility functions
def create_session_id() -> str:
    """Create a new session ID."""
    return str(uuid4())


def handle_api_error(error: Exception, context: str) -> JSONResponse:
    """Handle API errors consistently."""
    logger.error(f"API error in {context}", extra={"error": str(error)})

    if isinstance(error, ValueError):
        status_code = 400
    elif isinstance(error, FileNotFoundError):
        status_code = 404
    else:
        status_code = 500

    return JSONResponse(
        status_code=status_code,
        content=APIResponse(
            success=False,
            message=f"Error in {context}",
            errors=[str(error)]
        ).model_dump()
    )


# CV Generation Endpoints
@router.post("/cv/generate/basic", response_model=APIResponse)
async def generate_basic_cv(
    request: CVGenerationRequest,
    background_tasks: BackgroundTasks,
    integration: EnhancedCVIntegration = Depends(get_integration)
):
    """Generate a basic CV with essential sections."""
    start_time = datetime.now()
    session_id = request.session_id or create_session_id()

    try:
        logger.info("Basic CV generation requested", extra={
            "session_id": session_id,
            "name": request.personal_info.name
        })

        result = await integration.generate_basic_cv(
            personal_info=request.personal_info.model_dump(),
            experience=[exp.model_dump() for exp in request.experience],
            education=[edu.model_dump() for edu in request.education],
            session_id=session_id,
            skills=request.skills,
            projects=[proj.model_dump() for proj in request.projects] if request.projects else None,
            certifications=request.certifications,
            languages=request.languages,
            **(request.custom_options or {})
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return APIResponse(
            success=result["success"],
            data=result["results"],
            metadata={
                **result["metadata"],
                "session_id": session_id,
                "workflow_type": "basic_cv_generation"
            },
            processing_time=processing_time,
            errors=result.get("errors")
        )

    except Exception as e:
        return handle_api_error(e, "basic CV generation")


@router.post("/cv/generate/job-tailored", response_model=APIResponse)
async def generate_job_tailored_cv(
    request: JobTailoredCVRequest,
    background_tasks: BackgroundTasks,
    integration: EnhancedCVIntegration = Depends(get_integration)
):
    """Generate a CV tailored to specific job requirements."""
    start_time = datetime.now()
    session_id = request.session_id or create_session_id()

    try:
        logger.info("Job-tailored CV generation requested", extra={
            "session_id": session_id,
            "name": request.personal_info.name,
            "target_role": request.job_description.title
        })

        result = await integration.generate_job_tailored_cv(
            personal_info=request.personal_info.model_dump(),
            experience=[exp.model_dump() for exp in request.experience],
            job_description=request.job_description.model_dump(),
            session_id=session_id,
            education=[edu.model_dump() for edu in request.education],
            skills=request.skills,
            projects=[proj.model_dump() for proj in request.projects] if request.projects else None,
            certifications=request.certifications,
            languages=request.languages,
            **(request.custom_options or {})
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return APIResponse(
            success=result["success"],
            data=result["results"],
            metadata={
                **result["metadata"],
                "session_id": session_id,
                "workflow_type": "job_tailored_cv",
                "target_role": request.job_description.title
            },
            processing_time=processing_time,
            errors=result.get("errors")
        )

    except Exception as e:
        return handle_api_error(e, "job-tailored CV generation")


@router.post("/cv/optimize", response_model=APIResponse)
async def optimize_cv(
    request: CVOptimizationRequest,
    background_tasks: BackgroundTasks,
    integration: EnhancedCVIntegration = Depends(get_integration)
):
    """Optimize an existing CV for better impact."""
    start_time = datetime.now()
    session_id = request.session_id or create_session_id()

    try:
        logger.info("CV optimization requested", extra={
            "session_id": session_id,
            "target_role": request.target_role
        })

        result = await integration.optimize_cv(
            existing_cv=request.existing_cv,
            session_id=session_id,
            target_role=request.target_role,
            industry=request.industry,
            preferences=request.preferences,
            **(request.custom_options or {})
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return APIResponse(
            success=result["success"],
            data=result["results"],
            metadata={
                **result["metadata"],
                "session_id": session_id,
                "workflow_type": "cv_optimization"
            },
            processing_time=processing_time,
            errors=result.get("errors")
        )

    except Exception as e:
        return handle_api_error(e, "CV optimization")


@router.post("/cv/quality-check", response_model=APIResponse)
async def check_cv_quality(
    request: QualityCheckRequest,
    background_tasks: BackgroundTasks,
    integration: EnhancedCVIntegration = Depends(get_integration)
):
    """Perform quality assurance on CV content."""
    start_time = datetime.now()
    session_id = request.session_id or create_session_id()

    try:
        logger.info("CV quality check requested", extra={
            "session_id": session_id
        })

        result = await integration.check_cv_quality(
            cv_content=request.cv_content,
            session_id=session_id,
            quality_criteria=request.quality_criteria
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return APIResponse(
            success=result["success"],
            data=result["results"],
            metadata={
                **result["metadata"],
                "session_id": session_id,
                "workflow_type": "quality_assurance"
            },
            processing_time=processing_time,
            errors=result.get("errors")
        )

    except Exception as e:
        return handle_api_error(e, "CV quality check")


# Generic Workflow Endpoint
@router.post("/workflow/execute", response_model=APIResponse)
async def execute_workflow(
    request: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks,
    integration: EnhancedCVIntegration = Depends(get_integration)
):
    """Execute any predefined workflow."""
    start_time = datetime.now()
    session_id = request.session_id or create_session_id()

    try:
        logger.info("Workflow execution requested", extra={
            "session_id": session_id,
            "workflow_type": request.workflow_type.value
        })

        result = await integration.execute_workflow(
            workflow_type=request.workflow_type,
            input_data=request.input_data,
            session_id=session_id,
            custom_options=request.custom_options
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return APIResponse(
            success=result["success"],
            data=result["results"],
            metadata={
                **result["metadata"],
                "session_id": session_id,
                "workflow_type": request.workflow_type.value
            },
            processing_time=processing_time,
            errors=result.get("errors")
        )

    except Exception as e:
        return handle_api_error(e, "workflow execution")


# Content Management Endpoints
@router.post("/content/search", response_model=APIResponse)
async def search_content(
    request: ContentSearchRequest,
    integration: EnhancedCVIntegration = Depends(get_integration)
):
    """Search for similar content in the vector database."""
    try:
        logger.info("Content search requested", extra={
            "query": request.query[:100],  # Log first 100 chars
            "content_type": request.content_type.value if request.content_type else None,
            "limit": request.limit
        })

        results = await integration.search_content(
            query=request.query,
            content_type=request.content_type,
            limit=request.limit
        )

        return APIResponse(
            success=True,
            data=results,
            metadata={
                "query": request.query,
                "content_type": request.content_type.value if request.content_type else None,
                "result_count": len(results)
            }
        )

    except Exception as e:
        return handle_api_error(e, "content search")


@router.post("/content/store", response_model=APIResponse)
async def store_content(
    request: ContentStoreRequest,
    integration: EnhancedCVIntegration = Depends(get_integration)
):
    """Store content in the vector database."""
    try:
        logger.info("Content store requested", extra={
            "content_type": request.content_type.value,
            "content_length": len(request.content)
        })

        document_id = await integration.store_content(
            content=request.content,
            content_type=request.content_type,
            metadata=request.metadata
        )

        return APIResponse(
            success=True,
            data={"document_id": document_id},
            metadata={
                "content_type": request.content_type.value,
                "content_length": len(request.content)
            }
        )

    except Exception as e:
        return handle_api_error(e, "content storage")


@router.post("/content/find-similar", response_model=APIResponse)
async def find_similar_content(
    request: ContentSearchRequest,
    integration: EnhancedCVIntegration = Depends(get_integration)
):
    """Find content similar to the provided content."""
    try:
        logger.info("Similar content search requested", extra={
            "content_length": len(request.query),
            "content_type": request.content_type.value if request.content_type else None,
            "limit": request.limit
        })

        results = await integration.find_similar_content(
            content=request.query,
            content_type=request.content_type,
            limit=request.limit
        )

        return APIResponse(
            success=True,
            data=results,
            metadata={
                "content_type": request.content_type.value if request.content_type else None,
                "result_count": len(results)
            }
        )

    except Exception as e:
        return handle_api_error(e, "similar content search")


# Template Management Endpoints
@router.get("/templates", response_model=APIResponse)
async def list_templates(
    category: Optional[str] = Query(None, description="Template category filter"),
    integration: EnhancedCVIntegration = Depends(get_integration)
):
    """List available content templates."""
    try:
        templates = integration.list_templates(category)

        return APIResponse(
            success=True,
            data=templates,
            metadata={
                "category": category,
                "template_count": len(templates)
            }
        )

    except Exception as e:
        return handle_api_error(e, "template listing")


@router.get("/templates/{template_id}", response_model=APIResponse)
async def get_template(
    template_id: str,
    category: Optional[str] = Query(None, description="Template category"),
    integration: EnhancedCVIntegration = Depends(get_integration)
):
    """Get a specific template."""
    try:
        template = integration.get_template(template_id, category)

        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        return APIResponse(
            success=True,
            data=template,
            metadata={
                "template_id": template_id,
                "category": category
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        return handle_api_error(e, "template retrieval")


@router.post("/templates/format", response_model=APIResponse)
async def format_template(
    request: TemplateFormatRequest,
    integration: EnhancedCVIntegration = Depends(get_integration)
):
    """Format a template with provided variables."""
    try:
        formatted_content = integration.format_template(
            template_id=request.template_id,
            variables=request.variables,
            category=request.category
        )

        if formatted_content is None:
            raise HTTPException(status_code=404, detail="Template not found")

        return APIResponse(
            success=True,
            data={"formatted_content": formatted_content},
            metadata={
                "template_id": request.template_id,
                "category": request.category,
                "variable_count": len(request.variables)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        return handle_api_error(e, "template formatting")


# System Management Endpoints
@router.get("/system/health", response_model=APIResponse)
async def system_health(
    integration: EnhancedCVIntegration = Depends(get_integration)
):
    """Get system health status."""
    try:
        health = integration.health_check()

        return APIResponse(
            success=health["status"] != "unhealthy",
            data=health,
            metadata={"check_timestamp": health["timestamp"]}
        )

    except Exception as e:
        return handle_api_error(e, "health check")


@router.get("/system/stats", response_model=APIResponse)
async def system_stats(
    integration: EnhancedCVIntegration = Depends(get_integration)
):
    """Get system performance statistics."""
    try:
        stats = integration.get_performance_stats()

        return APIResponse(
            success=True,
            data=stats,
            metadata={"stats_timestamp": datetime.now().isoformat()}
        )

    except Exception as e:
        return handle_api_error(e, "statistics retrieval")


@router.post("/system/reset-stats", response_model=APIResponse)
async def reset_system_stats(
    integration: EnhancedCVIntegration = Depends(get_integration)
):
    """Reset system performance statistics."""
    try:
        integration.reset_performance_stats()

        return APIResponse(
            success=True,
            message="Performance statistics reset successfully",
            metadata={"reset_timestamp": datetime.now().isoformat()}
        )

    except Exception as e:
        return handle_api_error(e, "statistics reset")


@router.get("/workflows", response_model=APIResponse)
async def list_workflows(
    integration: EnhancedCVIntegration = Depends(get_integration)
):
    """List available workflow types."""
    try:
        workflows = [
            {
                "type": wf.value,
                "name": wf.value.replace("_", " ").title(),
                "description": f"Execute {wf.value.replace('_', ' ')} workflow"
            }
            for wf in WorkflowType
        ]

        return APIResponse(
            success=True,
            data=workflows,
            metadata={"workflow_count": len(workflows)}
        )

    except Exception as e:
        return handle_api_error(e, "workflow listing")


@router.get("/agents", response_model=APIResponse)
async def list_agents(
    integration: EnhancedCVIntegration = Depends(get_integration)
):
    """List available agent types."""
    try:
        agents = integration.list_agents()

        return APIResponse(
            success=True,
            data=agents,
            metadata={"agent_count": len(agents)}
        )

    except Exception as e:
        return handle_api_error(e, "agent listing")