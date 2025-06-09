from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import uuid
import json
import logging
import traceback
import sys
import time

# Import core components
from src.core.state_manager import StateManager, ItemStatus
from src.core.enhanced_orchestrator import EnhancedOrchestrator
from src.services.llm import LLM
from src.agents.parser_agent import ParserAgent
from src.agents.research_agent import ResearchAgent
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.formatter_agent import FormatterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.utils.template_renderer import TemplateRenderer
from src.utils.template_manager import TemplateManager
from src.agents.tools_agent import ToolsAgent
from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.agents.vector_store_agent import VectorStoreAgent
from src.services.vector_db import VectorDB
from src.core.state_manager import VectorStoreConfig, AgentIO
from src.config.logging_config import setup_logging, get_logger, log_request

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(title="CV Tailoring AI API")

# Setup templates
templates = Jinja2Templates(directory="src/frontend/templates")

# Mount static files
app.mount("/static", StaticFiles(directory="src/frontend/static"), name="static")

# Global state dictionary to store session data
sessions = {}


# Initialize core components
def initialize_components():
    """Initialize all AI components."""
    try:
        model = LLM()
        parser_agent = ParserAgent(
            name="ParserAgent",
            description="Agent for parsing job descriptions.",
            llm=model,
        )
        template_renderer = TemplateRenderer(
            name="TemplateRenderer",
            description="Agent for rendering CV templates.",
            model=model,
            input_schema=AgentIO(input={}, output={}, description="template renderer"),
            output_schema=AgentIO(input={}, output={}, description="template renderer"),
        )
        vector_db_config = VectorStoreConfig(dimension=768, index_type="IndexFlatL2")
        vector_db = VectorDB(config=vector_db_config)
        vector_store_agent = VectorStoreAgent(
            name="Vector Store Agent",
            description="Agent for managing vector store.",
            model=model,
            input_schema=AgentIO(input={}, output={}, description="vector store agent"),
            output_schema=AgentIO(input={}, output={}, description="vector store agent"),
            vector_db=vector_db,
        )
        tools_agent = ToolsAgent(name="ToolsAgent", description="Agent for content processing.")
        enhanced_content_writer = EnhancedContentWriterAgent(
            name="EnhancedContentWriterAgent",
            description="Enhanced agent for generating tailored CV content."
        )
        research_agent = ResearchAgent(
            name="ResearchAgent",
            description="Agent for researching job-related information.",
            llm=model,
        )
        cv_analyzer_agent = CVAnalyzerAgent(
            name="CVAnalyzerAgent", description="Agent for analyzing CVs.", llm=model
        )
        formatter_agent = FormatterAgent(
            name="FormatterAgent", description="Agent for formatting CV content."
        )
        quality_assurance_agent = QualityAssuranceAgent(
            name="QualityAssuranceAgent",
            description="Agent for quality assurance checks.",
        )
        orchestrator = EnhancedOrchestrator(
            llm_client=model
        )
        return orchestrator, parser_agent
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to initialize AI components")


# Create orchestrator instance
orchestrator, parser_agent = initialize_components()


# Pydantic models for API
class SessionRequest(BaseModel):
    session_id: Optional[str] = None


class CVInput(BaseModel):
    job_description: str
    cv_text: Optional[str] = None
    start_from_scratch: bool = False


class RegenerateRequest(BaseModel):
    session_id: str
    item_ids: List[str]


class ItemUpdateRequest(BaseModel):
    session_id: str
    item_id: str
    content: Optional[str] = None
    status: Optional[str] = None
    feedback: Optional[str] = None


class ExportRequest(BaseModel):
    session_id: str
    format: str = "markdown"


# Routes
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Return the index page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/session/new")
async def create_session():
    """Create a new session"""
    session_id = str(uuid.uuid4())
    state_manager = StateManager(session_id=session_id)
    sessions[session_id] = state_manager
    return {"session_id": session_id}


@app.post("/api/session/load")
async def load_session(request: SessionRequest):
    """Load an existing session"""
    if not request.session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")

    state_manager = StateManager(session_id=request.session_id)
    if state_manager.load_state():
        sessions[request.session_id] = state_manager
        return {"session_id": request.session_id, "status": "loaded"}
    else:
        raise HTTPException(status_code=404, detail="Session not found or failed to load")


@app.post("/api/cv/parse")
async def parse_cv(request: CVInput):
    """Parse the CV and job description"""
    try:
        # Create a new session
        session_id = str(uuid.uuid4())
        state_manager = StateManager(session_id=session_id)
        sessions[session_id] = state_manager

        # Parse the input
        parse_result = parser_agent.run(
            {
                "job_description": request.job_description,
                "cv_text": request.cv_text or "",
                "start_from_scratch": request.start_from_scratch,
            }
        )

        # Store results in state manager
        job_data = parse_result["job_description_data"]
        structured_cv = parse_result["structured_cv"]

        # Store in state manager
        state_manager._structured_cv = structured_cv

        # Save state
        state_file = state_manager.save_state()

        # Get items that need regeneration
        items_to_generate = []
        if structured_cv:
            items_to_generate = [
                item.id for item in structured_cv.get_items_by_status(ItemStatus.TO_REGENERATE)
            ]

        return {
            "session_id": session_id,
            "cv_id": structured_cv.id if structured_cv else None,
            "items_to_regenerate": items_to_generate,
            "status": "parsed",
        }
    except Exception as e:
        logger.error(f"Error parsing CV: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cv/structure/{session_id}")
async def get_cv_structure(session_id: str):
    """Get the current CV structure"""
    if session_id not in sessions:
        # Try to load from storage
        state_manager = StateManager(session_id=session_id)
        if state_manager.load_state():
            sessions[session_id] = state_manager
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    state_manager = sessions[session_id]
    structured_cv = state_manager.get_structured_cv()

    if not structured_cv:
        raise HTTPException(status_code=404, detail="CV data not found")

    # Convert to a simplified structure for the frontend
    sections = []

    for section in structured_cv.sections:
        section_data = {
            "id": section.id,
            "name": section.name,
            "content_type": section.content_type,
            "order": section.order,
            "items": [],
            "subsections": [],
        }

        # Add direct items
        for item in section.items:
            item_data = {
                "id": item.id,
                "content": item.content,
                "status": str(item.status),
                "item_type": str(item.item_type),
                "user_feedback": item.user_feedback,
            }
            section_data["items"].append(item_data)

        # Add subsections and their items
        for subsection in section.subsections:
            subsection_data = {
                "id": subsection.id,
                "name": subsection.name,
                "items": [],
            }

            for item in subsection.items:
                item_data = {
                    "id": item.id,
                    "content": item.content,
                    "status": str(item.status),
                    "item_type": str(item.item_type),
                    "user_feedback": item.user_feedback,
                }
                subsection_data["items"].append(item_data)

            section_data["subsections"].append(subsection_data)

        sections.append(section_data)

    return {"session_id": session_id, "sections": sections}


@app.post("/api/cv/regenerate")
async def regenerate_items(request: RegenerateRequest):
    """Regenerate specific items"""
    if request.session_id not in sessions:
        # Try to load from storage
        state_manager = StateManager(session_id=request.session_id)
        if state_manager.load_state():
            sessions[request.session_id] = state_manager
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    state_manager = sessions[request.session_id]
    structured_cv = state_manager.get_structured_cv()

    if not structured_cv:
        raise HTTPException(status_code=404, detail="CV data not found")

    try:
        # Call the enhanced content writer to regenerate specific items
        result = orchestrator.enhanced_content_writer.run(
            {
                "structured_cv": structured_cv,
                "regenerate_item_ids": request.item_ids,
                "job_description_data": parser_agent.get_job_data(),
                "research_results": (
                    orchestrator.research_agent.get_research_results()
                    if hasattr(orchestrator.research_agent, "get_research_results")
                    else {}
                ),
            }
        )

        # Update the state manager with the modified structured CV
        state_manager._structured_cv = result

        # Save state
        state_manager.save_state()

        return {"status": "success", "regenerated_items": request.item_ids}
    except Exception as e:
        logger.error(f"Error regenerating items: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cv/item/update")
async def update_item(request: ItemUpdateRequest):
    """Update a specific item's content or status"""
    if request.session_id not in sessions:
        # Try to load from storage
        state_manager = StateManager(session_id=request.session_id)
        if state_manager.load_state():
            sessions[request.session_id] = state_manager
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    state_manager = sessions[request.session_id]

    try:
        updated = False

        # Update content if provided
        if request.content is not None:
            if state_manager.update_item_content(request.item_id, request.content):
                updated = True

        # Update status if provided
        if request.status is not None:
            if state_manager.update_item_status(request.item_id, request.status):
                updated = True

        # Update feedback if provided
        if request.feedback is not None:
            item = state_manager.get_item(request.item_id)
            if item:
                item.user_feedback = request.feedback
                updated = True

        if updated:
            # Save state
            state_manager.save_state()
            return {"status": "success", "item_id": request.item_id}
        else:
            raise HTTPException(status_code=404, detail="Item not found or update failed")
    except Exception as e:
        logger.error(f"Error updating item: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cv/export")
async def export_cv(request: ExportRequest):
    """Export the CV in the specified format"""
    if request.session_id not in sessions:
        # Try to load from storage
        state_manager = StateManager(session_id=request.session_id)
        if state_manager.load_state():
            sessions[request.session_id] = state_manager
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    state_manager = sessions[request.session_id]
    structured_cv = state_manager.get_structured_cv()

    if not structured_cv:
        raise HTTPException(status_code=404, detail="CV data not found")

    try:
        # Convert to ContentData
        content_data = structured_cv.to_content_data()

        # Generate Markdown content (same as in the original main.py)
        rendered_cv = f"""
# {content_data.get('name', 'Your Name')}

## Contact Information
- Email: {content_data.get('email', '')}
- Phone: {content_data.get('phone', '')}
- LinkedIn: {content_data.get('linkedin', '')}
- GitHub: {content_data.get('github', '')}

## Executive Summary
{content_data.get('summary', '')}

## Key Qualifications
{content_data.get('skills_section', '')}

## Professional Experience
"""
        # Add experience bullets
        for bullet in content_data.get("experience_bullets", []):
            rendered_cv += f"- {bullet}\n"

        # Add projects
        rendered_cv += "\n## Project Experience\n"
        for project in content_data.get("projects", []):
            rendered_cv += f"### {project.get('name', 'Project')}\n"
            for bullet in project.get("bullets", []):
                rendered_cv += f"- {bullet}\n"

        # Add education
        rendered_cv += "\n## Education\n"
        for edu in content_data.get("education", []):
            rendered_cv += f"- {edu}\n"

        # Add certifications
        rendered_cv += "\n## Certifications\n"
        for cert in content_data.get("certifications", []):
            rendered_cv += f"- {cert}\n"

        # Add languages
        rendered_cv += "\n## Languages\n"
        for lang in content_data.get("languages", []):
            rendered_cv += f"- {lang}\n"

        # Save the file
        output_file = f"output_{request.session_id}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(rendered_cv)

        # For future: implement PDF generation with formatter_agent if format is "pdf"

        return {"status": "success", "content": rendered_cv, "format": request.format}
    except Exception as e:
        logger.error(f"Error exporting CV: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
