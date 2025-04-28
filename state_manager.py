from typing import TypedDict, List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field


from typing import List, TypedDict, Dict
from dataclasses import dataclass

class WorkflowStage(TypedDict):
    """
    Represents a stage in the workflow.
    """
    stage_name: str
    description: str
    is_completed: bool


class AgentIO(TypedDict):
    """
    Represents the input and output of an agent.
    """
    input: Dict[str, Any]  # Input data.
    output: Any  # Output data. 
    description: str  # Description of what the agent does.

# Ensure JobDescriptionData is the single source of truth
class JobDescriptionData:
    """
    Represents the parsed data of a job description.
    """
    def __init__(self, raw_text, skills, experience_level, responsibilities, industry_terms, company_values):
        self.raw_text = raw_text
        self.skills = skills
        self.experience_level = experience_level
        self.responsibilities = responsibilities
        self.industry_terms = industry_terms
        self.company_values = company_values

@dataclass
class VectorStoreConfig:
    dimension: int
    index_type: str #IndexFlatL2, IndexIVFFlat


@dataclass
class ContentPiece:
    """
    Represents a single piece of generated content (e.g., a bullet point, a summary paragraph).
    """
    content: str
    section_type: str # e.g., "summary", "experience", "skill", "project"
    piece_id: str # Unique identifier for the piece
    status: str = "pending_generation" # e.g., "pending_generation", "pending_review", "approved", "rejected"
    feedback: str = "" # Feedback from review
    revision_count: int = 0


class CVData(TypedDict): 
    """
    Represents the data extracted from the user's CV.
    Now includes fields for extracted raw text and lists of content pieces for different sections.
    """
    raw_text: str  # The original text of the CV
    summary: str # Original summary text from CV, if any
    experiences: List[str] # Original experiences list from CV, if any
    skills: List[str] # Original skills list from CV, if any
    education: List[str] # Original education list from CV, if any
    projects: List[str] # Original projects list from CV, if any


@dataclass
class SkillEntry:
    """
    Represents a skill entry.
    """
    text: str


@dataclass
class ExperienceEntry:
    """
    Represents an experience entry.
    """
    text: str

# ContentData will now represent the *iteratively built* content
# Instead of a single Dict output from ContentWriter, the state will manage 
# lists of ContentPiece objects for each section.
# We can keep ContentData for the final assembled content before rendering, 
# or adjust TemplateRenderer to work directly with lists of ContentPiece.
# Let's keep it for now, representing the *final* assembled generated content.
class ContentData(Dict):
    """
    Represents the assembled tailored CV content for rendering.
    """
    def __init__(self, summary="", experience_bullets=None, skills_section="", projects=None, other_content=None):
        super().__init__()
        self["summary"] = summary
        self["experience_bullets"] = experience_bullets if experience_bullets is not None else []
        self["skills_section"] = skills_section
        self["projects"] = projects if projects is not None else []
        self["other_content"] = other_content if other_content is not None else {}

    @property
    def summary(self):
        return self.get("summary")

    @property
    def experience_bullets(self):
        return self.get("experience_bullets")

    @property
    def skills_section(self):
        return self.get("skills_section")

    @property
    def projects(self):
        return self.get("projects")

    @property
    def other_content(self):
        return self.get("other_content")


class WorkflowState(TypedDict):
    """
    Represents the overall state of the CV tailoring workflow.
    Now tracks content pieces and their review status for iterative refinement.
    """
    job_description: Dict # Parsed job description data
    user_cv: CVData # Extracted user CV data
    extracted_skills: Dict # Deprecate or refine - skills are now in user_cv
    content_pieces: Dict[str, List[ContentPiece]] # Dictionary holding lists of ContentPiece by section type
    current_content_piece_id: Optional[str] # ID of the content piece currently being processed/reviewed
    content_generation_plan: List[Dict[str, Any]] # A plan of content pieces to generate/review
    plan_index: int # To track progress through the generation plan
    formatted_cv_text: str # Final formatted text before rendering
    rendered_cv: str # Final rendered output
    feedback: List[Dict[str, Any]] # List of feedback entries for transparency
    revision_history: List[Dict[str, Any]] # List of revision history entries
    current_stage: WorkflowStage # Current stage of the overall workflow
    workflow_id: str # Workflow ID
    relevant_experiences: List[str] # Relevant experiences from vector search
    research_results: Dict[str, Any] # Research results
    review_status: str  # e.g., "pending", "approved", "rejected"
    review_feedback: str # Store feedback comments for the current piece
