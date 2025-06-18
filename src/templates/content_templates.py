"""Content templates for CV generation with Phase 1 infrastructure integration."""

from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from ..models.data_models import ContentType
from ..config.logging_config import get_structured_logger

logger = get_structured_logger("content_templates")


class TemplateCategory(Enum):
    """Categories of content templates."""

    PROMPT = "prompt"
    FORMAT = "format"
    VALIDATION = "validation"
    FALLBACK = "fallback"


@dataclass
class ContentTemplate:
    """Structured content template."""

    name: str
    category: TemplateCategory
    content_type: ContentType
    template: str
    variables: List[str]
    description: str
    version: str = "1.0"
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ContentTemplateManager:
    """Manager for content templates with caching and validation."""

    def __init__(self):
        self.templates: Dict[str, ContentTemplate] = {}
        self._load_default_templates()

        logger.info(
            "Content template manager initialized", template_count=len(self.templates)
        )

    def _load_default_templates(self):
        """Load default templates for all content types."""

        # Qualification templates
        self._register_qualification_templates()

        # Experience templates
        self._register_experience_templates()

        # Project templates
        self._register_project_templates()

        # Executive summary templates
        self._register_executive_summary_templates()

        # Professional summary templates removed - using executive summary only

        # Format templates
        self._register_format_templates()

        # Validation templates
        self._register_validation_templates()

        # Fallback templates
        self._register_fallback_templates()

    def _register_qualification_templates(self):
        """Register qualification-specific templates."""

        # Basic qualification prompt
        self.register_template(
            ContentTemplate(
                name="qualification_basic",
                category=TemplateCategory.PROMPT,
                content_type=ContentType.QUALIFICATION,
                template="""
Generate a professional qualification statement for a {job_title} position at {company_name}.

Job Requirements:
{job_description}

Current Qualification:
{qualification_title}: {qualification_description}

Additional Context:
{additional_context}

Generate a concise, impactful qualification statement (2-3 sentences) that:
1. Highlights relevant skills and experience
2. Aligns with the job requirements
3. Uses professional language
4. Demonstrates value to the employer
5. Avoids generic statements

Qualification Statement:""",
                variables=[
                    "job_title",
                    "company_name",
                    "job_description",
                    "qualification_title",
                    "qualification_description",
                    "additional_context",
                ],
                description="Basic template for generating qualification statements",
            )
        )

        # Technical qualification prompt
        self.register_template(
            ContentTemplate(
                name="qualification_technical",
                category=TemplateCategory.PROMPT,
                content_type=ContentType.QUALIFICATION,
                template="""
Generate a technical qualification statement for a {job_title} position.

Technical Requirements:
{technical_requirements}

Current Technical Skills:
{technical_skills}

Experience Level: {experience_level}
Specialization: {specialization}

Generate a technical qualification statement that:
1. Emphasizes relevant technical expertise
2. Mentions specific technologies and tools
3. Quantifies experience where possible
4. Aligns with job technical requirements
5. Uses industry-standard terminology

Technical Qualification:""",
                variables=[
                    "job_title",
                    "technical_requirements",
                    "technical_skills",
                    "experience_level",
                    "specialization",
                ],
                description="Template for technical qualification statements",
            )
        )

        # Leadership qualification prompt
        self.register_template(
            ContentTemplate(
                name="qualification_leadership",
                category=TemplateCategory.PROMPT,
                content_type=ContentType.QUALIFICATION,
                template="""
Generate a leadership qualification statement for a {job_title} position.

Leadership Requirements:
{leadership_requirements}

Leadership Experience:
{leadership_experience}

Team Size Managed: {team_size}
Management Style: {management_style}

Generate a leadership qualification statement that:
1. Highlights leadership achievements
2. Demonstrates team management capabilities
3. Shows impact on organizational goals
4. Emphasizes communication and mentoring skills
5. Aligns with the leadership requirements

Leadership Qualification:""",
                variables=[
                    "job_title",
                    "leadership_requirements",
                    "leadership_experience",
                    "team_size",
                    "management_style",
                ],
                description="Template for leadership qualification statements",
            )
        )

    def _register_experience_templates(self):
        """Register experience-specific templates."""

        # Basic experience prompt
        self.register_template(
            ContentTemplate(
                name="experience_basic",
                category=TemplateCategory.PROMPT,
                content_type=ContentType.EXPERIENCE,
                template="""
Enhance the following work experience for a {job_title} position at {company_name}.

Job Requirements:
{job_description}

Current Experience:
Position: {position}
Company: {current_company}
Duration: {duration}
Description: {experience_description}

Key Achievements: {achievements}
Skills Used: {skills_used}

Generate enhanced experience content that:
1. Emphasizes achievements and quantifiable results
2. Uses strong action verbs
3. Aligns with target job requirements
4. Demonstrates progression and growth
5. Includes 3-5 impactful bullet points

Enhanced Experience:""",
                variables=[
                    "job_title",
                    "company_name",
                    "job_description",
                    "position",
                    "current_company",
                    "duration",
                    "experience_description",
                    "achievements",
                    "skills_used",
                ],
                description="Basic template for enhancing work experience",
            )
        )

        # Senior-level experience prompt
        self.register_template(
            ContentTemplate(
                name="experience_senior",
                category=TemplateCategory.PROMPT,
                content_type=ContentType.EXPERIENCE,
                template="""
Enhance senior-level experience for a {job_title} position.

Senior Role Requirements:
{senior_requirements}

Current Senior Experience:
Position: {position}
Company: {current_company}
Team Size: {team_size}
Budget Managed: {budget_managed}
Key Initiatives: {key_initiatives}
Strategic Impact: {strategic_impact}

Generate senior-level experience content that:
1. Emphasizes strategic thinking and leadership
2. Quantifies business impact and ROI
3. Demonstrates cross-functional collaboration
4. Shows mentoring and team development
5. Highlights innovation and process improvement

Senior Experience:""",
                variables=[
                    "job_title",
                    "senior_requirements",
                    "position",
                    "current_company",
                    "team_size",
                    "budget_managed",
                    "key_initiatives",
                    "strategic_impact",
                ],
                description="Template for senior-level experience enhancement",
            )
        )

    def _register_project_templates(self):
        """Register project-specific templates."""

        # Technical project prompt
        self.register_template(
            ContentTemplate(
                name="project_technical",
                category=TemplateCategory.PROMPT,
                content_type=ContentType.PROJECT,
                template="""
Enhance the following technical project for a {job_title} position.

Job Technical Requirements:
{technical_requirements}

Current Project:
Project Name: {project_name}
Description: {project_description}
Technologies Used: {technologies}
Team Size: {team_size}
Duration: {duration}
Role: {role}

Key Challenges: {challenges}
Solutions Implemented: {solutions}
Results Achieved: {results}

Generate enhanced project content that:
1. Highlights technical complexity and innovation
2. Emphasizes problem-solving capabilities
3. Quantifies project impact and success metrics
4. Demonstrates relevant technology expertise
5. Shows collaboration and technical leadership

Enhanced Project:""",
                variables=[
                    "job_title",
                    "technical_requirements",
                    "project_name",
                    "project_description",
                    "technologies",
                    "team_size",
                    "duration",
                    "role",
                    "challenges",
                    "solutions",
                    "results",
                ],
                description="Template for technical project enhancement",
            )
        )

        # Business project prompt
        self.register_template(
            ContentTemplate(
                name="project_business",
                category=TemplateCategory.PROMPT,
                content_type=ContentType.PROJECT,
                template="""
Enhance the following business project for a {job_title} position.

Business Requirements:
{business_requirements}

Current Project:
Project Name: {project_name}
Objective: {project_objective}
Stakeholders: {stakeholders}
Budget: {budget}
Timeline: {timeline}
Role: {role}

Business Impact: {business_impact}
KPIs Achieved: {kpis}
Lessons Learned: {lessons_learned}

Generate enhanced project content that:
1. Emphasizes business value and ROI
2. Demonstrates stakeholder management
3. Quantifies business impact and metrics
4. Shows strategic thinking and execution
5. Highlights cross-functional collaboration

Enhanced Project:""",
                variables=[
                    "job_title",
                    "business_requirements",
                    "project_name",
                    "project_objective",
                    "stakeholders",
                    "budget",
                    "timeline",
                    "role",
                    "business_impact",
                    "kpis",
                    "lessons_learned",
                ],
                description="Template for business project enhancement",
            )
        )

    def _register_executive_summary_templates(self):
        """Register executive summary templates."""

        # Professional executive summary
        self.register_template(
            ContentTemplate(
                name="executive_summary_professional",
                category=TemplateCategory.PROMPT,
                content_type=ContentType.EXECUTIVE_SUMMARY,
                template="""
Generate a compelling executive summary for a {job_title} position at {company_name}.

Job Requirements:
{job_description}

Candidate Profile:
Years of Experience: {years_experience}
Industry Background: {industry_background}
Core Competencies: {core_competencies}
Key Achievements: {key_achievements}
Education: {education}
Certifications: {certifications}

Career Highlights:
{career_highlights}

Generate a professional executive summary (3-4 sentences) that:
1. Captures unique value proposition
2. Aligns with target role requirements
3. Highlights quantifiable achievements
4. Demonstrates industry expertise
5. Uses confident, professional language
6. Focuses on what you can deliver for the employer

Executive Summary:""",
                variables=[
                    "job_title",
                    "company_name",
                    "job_description",
                    "years_experience",
                    "industry_background",
                    "core_competencies",
                    "key_achievements",
                    "education",
                    "certifications",
                    "career_highlights",
                ],
                description="Professional executive summary template",
            )
        )

        # Career transition executive summary
        self.register_template(
            ContentTemplate(
                name="executive_summary_transition",
                category=TemplateCategory.PROMPT,
                content_type=ContentType.EXECUTIVE_SUMMARY,
                template="""
Generate an executive summary for a career transition to {job_title}.

Target Role:
{target_role_description}

Current Background:
Current Industry: {current_industry}
Transferable Skills: {transferable_skills}
Relevant Experience: {relevant_experience}
Education/Training: {education_training}
Motivation for Change: {motivation}

Bridge Elements:
{bridge_elements}

Generate a transition-focused executive summary that:
1. Emphasizes transferable skills and experience
2. Addresses potential concerns about career change
3. Demonstrates passion and commitment to new field
4. Highlights relevant achievements and learning
5. Shows clear understanding of target role

Transition Executive Summary:""",
                variables=[
                    "job_title",
                    "target_role_description",
                    "current_industry",
                    "transferable_skills",
                    "relevant_experience",
                    "education_training",
                    "motivation",
                    "bridge_elements",
                ],
                description="Executive summary template for career transitions",
            )
        )

    # _register_professional_summary_templates method removed - using executive summary only

    def _register_format_templates(self):
        """Register formatting templates."""

        # Bullet point format
        self.register_template(
            ContentTemplate(
                name="format_bullet_points",
                category=TemplateCategory.FORMAT,
                content_type=ContentType.EXPERIENCE,
                template="""
• {achievement_1}
• {achievement_2}
• {achievement_3}
• {achievement_4}
• {achievement_5}""",
                variables=[
                    "achievement_1",
                    "achievement_2",
                    "achievement_3",
                    "achievement_4",
                    "achievement_5",
                ],
                description="Standard bullet point format for achievements",
            )
        )

        # Paragraph format
        self.register_template(
            ContentTemplate(
                name="format_paragraph",
                category=TemplateCategory.FORMAT,
                content_type=ContentType.EXECUTIVE_SUMMARY,
                template="{opening_statement} {experience_highlight} {key_strengths} {value_proposition}",
                variables=[
                    "opening_statement",
                    "experience_highlight",
                    "key_strengths",
                    "value_proposition",
                ],
                description="Paragraph format for executive summaries",
            )
        )

    def _register_validation_templates(self):
        """Register validation templates."""

        # Content quality validation
        self.register_template(
            ContentTemplate(
                name="validation_quality",
                category=TemplateCategory.VALIDATION,
                content_type=ContentType.QUALIFICATION,
                template="""
Validate the following content for quality and professionalism:

Content: {content}
Content Type: {content_type}
Target Role: {target_role}

Check for:
1. Professional language and tone
2. Clarity and conciseness
3. Relevance to target role
4. Quantifiable achievements
5. Grammar and spelling
6. Appropriate length

Provide validation results with:
- Overall score (0-10)
- Specific issues found
- Improvement suggestions
- Approval status (approved/needs_revision)

Validation Results:""",
                variables=["content", "content_type", "target_role"],
                description="Template for content quality validation",
            )
        )

    def _register_fallback_templates(self):
        """Register fallback templates."""

        # Generic fallback content
        fallback_content = {
            ContentType.QUALIFICATION: "Strong professional background with relevant skills and experience that align with the position requirements.",
            ContentType.EXPERIENCE: "Valuable professional experience contributing to organizational success and demonstrating growth in responsibilities and achievements.",
            ContentType.PROJECT: "Successfully completed project demonstrating technical capabilities, problem-solving skills, and ability to deliver results.",
            ContentType.EXECUTIVE_SUMMARY: "Experienced professional with a proven track record of success and a strong commitment to delivering exceptional results in challenging environments.",
        }

        for content_type, content in fallback_content.items():
            self.register_template(
                ContentTemplate(
                    name=f"fallback_{content_type.value}",
                    category=TemplateCategory.FALLBACK,
                    content_type=content_type,
                    template=content,
                    variables=[],
                    description=f"Fallback content for {content_type.value}",
                )
            )

    def register_template(self, template: ContentTemplate):
        """Register a new template."""
        template_key = (
            f"{template.content_type.value}_{template.category.value}_{template.name}"
        )
        self.templates[template_key] = template

        logger.debug(
            "Template registered",
            template_name=template.name,
            content_type=template.content_type.value,
            category=template.category.value,
        )

    def get_template(
        self,
        name: str,
        content_type: ContentType,
        category: TemplateCategory = TemplateCategory.PROMPT,
    ) -> Optional[ContentTemplate]:
        """Get a specific template."""
        template_key = f"{content_type.value}_{category.value}_{name}"
        return self.templates.get(template_key)

    def get_templates_by_type(
        self, content_type: ContentType, category: Optional[TemplateCategory] = None
    ) -> List[ContentTemplate]:
        """Get all templates for a specific content type."""
        templates = []
        for template in self.templates.values():
            if template.content_type == content_type:
                if category is None or template.category == category:
                    templates.append(template)
        return templates

    def format_template(
        self, template: ContentTemplate, variables: Dict[str, Any]
    ) -> str:
        """Format a template with provided variables."""
        try:
            # Ensure all required variables are provided
            missing_vars = [var for var in template.variables if var not in variables]
            if missing_vars:
                logger.warning(
                    "Missing template variables",
                    template_name=template.name,
                    missing_variables=missing_vars,
                )
                # Provide default values for missing variables
                for var in missing_vars:
                    variables[var] = f"[{var}]"

            # Format the template
            formatted_content = template.template.format(**variables)

            logger.debug(
                "Template formatted successfully",
                template_name=template.name,
                content_length=len(formatted_content),
            )

            return formatted_content

        except Exception as e:
            logger.error(
                "Template formatting failed", template_name=template.name, error=str(e)
            )
            return template.template  # Return unformatted template as fallback

    def get_fallback_content(self, content_type: ContentType) -> str:
        """Get fallback content for a content type."""
        template = self.get_template(
            f"fallback_{content_type.value}", content_type, TemplateCategory.FALLBACK
        )

        if template:
            return template.template

        # Ultimate fallback
        return "Professional experience and qualifications relevant to the position."

    def list_templates(self) -> Dict[str, List[str]]:
        """List all available templates by category."""
        template_list = {}

        for template in self.templates.values():
            category_key = f"{template.content_type.value}_{template.category.value}"
            if category_key not in template_list:
                template_list[category_key] = []
            template_list[category_key].append(template.name)

        return template_list


# Global template manager instance
_template_manager = None


def get_template_manager() -> ContentTemplateManager:
    """Get the global template manager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = ContentTemplateManager()
    return _template_manager


# Convenience functions
def get_prompt_template(
    content_type: ContentType, template_name: str = "basic"
) -> Optional[ContentTemplate]:
    """Get a prompt template for a content type."""
    manager = get_template_manager()
    return manager.get_template(
        f"{content_type.value}_{template_name}", content_type, TemplateCategory.PROMPT
    )


def format_content_prompt(
    content_type: ContentType, template_name: str, variables: Dict[str, Any]
) -> str:
    """Format a content prompt with variables."""
    manager = get_template_manager()
    template = manager.get_template(
        f"{content_type.value}_{template_name}", content_type, TemplateCategory.PROMPT
    )

    if template:
        return manager.format_template(template, variables)

    # Fallback to basic template
    basic_template = manager.get_template(
        f"{content_type.value}_basic", content_type, TemplateCategory.PROMPT
    )

    if basic_template:
        return manager.format_template(basic_template, variables)

    return f"Generate professional {content_type.value} content."


def get_fallback_content(content_type: ContentType) -> str:
    """Get fallback content for a content type."""
    manager = get_template_manager()
    return manager.get_fallback_content(content_type)
