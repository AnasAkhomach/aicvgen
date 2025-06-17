from src.agents.agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from src.services.llm_service import get_llm_service
from src.services.vector_db import get_enhanced_vector_db
from .agent_base import EnhancedAgentBase
from src.models.data_models import (
    JobDescriptionData,
    StructuredCV,
    Section,
    Subsection,
    Item,
    ItemStatus,
    ItemType,
    AgentIO,
    PersonalInfo,
    Experience,
    Education,
    Skill,
    Project,
    Certification,
    Language,
)
from src.config.logging_config import get_structured_logger
from src.models.data_models import AgentDecisionLog, AgentExecutionLog
from src.config.settings import get_config
from src.services.llm import LLMResponse
from src.orchestration.state import AgentState
from src.core.async_optimizer import optimize_async
from src.utils.exceptions import (
    LLMResponseParsingError,
    ValidationError,
    WorkflowPreconditionError,
    AgentExecutionError,
    ConfigurationError,
    StateManagerError,
)
import json  # Import json for parsing LLM output
from typing import List, Optional, Dict, Any, Union
import re  # For regex parsing of Markdown
import logging  # For logging
import asyncio

# Set up structured logging
logger = get_structured_logger(__name__)


class ParserAgent(EnhancedAgentBase):
    """Agent responsible for parsing job descriptions and extracting key information, and parsing CVs into StructuredCV objects."""

    def __init__(self, name: str, description: str, llm_service=None, llm=None):
        """
        Initialize the ParserAgent.

        Args:
            name: The name of the agent.
            description: A description of the agent's purpose.
            llm_service: Optional LLM service instance. If None, will use get_llm_service().
            llm: Alternative parameter name for LLM service (for backward compatibility).
        """
        # Define input and output schemas
        input_schema = AgentIO(
            description="Input schema for parsing job descriptions and CV text",
            required_fields=["raw_text"],
            optional_fields=["metadata"],
        )
        output_schema = AgentIO(
            description="Output schema for structured parsing results",
            required_fields=["parsed_data"],
            optional_fields=["confidence_score", "error_message"],
        )

        # Call parent constructor
        super().__init__(name, description, input_schema, output_schema)

        # Accept either llm_service or llm parameter
        self.llm = llm_service or llm or get_llm_service()

        # Initialize settings for prompt loading
        self.settings = get_config()

    async def parse_job_description(
        self, raw_text: str, trace_id: Optional[str] = None
    ) -> JobDescriptionData:
        """
        Parses a raw job description using an LLM and extracts key information.
        Uses robust JSON validation with Pydantic models.

        Args:
            raw_text: The raw job description as a string.

        Returns:
            A JobDescriptionData object with the parsed content.
        """
        if not raw_text:
            # Log validation decision with structured logging
            self.log_decision(
                "Empty job description provided, returning default structure",
                None,
                "validation"
            )

            return JobDescriptionData(
                raw_text=raw_text,
                skills=[],
                responsibilities=[],
                company_values=[],
                industry_terms=[],
                experience_level="N/A",
            )

        return await self._parse_job_description_with_llm(raw_text, trace_id=trace_id)

    async def _parse_job_description_with_llm(
        self, raw_text: str, trace_id: Optional[str] = None
    ) -> JobDescriptionData:
        """
        Internal method to parse job description using LLM.

        Args:
            raw_text: The raw job description as a string.

        Returns:
            A JobDescriptionData object with the parsed content.
        """
        # Load the updated prompt
        prompt_path = self.settings.get_prompt_path_by_key("job_description_parser")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        prompt = prompt_template.format(raw_text=raw_text)

        try:
            # 1. Get response from LLM
            response = await self.llm.generate_content(prompt, trace_id=trace_id)
            raw_response_content = response.content

            # 2. Extract the JSON block from the raw response
            # A simple but effective way to handle markdown code blocks or other noise
            json_start = raw_response_content.find("{")
            json_end = raw_response_content.rfind("}") + 1
            if json_start == -1 or json_end <= json_start:
                raise LLMResponseParsingError(
                    "No valid JSON object found in LLM response.",
                    raw_response=raw_response_content,
                )

            json_str = raw_response_content[json_start:json_end]

            # 3. Parse the JSON string
            parsed_data = json.loads(json_str)

            # 4. Validate with Pydantic
            from src.models.validation_schemas import LLMJobDescriptionOutput

            validated_output = LLMJobDescriptionOutput.model_validate(parsed_data)

            # 5. Map validated data to the application's main data model
            job_data = JobDescriptionData(
                raw_text=raw_text,
                skills=validated_output.skills,
                experience_level=validated_output.experience_level,
                responsibilities=validated_output.responsibilities,
                industry_terms=validated_output.industry_terms,
                company_values=validated_output.company_values,
                status=ItemStatus.GENERATED,
            )
            logger.info(
                "Job description successfully parsed and validated using LLM-generated JSON."
            )
            return job_data

        except (json.JSONDecodeError, ValidationError, LLMResponseParsingError) as e:
            error_message = f"Failed to parse or validate LLM response for job description: {str(e)}"
            logger.error(error_message, exc_info=True)
            # Create a failed state object
            return JobDescriptionData(
                raw_text=raw_text,
                skills=[],
                responsibilities=[],
                company_values=[],
                industry_terms=[],
                experience_level="N/A",
                error=error_message,
                status=ItemStatus.GENERATION_FAILED,
            )

    def _parse_job_description_with_regex(self, raw_text: str) -> JobDescriptionData:
        """
        Fallback method to parse job description using regex patterns.

        Args:
            raw_text: The raw job description text

        Returns:
            JobDescriptionData object with extracted information
        """
        logger.info("Using regex-based fallback parsing for job description")

        # Initialize lists for extracted data
        skills = []
        responsibilities = []
        industry_terms = []
        company_values = []
        experience_level = "N/A"

        # Convert to lowercase for pattern matching
        text_lower = raw_text.lower()

        # Extract experience level using common patterns
        exp_patterns = [
            (
                r"(\d+)\+?\s*years?\s*(?:of\s*)?experience",
                lambda m: f"{m.group(1)}+ years",
            ),
            (r"entry\s*level", lambda m: "Entry Level"),
            (r"junior", lambda m: "Junior"),
            (r"senior", lambda m: "Senior"),
            (r"mid\s*level", lambda m: "Mid Level"),
            (r"lead", lambda m: "Lead"),
            (r"principal", lambda m: "Principal"),
        ]

        for pattern, formatter in exp_patterns:
            match = re.search(pattern, text_lower)
            if match:
                experience_level = formatter(match)
                break

        # Extract skills using common technology and skill keywords
        skill_keywords = [
            "python",
            "java",
            "javascript",
            "typescript",
            "react",
            "angular",
            "vue",
            "node.js",
            "express",
            "django",
            "flask",
            "spring",
            "hibernate",
            "sql",
            "mysql",
            "postgresql",
            "mongodb",
            "redis",
            "elasticsearch",
            "aws",
            "azure",
            "gcp",
            "docker",
            "kubernetes",
            "jenkins",
            "git",
            "agile",
            "scrum",
            "devops",
            "ci/cd",
            "testing",
            "debugging",
            "html",
            "css",
            "bootstrap",
            "sass",
            "webpack",
            "npm",
            "yarn",
            "rest",
            "api",
            "microservices",
            "graphql",
            "oauth",
            "jwt",
            "machine learning",
            "ai",
            "data science",
            "analytics",
            "tableau",
            "communication",
            "teamwork",
            "leadership",
            "problem solving",
        ]

        for keyword in skill_keywords:
            if keyword in text_lower:
                skills.append(keyword.title())

        # Extract responsibilities using bullet points and common action verbs
        responsibility_patterns = [
            r"[•\*\-]\s*([^\n]+)",  # Bullet points
            r"(?:develop|design|implement|maintain|create|build|manage|lead|coordinate)\s+([^\n\.]+)",
            r"responsible for\s+([^\n\.]+)",
            r"duties include\s+([^\n\.]+)",
        ]

        for pattern in responsibility_patterns:
            matches = re.findall(pattern, raw_text, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 10:  # Filter out very short matches
                    responsibilities.append(match.strip())

        # Extract industry terms using common business and tech keywords
        industry_keywords = [
            "fintech",
            "healthcare",
            "e-commerce",
            "saas",
            "b2b",
            "b2c",
            "startup",
            "enterprise",
            "digital transformation",
            "innovation",
            "scalability",
            "performance",
            "security",
            "compliance",
            "gdpr",
            "automation",
            "optimization",
            "integration",
            "migration",
            "cloud",
            "on-premise",
            "hybrid",
            "distributed",
            "real-time",
        ]

        for keyword in industry_keywords:
            if keyword in text_lower:
                industry_terms.append(keyword.title())

        # Extract company values using common value keywords
        value_keywords = [
            "innovation",
            "collaboration",
            "integrity",
            "excellence",
            "diversity",
            "inclusion",
            "transparency",
            "accountability",
            "customer-focused",
            "quality",
            "continuous learning",
            "growth mindset",
            "teamwork",
        ]

        for keyword in value_keywords:
            if keyword in text_lower:
                company_values.append(keyword.title())

        # Remove duplicates and limit results
        skills = list(set(skills))[:15]  # Limit to 15 skills
        responsibilities = list(set(responsibilities))[
            :10
        ]  # Limit to 10 responsibilities
        industry_terms = list(set(industry_terms))[:8]  # Limit to 8 terms
        company_values = list(set(company_values))[:6]  # Limit to 6 values

        return JobDescriptionData(
            raw_text=raw_text,
            skills=skills,
            experience_level=experience_level,
            responsibilities=responsibilities,
            industry_terms=industry_terms,
            company_values=company_values,
            error=None,  # No error for successful regex parsing
        )

    def parse_cv_text(self, cv_text: str, job_data: JobDescriptionData) -> StructuredCV:
        """
        Parses CV text into a StructuredCV object.

        Args:
            cv_text: The raw CV text as a string.
            job_data: The parsed job description data.

        Returns:
            A StructuredCV object representing the parsed CV.
        """
        # Create a new StructuredCV
        structured_cv = StructuredCV()

        # Add metadata - handle both dict and JobDescriptionData object types
        if job_data:
            if hasattr(job_data, "to_dict"):
                # JobDescriptionData object
                structured_cv.metadata["job_description"] = job_data.to_dict()
            elif isinstance(job_data, dict):
                # Already a dictionary
                structured_cv.metadata["job_description"] = job_data
            else:
                # Fallback for other types
                structured_cv.metadata["job_description"] = {}
        else:
            structured_cv.metadata["job_description"] = {}
        structured_cv.metadata["original_cv_text"] = cv_text

        # If no CV text, return empty structure
        if not cv_text:
            logger.warning("Empty CV text provided to ParserAgent.")
            return structured_cv

        # Process CV text line by line to extract sections
        lines = cv_text.split("\n")
        current_section = None
        current_subsection = None

        # Regular expressions for Markdown patterns - Made more flexible to capture variations
        section_pattern = re.compile(
            r"^#{1,3}\s+\**([^*#]+)\**$"
        )  # Matches ###, ##, or # headings with optional bold formatting
        subsection_pattern = re.compile(
            r"^#{3,4}\s+\**([^*#]+)\**$"
        )  # Matches #### or ### headings for subsections
        contact_pattern = re.compile(
            r"\*\*([^*]+)\*\*\s*\|\s*(.+)"
        )  # **Name** | Contact Info
        bullet_pattern = re.compile(r"^\s*[\*\-]\s+(.+)$")  # * Bullet or - Bullet

        # Function to normalize section names by removing formatting and standardizing
        def normalize_section_name(name):
            # Remove any Markdown formatting (**, __, etc.)
            name = re.sub(r"[*_]", "", name).strip()

            # Remove trailing colon if present
            if name.endswith(":"):
                name = name[:-1].strip()

            # Map common variations to standard names
            section_map = {
                "summary": "Executive Summary",
                "profile": "Executive Summary",
                "overview": "Executive Summary",
                "about me": "Executive Summary",
                "professional profile": "Executive Summary",
                "professional summary": "Executive Summary",
                "executive summary": "Executive Summary",
                "skills": "Key Qualifications",
                "key skills": "Key Qualifications",
                "skill set": "Key Qualifications",
                "core competencies": "Key Qualifications",
                "technical skills": "Key Qualifications",
                "key qualifications": "Key Qualifications",
                "experience": "Professional Experience",
                "work experience": "Professional Experience",
                "employment history": "Professional Experience",
                "work history": "Professional Experience",
                "professional experience": "Professional Experience",
                "projects": "Project Experience",
                "personal projects": "Project Experience",
                "project experience": "Project Experience",
                "education": "Education",
                "academic background": "Education",
                "certifications": "Certifications",
                "certificates": "Certifications",
                "credentials": "Certifications",
                "languages": "Languages",
                "language proficiency": "Languages",
            }

            # Check for case-insensitive match in our map
            for key, value in section_map.items():
                if key.lower() == name.lower():
                    logger.info(f"Normalized section name: '{name}' -> '{value}'")
                    return value

            # If no match found, return the original name
            return name

        # Extract contact info from first few lines
        for i, line in enumerate(
            lines[:10]
        ):  # Check only first 10 lines for contact info
            contact_match = contact_pattern.match(line)
            if contact_match:
                name = contact_match.group(1).strip()
                contact_info = contact_match.group(2).strip()
                structured_cv.metadata["name"] = name

                # Extract email, phone, LinkedIn, GitHub from contact info
                email_match = re.search(
                    r"\[([^\]]+@[^\]]+)\]\(mailto:[^\)]+\)", contact_info
                )
                if email_match:
                    structured_cv.metadata["email"] = email_match.group(1)

                phone_match = re.search(r"📞\s*([^|]+)", contact_info)
                if phone_match:
                    structured_cv.metadata["phone"] = phone_match.group(1).strip()

                linkedin_match = re.search(r"\[LinkedIn\]\(([^\)]+)\)", contact_info)
                if linkedin_match:
                    structured_cv.metadata["linkedin"] = linkedin_match.group(1)

                github_match = re.search(r"\[GitHub\]\(([^\)]+)\)", contact_info)
                if github_match:
                    structured_cv.metadata["github"] = github_match.group(1)

                break

        # Process the rest of the document to extract sections, subsections, and items
        section_order = 0
        for line in lines:
            # Check for section headings
            section_match = section_pattern.match(line)
            if section_match:
                raw_section_name = section_match.group(1).strip()
                section_name = normalize_section_name(raw_section_name)

                # Determine if this is a dynamic section (to be tailored) or static section
                dynamic_sections = [
                    "Executive Summary",
                    "Key Qualifications",
                    "Professional Experience",
                    "Project Experience",
                ]
                content_type = (
                    "DYNAMIC"
                    if any(section_name.lower() == s.lower() for s in dynamic_sections)
                    else "STATIC"
                )

                logger.info(
                    f"Parsed section: '{raw_section_name}' -> '{section_name}' (Type: {content_type})"
                )

                current_section = Section(
                    name=section_name,
                    content_type=content_type,
                    order=section_order,
                    raw_text=line,
                )
                section_order += 1
                structured_cv.sections.append(current_section)
                current_subsection = None
                continue

            # Check for subsection headings
            subsection_match = subsection_pattern.match(line)
            if subsection_match and current_section:
                subsection_name = subsection_match.group(1).strip()
                current_subsection = Subsection(name=subsection_name, raw_text=line)
                current_section.subsections.append(current_subsection)
                continue

            # Check for bullet points
            bullet_match = bullet_pattern.match(line)
            if bullet_match and (current_section or current_subsection):
                content = bullet_match.group(1).strip()

                # Determine item type based on current section
                item_type = ItemType.BULLET_POINT
                if current_section and current_section.name == "Key Qualifications":
                    item_type = ItemType.KEY_QUALIFICATION
                elif current_section and current_section.name == "Executive Summary":
                    item_type = ItemType.EXECUTIVE_SUMMARY_PARA
                elif current_section and current_section.name == "Education":
                    item_type = ItemType.EDUCATION_ENTRY
                elif current_section and current_section.name == "Certifications":
                    item_type = ItemType.CERTIFICATION_ENTRY
                elif current_section and current_section.name == "Languages":
                    item_type = ItemType.LANGUAGE_ENTRY

                # Determine status (dynamic sections start as INITIAL, static as STATIC)
                status = (
                    ItemStatus.INITIAL
                    if (current_section and current_section.content_type == "DYNAMIC")
                    else ItemStatus.STATIC
                )

                item = Item(content=content, status=status, item_type=item_type)

                # Add to subsection if we're in one, otherwise directly to section
                if current_subsection:
                    current_subsection.items.append(item)
                elif current_section:
                    current_section.items.append(item)

            # Handle non-bullet content for sections like Executive Summary
            elif (
                line.strip()
                and current_section
                and not current_subsection
                and current_section.name == "Executive Summary"
            ):
                # For Executive Summary, we'll consider non-bullet content as summary paragraphs
                if not line.startswith("---") and not line.startswith("#"):
                    item = Item(
                        content=line.strip(),
                        status=(
                            ItemStatus.INITIAL
                            if current_section.content_type == "DYNAMIC"
                            else ItemStatus.STATIC
                        ),
                        item_type=ItemType.SUMMARY_PARAGRAPH,
                    )
                    current_section.items.append(item)

        # Add special handling for Key Qualifications section if it contains a list with | separators
        key_quals_section = next(
            (
                section
                for section in structured_cv.sections
                if section.name == "Key Qualifications"
            ),
            None,
        )
        if key_quals_section and key_quals_section.items:
            # Check if the section contains a single line with | separators
            for i, item in enumerate(key_quals_section.items):
                if "|" in item.content:
                    # Split the content by the | character
                    qual_items = [qual.strip() for qual in item.content.split("|")]
                    # Remove the original item
                    key_quals_section.items.pop(i)
                    # Add each qualification as a separate item
                    for qual in qual_items:
                        if qual:  # Skip empty qualifications
                            key_quals_section.items.append(
                                Item(
                                    content=qual,
                                    status=ItemStatus.INITIAL,
                                    item_type=ItemType.KEY_QUALIFICATION,
                                )
                            )

        # Apply LLM-powered parsing for specific sections
        import asyncio

        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
            # If we're in an async context, we need to handle this differently
            # For now, we'll skip the LLM enhancement in sync context
            logger.warning("Skipping LLM enhancement in synchronous context")
        except RuntimeError:
            # No event loop running, we can create one
            asyncio.run(self._enhance_sections_with_llm(structured_cv, cv_text))

        return structured_cv

    async def _enhance_sections_with_llm(
        self, structured_cv: StructuredCV, cv_text: str
    ) -> None:
        """
        Enhance specific sections using LLM-powered parsing.

        Args:
            structured_cv: The structured CV to enhance
            cv_text: The original CV text for context
        """
        # Find and enhance Professional Experience section
        experience_section = next(
            (
                section
                for section in structured_cv.sections
                if section.name == "Professional Experience"
            ),
            None,
        )
        if experience_section:
            enhanced_experience = await self._parse_experience_section_with_llm(cv_text)
            if enhanced_experience:
                # Replace the basic parsed content with LLM-enhanced structure
                experience_section.subsections = enhanced_experience
                logger.info(
                    f"Enhanced Professional Experience section with {len(enhanced_experience)} roles"
                )

        # Find and enhance Project Experience section
        project_section = next(
            (
                section
                for section in structured_cv.sections
                if section.name == "Project Experience"
            ),
            None,
        )
        if project_section:
            enhanced_projects = await self._parse_projects_section_with_llm(cv_text)
            if enhanced_projects:
                # Replace the basic parsed content with LLM-enhanced structure
                project_section.subsections = enhanced_projects
                logger.info(
                    f"Enhanced Project Experience section with {len(enhanced_projects)} projects"
                )

    async def _parse_experience_section_with_llm(
        self, cv_text: str
    ) -> List[Subsection]:
        """
        Parse the Professional Experience section using LLM to extract structured role information.

        Args:
            cv_text: The complete CV text

        Returns:
            List of Subsection objects representing individual roles
        """
        try:
            # Extract the experience section from the CV text
            experience_text = await self._extract_section_text(
                cv_text, "Professional Experience"
            )
            if not experience_text:
                logger.warning("No Professional Experience section found in CV text")
                return []

            # Create prompt for LLM to parse experience
            prompt = f"""
            Parse the following Professional Experience section and extract structured information for each role.

            Experience Text:
            {experience_text}

            Please return a JSON array where each object represents a role with the following structure:
            {{
                "title": "Job Title",
                "company": "Company Name",
                "duration": "Start Date - End Date",
                "location": "City, State/Country (if available)",
                "responsibilities": ["responsibility 1", "responsibility 2", "responsibility 3"]
            }}

            Extract all roles found in the text. If some information is missing, use reasonable defaults or leave empty strings.
            Return only the JSON array, no additional text.
            """

            # Generate response using LLM
            response = await self.llm.generate_content(prompt)

            if not response:
                logger.error("Empty response from LLM for experience parsing")
                return []

            # Handle response content
            content = response.content if hasattr(response, "content") else response

            # Parse JSON response
            json_start = content.find("[")
            json_end = content.rfind("]") + 1
            if json_start == -1 or json_end == 0:
                logger.error("No valid JSON array found in LLM response")
                return []

            json_str = content[json_start:json_end]
            parsed_roles = json.loads(json_str)

            # Convert to Subsection objects
            subsections = []
            for role in parsed_roles:
                # Create subsection name from title and company
                title = role.get("title", "Position")
                company = role.get("company", "Company")
                duration = role.get("duration", "")
                location = role.get("location", "")

                # Format subsection name
                subsection_name = f"{title} at {company}"
                if duration:
                    subsection_name += f" ({duration})"
                if location:
                    subsection_name += f" - {location}"

                subsection = Subsection(name=subsection_name)

                # Add responsibilities as items
                responsibilities = role.get("responsibilities", [])
                for responsibility in responsibilities:
                    if responsibility.strip():
                        item = Item(
                            content=responsibility.strip(),
                            status=ItemStatus.INITIAL,
                            item_type=ItemType.BULLET_POINT,
                        )
                        subsection.items.append(item)

                subsections.append(subsection)

            logger.info(
                f"Successfully parsed {len(subsections)} roles from Professional Experience"
            )
            return subsections

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for experience: {e}")
            return []
        except Exception as e:
            logger.error(f"Error in _parse_experience_section_with_llm: {e}")
            return []

    async def _parse_projects_section_with_llm(self, cv_text: str) -> List[Subsection]:
        """
        Parse the Project Experience section using LLM to extract structured project information.

        Args:
            cv_text: The complete CV text

        Returns:
            List of Subsection objects representing individual projects
        """
        try:
            # Extract the projects section from the CV text
            projects_text = await self._extract_section_text(
                cv_text, "Project Experience"
            )
            if not projects_text:
                logger.warning("No Project Experience section found in CV text")
                return []

            # Create prompt for LLM to parse projects
            prompt = f"""
            Parse the following Project Experience section and extract structured information for each project.

            Projects Text:
            {projects_text}

            Please return a JSON array where each object represents a project with the following structure:
            {{
                "name": "Project Name",
                "technologies": "Technologies/Tools used (if available)",
                "duration": "Project duration or timeframe (if available)",
                "description": ["key point 1", "key point 2", "key point 3"]
            }}

            Extract all projects found in the text. Focus on technical achievements, technologies used, and impact.
            If some information is missing, use reasonable defaults or leave empty strings.
            Return only the JSON array, no additional text.
            """

            # Generate response using LLM
            response = await self.llm.generate_content(prompt)

            if not response:
                logger.error("Empty response from LLM for projects parsing")
                return []

            # Handle response content
            content = response.content if hasattr(response, "content") else response

            # Parse JSON response
            json_start = content.find("[")
            json_end = content.rfind("]") + 1
            if json_start == -1 or json_end == 0:
                logger.error("No valid JSON array found in LLM response")
                return []

            json_str = content[json_start:json_end]
            parsed_projects = json.loads(json_str)

            # Convert to Subsection objects
            subsections = []
            for project in parsed_projects:
                # Create subsection name from project name and technologies
                name = project.get("name", "Project")
                technologies = project.get("technologies", "")
                duration = project.get("duration", "")

                # Format subsection name
                subsection_name = name
                if technologies:
                    subsection_name += f" ({technologies})"
                if duration:
                    subsection_name += f" - {duration}"

                subsection = Subsection(name=subsection_name)

                # Add description points as items
                description_points = project.get("description", [])
                for point in description_points:
                    if point.strip():
                        item = Item(
                            content=point.strip(),
                            status=ItemStatus.INITIAL,
                            item_type=ItemType.BULLET_POINT,
                        )
                        subsection.items.append(item)

                subsections.append(subsection)

            logger.info(
                f"Successfully parsed {len(subsections)} projects from Project Experience"
            )
            return subsections

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for projects: {e}")
            return []
        except Exception as e:
            logger.error(f"Error in _parse_projects_section_with_llm: {e}")
            return []

    async def _extract_section_text(self, cv_text: str, section_name: str) -> str:
        """
        Extract the text content of a specific section from the CV using LLM.

        Args:
            cv_text: The complete CV text
            section_name: The name of the section to extract

        Returns:
            The text content of the section, or empty string if not found
        """
        try:
            # Create prompt for LLM to extract the specific section
            prompt = f"""
            Extract the "{section_name}" section from the following CV text.

            CV Text:
            {cv_text}

            Please return ONLY the content of the "{section_name}" section, without the section header.
            If the section is not found, return "SECTION_NOT_FOUND".
            Do not include any other text or explanations.

            The section content should include all subsections, bullet points, and details that belong to "{section_name}".
            """

            # Generate response using LLM
            response = await self.llm.generate_content(prompt)

            if (
                not response
                or (
                    hasattr(response, "content")
                    and response.content.strip() == "SECTION_NOT_FOUND"
                )
                or response.strip() == "SECTION_NOT_FOUND"
            ):
                logger.warning(f"Section '{section_name}' not found in CV text")
                return ""

            content = response.content if hasattr(response, "content") else response
            return content.strip()

        except Exception as e:
            logger.error(f"Error extracting section '{section_name}' with LLM: {e}")
            return ""

    def create_empty_cv_structure(self, job_data: JobDescriptionData) -> StructuredCV:
        """
        Creates an empty CV structure for the "Start from Scratch" option.

        Args:
            job_data: The parsed job description data.

        Returns:
            A StructuredCV object with empty sections.
        """
        # Create a new StructuredCV
        structured_cv = StructuredCV()

        # Add metadata - handle both dict and JobDescriptionData object types
        if job_data:
            if hasattr(job_data, "to_dict"):
                # JobDescriptionData object
                structured_cv.metadata["job_description"] = job_data.to_dict()
            elif isinstance(job_data, dict):
                # Already a dictionary
                structured_cv.metadata["job_description"] = job_data
            else:
                # Fallback for other types
                structured_cv.metadata["job_description"] = {}
        else:
            structured_cv.metadata["job_description"] = {}
        structured_cv.metadata["start_from_scratch"] = True

        # Create standard CV sections with proper order
        sections = [
            {"name": "Executive Summary", "type": "DYNAMIC", "order": 0},
            {"name": "Key Qualifications", "type": "DYNAMIC", "order": 1},
            {"name": "Professional Experience", "type": "DYNAMIC", "order": 2},
            {"name": "Project Experience", "type": "DYNAMIC", "order": 3},
            {"name": "Education", "type": "STATIC", "order": 4},
            {"name": "Certifications", "type": "STATIC", "order": 5},
            {"name": "Languages", "type": "STATIC", "order": 6},
        ]

        # Create and add the sections
        for section_info in sections:
            section = Section(
                name=section_info["name"],
                content_type=section_info["type"],
                order=section_info["order"],
            )

            # For Executive Summary, add an empty item
            if section.name == "Executive Summary":
                section.items.append(
                    Item(
                        content="",
                        status=ItemStatus.TO_REGENERATE,
                        item_type=ItemType.EXECUTIVE_SUMMARY_PARA,
                    )
                )

            # For Key Qualifications, add empty items based on skills from job description
            skills = None
            if job_data:
                if hasattr(job_data, "skills"):
                    skills = job_data.skills
                elif isinstance(job_data, dict) and "skills" in job_data:
                    skills = job_data["skills"]

            if section.name == "Key Qualifications" and skills:
                # Limit to 8 skills maximum
                for skill in skills[:8]:
                    section.items.append(
                        Item(
                            content=skill,
                            status=ItemStatus.TO_REGENERATE,
                            item_type=ItemType.KEY_QUALIFICATION,
                        )
                    )

            # For Professional Experience, add an empty subsection
            if section.name == "Professional Experience":
                subsection = Subsection(name="Position Title at Company Name")
                # Add some default bullet points
                for _ in range(3):
                    subsection.items.append(
                        Item(
                            content="",
                            status=ItemStatus.TO_REGENERATE,
                            item_type=ItemType.BULLET_POINT,
                        )
                    )
                section.subsections.append(subsection)

            # For Project Experience, add an empty subsection
            if section.name == "Project Experience":
                subsection = Subsection(name="Project Name")
                # Add some default bullet points
                for _ in range(2):
                    subsection.items.append(
                        Item(
                            content="",
                            status=ItemStatus.TO_REGENERATE,
                            item_type=ItemType.BULLET_POINT,
                        )
                    )
                section.subsections.append(subsection)

            structured_cv.sections.append(section)

        return structured_cv

    def get_job_data(self) -> Dict[str, Any]:
        """
        Returns the previously parsed job description data.

        Returns:
            The parsed job description data as a dictionary, or an empty dict if no job has been parsed
        """
        if hasattr(self, "_job_data") and self._job_data:
            return self._job_data
        return {}

    def get_confidence_score(self, output_data: Any) -> float:
        """
        Returns a confidence score for the agent's output based on data completeness.

        Args:
            output_data: The output data generated by the agent.

        Returns:
            A float representing the confidence score (0.0 to 1.0).
        """
        if not output_data:
            return 0.0

        # Base confidence score
        confidence = 0.3

        # Check for required fields in job description data
        required_fields = [
            "job_title",
            "required_skills",
            "experience_level",
            "education_requirements",
        ]
        present_fields = 0

        for field in required_fields:
            if field in output_data and output_data[field]:
                present_fields += 1

        # Calculate confidence based on completeness
        field_completeness = present_fields / len(required_fields)
        confidence += field_completeness * 0.7

        return min(confidence, 1.0)

    def _extract_skills_from_text(self, text: str) -> List[str]:
        """
        Extract skills from text using pattern matching.

        Args:
            text: The text to extract skills from

        Returns:
            List of extracted skills
        """
        if not text:
            return []

        # Common skill patterns and keywords
        skill_patterns = [
            r"\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|PHP|Ruby|Go|Rust|Swift|Kotlin)\b",
            r"\b(?:React|Angular|Vue|Django|Flask|Spring|Express|Laravel|Rails)\b",
            r"\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|Linux|Windows)\b",
            r"\b(?:SQL|MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch)\b",
            r"\b(?:HTML|CSS|SASS|LESS|Bootstrap|Tailwind)\b",
            r"\b(?:Node\.js|React\.js|Vue\.js|Next\.js|Nuxt\.js)\b",
            r"\b(?:Machine Learning|AI|Data Science|Analytics|Statistics)\b",
            r"\b(?:Agile|Scrum|DevOps|CI/CD|TDD|BDD)\b",
        ]

        skills = []
        text_lower = text.lower()

        # Extract skills using patterns
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend(matches)

        # Also look for comma-separated skills
        # Common phrases that indicate skill lists
        skill_indicators = [
            r"(?:experienced in|proficient in|skilled in|expertise in|knowledge of)\s*([^.]+)",
            r"(?:technologies|skills|tools)\s*:?\s*([^.]+)",
            r"(?:including|such as)\s*([^.]+)",
        ]

        for indicator in skill_indicators:
            matches = re.findall(indicator, text, re.IGNORECASE)
            for match in matches:
                # Split by common separators and clean up
                potential_skills = re.split(r"[,;\n]+", match)
                for skill in potential_skills:
                    skill = skill.strip().strip("and").strip()
                    if skill and len(skill) > 1:
                        skills.append(skill)

        # Remove duplicates and clean up
        unique_skills = []
        for skill in skills:
            skill = skill.strip()
            if skill and skill not in unique_skills:
                unique_skills.append(skill)

        return unique_skills

    def _extract_sections(self, cv_content: str) -> List[Section]:
        """
        Extract sections from CV content.

        Args:
            cv_content: The CV content to parse

        Returns:
            List of Section objects
        """
        sections = []

        # Simple section extraction based on common CV patterns
        lines = cv_content.split("\n")
        current_section = None
        current_items = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line looks like a section header
            if self._is_section_header(line):
                # Save previous section if exists
                if current_section:
                    sections.append(
                        Section(
                            name=current_section,
                            items=[
                                Item(content=item, status=ItemStatus.INITIAL)
                                for item in current_items
                            ],
                        )
                    )

                # Start new section
                current_section = line
                current_items = []
            else:
                # Add line as content to current section
                if current_section:
                    current_items.append(line)
                else:
                    # If no section header found yet, treat as personal info
                    if not sections:
                        current_section = "Personal Information"
                        current_items = [line]

        # Add the last section
        if current_section:
            sections.append(
                Section(
                    name=current_section,
                    items=[
                        Item(content=item, status=ItemStatus.INITIAL)
                        for item in current_items
                    ],
                )
            )

        return sections

    def _is_section_header(self, line: str) -> bool:
        """
        Determine if a line is likely a section header.

        Args:
            line: The line to check

        Returns:
            True if the line appears to be a section header
        """
        # Common section headers
        section_keywords = [
            "personal information",
            "contact",
            "summary",
            "objective",
            "experience",
            "employment",
            "work history",
            "professional experience",
            "education",
            "qualifications",
            "skills",
            "technical skills",
            "projects",
            "achievements",
            "certifications",
            "languages",
            "references",
            "interests",
            "hobbies",
        ]

        line_lower = line.lower()

        # Check if line contains section keywords
        for keyword in section_keywords:
            if keyword in line_lower:
                return True

        # Check if line is all caps (common for headers)
        if line.isupper() and len(line) > 3:
            return True

        # Check if line ends with colon
        if line.endswith(":"):
            return True

        return False

    async def parse_cv(self, cv_content: str) -> StructuredCV:
        """
        Parse CV content and return a structured CV.

        Args:
            cv_content: The CV content to parse

        Returns:
            StructuredCV object with parsed sections

        Raises:
            ValueError: If CV content is empty or None
        """
        if not cv_content or not cv_content.strip():
            raise ValueError("CV content cannot be empty")

        # Extract sections from CV content
        sections = self._extract_sections(cv_content)

        # Create and return StructuredCV
        return StructuredCV(sections=sections)

    async def run_async(
        self, input_data: Any, context: "AgentExecutionContext"
    ) -> "AgentResult":
        """
        Legacy async method required by abstract base class.

        This method provides compatibility with the abstract base class
        but is not the primary execution path for LangGraph workflows.
        """
        from .agent_base import AgentResult
        from src.orchestration.state import AgentState
        from src.models.data_models import JobDescriptionData, StructuredCV

        try:
            # Simple implementation that returns a basic result
            # The main execution path is through run_as_node
            return AgentResult(
                success=True,
                output_data={
                    "message": "ParserAgent executed via run_async - use run_as_node for full functionality"
                },
                confidence_score=1.0,
                metadata={"agent_type": "parser", "execution_method": "run_async"},
            )

        except Exception as e:
            logger.error(f"Error in ParserAgent.run_async: {e}", exc_info=True)
            return AgentResult(
                success=False,
                output_data={"error": str(e)},
                confidence_score=0.0,
                error_message=str(e),
                metadata={"agent_type": "parser", "error": True},
            )

    @optimize_async("agent_execution", "parser")
    async def run_as_node(self, state: AgentState) -> dict:
        """
        Executes the complete parsing logic as a LangGraph node.

        This method now correctly handles parsing the job description,
        parsing an existing CV text from metadata, or creating an empty CV
        structure based on the initial state, ensuring the agent's full
        capability is exposed to the workflow.
        """
        logger.info(
            "ParserAgent node running with consolidated logic.",
            extra={"trace_id": state.trace_id},
        )

        try:
            # Initialize job_data to None
            job_data = None

            # 1. Always parse the job description first, if it exists.
            if state.job_description_data and state.job_description_data.raw_text:
                logger.info(
                    "Starting job description parsing.",
                    extra={"trace_id": state.trace_id},
                )
                job_data = await self.parse_job_description(
                    state.job_description_data.raw_text, trace_id=state.trace_id
                )
            else:
                logger.warning(
                    "No job description text found in the state.",
                    extra={"trace_id": state.trace_id},
                )
                # Create empty JobDescriptionData if none exists
                job_data = JobDescriptionData(raw_text="")

            # 2. Determine the CV processing path from the structured_cv metadata.
            # This metadata is set by the create_agent_state_from_ui function.
            cv_metadata = state.structured_cv.metadata if state.structured_cv else {}
            start_from_scratch = cv_metadata.get("start_from_scratch", False)
            original_cv_text = cv_metadata.get("original_cv_text", "")

            final_cv = None
            if start_from_scratch:
                logger.info(
                    "Creating empty CV structure for 'Start from Scratch' option.",
                    extra={"trace_id": state.trace_id},
                )
                final_cv = self.create_empty_cv_structure(job_data)
            elif original_cv_text:
                logger.info(
                    "Parsing provided CV text.", extra={"trace_id": state.trace_id}
                )
                final_cv = self.parse_cv_text(original_cv_text, job_data)
            else:
                logger.warning(
                    "No CV text provided and not starting from scratch. Passing CV state through."
                )
                final_cv = state.structured_cv

            # 3. Return the complete, updated state.
            return {"structured_cv": final_cv, "job_description_data": job_data}

        except Exception as e:
            logger.error(f"Critical error in ParserAgent node: {e}", exc_info=True)
            error_list = state.error_messages or []
            error_list.append(f"ParserAgent Error: {str(e)}")
            # Return the errors to be handled by the graph's error handling mechanism
            return {"error_messages": error_list}
