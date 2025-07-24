"""Debug script to capture real prompts and responses for KeyQualificationsWriterAgent.

This script uses the actual LLM service to see the exact prompts sent to the LLM
and the actual responses generated, without any mocks or templates.
"""

import asyncio
from typing import Dict, Any

from src.core.container import get_container
from src.models.cv_models import StructuredCV, JobDescriptionData
from src.models.llm_data_models import (
    PersonalInfo, Experience, Education, Project, Skill,
    Certification, Language, BasicCVInfo, LLMResponse
)
from src.services.llm_service_interface import LLMServiceInterface
from src.config.logging_config import get_logger

logger = get_logger(__name__)


class PromptCaptureLLMService(LLMServiceInterface):
    """LLM Service wrapper that captures and logs prompts and responses."""
    
    def __init__(self, real_llm_service: LLMServiceInterface):
        self.real_llm_service = real_llm_service
        self.captured_prompts = []
        self.captured_responses = []
    
    async def generate_content(
        self,
        prompt: str,
        system_instruction: str = None,
        **kwargs
    ) -> LLMResponse:
        """Generate content and capture the prompt and response."""
        print("\n" + "="*80)
        print("CAPTURED PROMPT:")
        print("="*80)
        if system_instruction:
            print(f"SYSTEM INSTRUCTION:\n{system_instruction}\n")
        print(f"USER PROMPT:\n{prompt}")
        print("="*80)
        
        # Store the prompt
        self.captured_prompts.append({
            'system_instruction': system_instruction,
            'user_prompt': prompt,
            'kwargs': kwargs
        })
        
        # Call the real LLM service
        response = await self.real_llm_service.generate_content(
            prompt=prompt,
            system_instruction=system_instruction,
            **kwargs
        )
        
        print("\nCAPTURED RESPONSE:")
        print("="*80)
        print(f"CONTENT:\n{response.content}")
        print(f"\nTOKENS USED: {response.tokens_used}")
        print(f"PROCESSING TIME: {response.processing_time}")
        print(f"MODEL USED: {response.model_used}")
        print(f"SUCCESS: {response.success}")
        if response.error_message:
            print(f"ERROR: {response.error_message}")
        print("="*80)
        
        # Store the response
        self.captured_responses.append(response)
        
        return response
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate method wrapper."""
        return await self.generate_content(prompt, **kwargs)
    
    async def validate_api_key(self) -> bool:
        """Validate the current API key."""
        return await self.real_llm_service.validate_api_key()
    
    def get_current_api_key_info(self):
        """Get information about the currently active API key."""
        return self.real_llm_service.get_current_api_key_info()
    
    async def ensure_api_key_valid(self):
        """Ensure the API key is valid."""
        return await self.real_llm_service.ensure_api_key_valid()
    
    def get_captured_data(self):
        """Return all captured prompts and responses."""
        return {
            'prompts': self.captured_prompts,
            'responses': self.captured_responses
        }


def create_sample_structured_cv() -> StructuredCV:
    """Create a sample StructuredCV for testing."""
    from src.models.cv_models import (
        Section, Subsection, Item, ItemStatus, ItemType
    )
    from uuid import uuid4
    
    return StructuredCV(
        sections=[
            Section(
                id=uuid4(),
                name="Executive Summary",
                content_type="DYNAMIC",
                order=0,
                status=ItemStatus.INITIAL,
                items=[
                    Item(
                        id=uuid4(),
                        content="Experienced software engineer with 8+ years in full-stack development",
                        status=ItemStatus.INITIAL,
                        item_type=ItemType.BULLET_POINT
                    )
                ]
            ),
            Section(
                id=uuid4(),
                name="Key Qualifications",
                content_type="DYNAMIC",
                order=1,
                status=ItemStatus.INITIAL,
                items=[
                    Item(
                        id=uuid4(),
                        content="Python, JavaScript, React, Node.js",
                        status=ItemStatus.INITIAL,
                        item_type=ItemType.BULLET_POINT
                    ),
                    Item(
                        id=uuid4(),
                        content="AWS, Docker, Kubernetes",
                        status=ItemStatus.INITIAL,
                        item_type=ItemType.BULLET_POINT
                    )
                ]
            ),
            Section(
                id=uuid4(),
                name="Professional Experience",
                content_type="DYNAMIC",
                order=2,
                status=ItemStatus.INITIAL,
                subsections=[
                    Subsection(
                        id=uuid4(),
                        name="Senior Software Engineer - Tech Corp (2020-Present)",
                        order=0,
                        status=ItemStatus.INITIAL,
                        items=[
                            Item(
                                id=uuid4(),
                                content="Led development of microservices architecture serving 1M+ users",
                                status=ItemStatus.INITIAL,
                                item_type=ItemType.BULLET_POINT
                            ),
                            Item(
                                id=uuid4(),
                                content="Reduced API response time by 40% through optimization",
                                status=ItemStatus.INITIAL,
                                item_type=ItemType.BULLET_POINT
                            )
                        ]
                    ),
                    Subsection(
                        id=uuid4(),
                        name="Software Engineer - StartupXYZ (2018-2019)",
                        order=1,
                        status=ItemStatus.INITIAL,
                        items=[
                            Item(
                                id=uuid4(),
                                content="Developed full-stack web applications using React and Node.js",
                                status=ItemStatus.INITIAL,
                                item_type=ItemType.BULLET_POINT
                            ),
                            Item(
                                id=uuid4(),
                                content="Built user authentication system handling 10K+ users",
                                status=ItemStatus.INITIAL,
                                item_type=ItemType.BULLET_POINT
                            )
                        ]
                    )
                ]
            )
        ]
    )


def create_sample_job_description() -> JobDescriptionData:
    """Create a sample JobDescriptionData for testing."""
    return JobDescriptionData(
        raw_text="""Senior Full Stack Developer - InnovativeTech Solutions
        
We are seeking a Senior Full Stack Developer to join our dynamic team and help build next-generation web applications that serve millions of users worldwide.
        
Required Skills:
        - 5+ years of experience in full-stack development
        - Proficiency in React, Node.js, and TypeScript
        - Experience with cloud platforms (AWS, Azure, or GCP)
        - Strong understanding of RESTful APIs and microservices
        - Experience with database design and optimization
        
Preferred Skills:
        - Experience with containerization (Docker, Kubernetes)
        - Knowledge of CI/CD pipelines
        - Experience with agile development methodologies
        - Strong problem-solving and communication skills
        
Education: Bachelor's degree in Computer Science or related field
        Salary: $120,000 - $180,000
        """,
        job_title="Senior Full Stack Developer",
        company_name="InnovativeTech Solutions",
        main_job_description_raw="We are seeking a Senior Full Stack Developer to join our dynamic team and help build next-generation web applications that serve millions of users worldwide.",
        skills=[
            "React", "Node.js", "TypeScript", "AWS", "Azure", "GCP",
            "RESTful APIs", "Microservices", "Database Design", "Docker",
            "Kubernetes", "CI/CD", "Agile Development"
        ],
        experience_level="Senior",
        responsibilities=[
            "Build next-generation web applications",
            "Develop full-stack solutions",
            "Work with cloud platforms",
            "Design and optimize databases",
            "Implement microservices architecture"
        ],
        industry_terms=[
            "Full-stack development", "Microservices", "Cloud platforms",
            "RESTful APIs", "Database optimization", "Containerization"
        ],
        company_values=[
            "Innovation", "Teamwork", "Quality", "Scalability", "User-focused"
        ]
    )


async def main():
    """Main function to test KeyQualificationsWriterAgent with real LLM service."""
    print("Starting Key Qualifications Agent Debug with Real LLM Service...")
    
    # Get the container and real LLM service
    container = get_container()
    real_llm_service = container.llm_service()
    
    # Wrap the real LLM service with our prompt capture wrapper
    prompt_capture_service = PromptCaptureLLMService(real_llm_service)
    
    # Get the key qualifications agent from container
    key_qualifications_agent = container.key_qualifications_writer_agent()
    
    # Replace the agent's LLM service with our capturing service
    key_qualifications_agent.llm_service = prompt_capture_service
    
    # Create sample data
    structured_cv = create_sample_structured_cv()
    job_description_data = create_sample_job_description()
    
    print("\nCreated sample data:")
    print(f"CV: Sample CV with {len(structured_cv.sections)} sections")
    print(f"Job: {job_description_data.job_title} at {job_description_data.company_name}")
    
    # Execute the agent
    print("\nExecuting KeyQualificationsWriterAgent...")
    try:
        result = await key_qualifications_agent._execute(
            structured_cv=structured_cv,
            job_description_data=job_description_data
        )
        
        print("\n" + "="*80)
        print("AGENT EXECUTION RESULT:")
        print("="*80)
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        # Show captured data summary
        captured_data = prompt_capture_service.get_captured_data()
        print(f"\nCaptured {len(captured_data['prompts'])} prompts and {len(captured_data['responses'])} responses")
        
    except Exception as e:
        print(f"\nError during agent execution: {e}")
        logger.error(f"Agent execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())