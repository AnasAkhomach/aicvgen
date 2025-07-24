#!/usr/bin/env python3
"""
Simple debug script to capture Key Qualifications Agent prompts and responses.
Saves output to files for easy viewing.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.container import ContainerSingleton
from src.models.cv_models import StructuredCV, JobDescriptionData
from src.models.llm_data_models import LLMResponse
from src.services.llm_service_interface import LLMServiceInterface


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
        # Store the prompt
        prompt_data = {
            'system_instruction': system_instruction,
            'user_prompt': prompt,
            'kwargs': kwargs
        }
        self.captured_prompts.append(prompt_data)
        
        # Call the real LLM service
        response = await self.real_llm_service.generate_content(
            prompt=prompt,
            system_instruction=system_instruction,
            **kwargs
        )
        
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
    
    def save_captured_data(self, filename_prefix="key_qualifications_debug"):
        """Save captured prompts and responses to files."""
        # Save prompts
        with open(f"{filename_prefix}_prompts.txt", "w", encoding="utf-8") as f:
            for i, prompt_data in enumerate(self.captured_prompts):
                f.write(f"\n{'='*80}\n")
                f.write(f"PROMPT {i+1}\n")
                f.write(f"{'='*80}\n")
                if prompt_data['system_instruction']:
                    f.write(f"SYSTEM INSTRUCTION:\n{prompt_data['system_instruction']}\n\n")
                f.write(f"USER PROMPT:\n{prompt_data['user_prompt']}\n")
                if prompt_data['kwargs']:
                    f.write(f"\nKWARGS: {prompt_data['kwargs']}\n")
        
        # Save responses
        with open(f"{filename_prefix}_responses.txt", "w", encoding="utf-8") as f:
            for i, response in enumerate(self.captured_responses):
                f.write(f"\n{'='*80}\n")
                f.write(f"RESPONSE {i+1}\n")
                f.write(f"{'='*80}\n")
                f.write(f"CONTENT:\n{response.content}\n\n")
                f.write(f"TOKENS USED: {response.tokens_used}\n")
                f.write(f"PROCESSING TIME: {response.processing_time}\n")
                f.write(f"MODEL USED: {response.model_used}\n")
                f.write(f"SUCCESS: {response.success}\n")
                if response.error_message:
                    f.write(f"ERROR: {response.error_message}\n")


def create_sample_structured_cv() -> StructuredCV:
    """Create a sample StructuredCV for testing."""
    return StructuredCV.create_empty(
        cv_text="Sample CV for John Doe, experienced software engineer"
    )


def create_sample_job_description() -> JobDescriptionData:
    """Create a sample JobDescriptionData for testing."""
    return JobDescriptionData(
        raw_text="""Senior Software Engineer - Full Stack Development
        
We are seeking a Senior Software Engineer to join our dynamic team. The ideal candidate will have:
        
        Required Skills:
        - 5+ years of experience in Python and JavaScript
        - Experience with React, Node.js, and modern web frameworks
        - Strong knowledge of cloud platforms (AWS, Azure, GCP)
        - Experience with microservices architecture
        - Proficiency in database design and optimization
        - Knowledge of DevOps practices and CI/CD pipelines
        - Experience with containerization (Docker, Kubernetes)
        - Strong problem-solving and analytical skills
        - Excellent communication and teamwork abilities
        - Experience with Agile development methodologies
        
        Responsibilities:
        - Design and develop scalable web applications
        - Collaborate with cross-functional teams
        - Mentor junior developers
        - Participate in code reviews and technical discussions
        - Implement best practices for software development
        """,
        job_title="Senior Software Engineer",
        company_name="TechCorp Inc.",
        main_job_description_raw="Senior Software Engineer position focusing on full-stack development",
        skills=["Python", "JavaScript", "React", "Node.js", "AWS", "Docker", "Kubernetes"],
        experience_level="Senior (5+ years)",
        responsibilities=[
            "Design and develop scalable web applications",
            "Collaborate with cross-functional teams",
            "Mentor junior developers"
        ],
        industry_terms=["microservices", "CI/CD", "DevOps", "Agile"],
        company_values=["innovation", "collaboration", "excellence"]
    )


async def main():
    """Main function to test the KeyQualificationsWriterAgent."""
    print("Starting Key Qualifications Agent Debug...")
    
    # Get the container and real LLM service
    container = ContainerSingleton.get_instance()
    real_llm_service = container.llm_service()
    
    # Wrap with our capture service
    capture_service = PromptCaptureLLMService(real_llm_service)
    
    # Get the agent and replace its LLM service
    agent = container.key_qualifications_writer_agent()
    agent.llm_service = capture_service
    
    # Create sample data
    structured_cv = create_sample_structured_cv()
    job_description_data = create_sample_job_description()
    
    print(f"CV: Sample CV with {len(structured_cv.sections)} sections")
    print(f"Job: {job_description_data.job_title} at {job_description_data.company_name}")
    
    # Execute the agent
    try:
        result = await agent._execute(
            structured_cv=structured_cv,
            job_description_data=job_description_data,
            current_item_id="key_qualifications_section",
            research_findings=None,
            session_id="debug_session_123"
        )
        
        print(f"\nAgent execution completed successfully!")
        print(f"Result type: {type(result)}")
        
        # Save captured data to files
        capture_service.save_captured_data()
        print(f"\nCaptured data saved to:")
        print(f"- key_qualifications_debug_prompts.txt")
        print(f"- key_qualifications_debug_responses.txt")
        
    except Exception as e:
        print(f"Error during agent execution: {e}")
        import traceback
        traceback.print_exc()
        
        # Still save captured data even if there was an error
        capture_service.save_captured_data()
        print(f"\nCaptured data saved to files despite error.")


if __name__ == "__main__":
    asyncio.run(main())