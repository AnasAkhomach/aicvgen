from agent_base import AgentBase
from state_manager import AgentIO, JobDescriptionData
from typing import Dict, Any, List

class QualityAssuranceAgent(AgentBase):
    """
    Agent responsible for performing quality checks on the tailored CV content.
    """

    def __init__(self, name: str, description: str):
        """
        Initializes the QualityAssuranceAgent.

        Args:
            name: The name of the agent.
            description: A description of the agent.
        """
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                input={
                    "formatted_cv_text": str, # Takes the formatted CV content
                    "job_description": Dict[str, Any] # May need job description for context (e.g., ATS keywords)
                },
                output=Dict[str, Any], # Outputs validation results and feedback
                description="Formatted CV content and job description for quality assurance.",
            ),
            output_schema=AgentIO(
                input={
                    "formatted_cv_text": str,
                    "job_description": Dict[str, Any]
                },
                output=Dict[str, Any],
                description="Quality assurance results and feedback.",
            ),
        )

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs quality checks on the formatted CV content.

        Args:
            input: A dictionary containing 'formatted_cv_text' (str) and optional 'job_description' (Dict).

        Returns:
            A dictionary containing quality assurance results and feedback.
            Expected keys: 'is_quality_ok', 'feedback', 'suggestions'.
        """
        formatted_cv_text: str = input.get("formatted_cv_text", "")
        job_description_data: Dict[str, Any] = input.get("job_description", {})

        print("Executing: QualityAssuranceAgent")

        # --- Placeholder Quality Assurance Logic ---
        # Simulate some checks
        feedback: List[str] = []
        suggestions: List[str] = []
        is_quality_ok = True

        # Simulate grammar check (very basic)
        if len(formatted_cv_text.split()) < 10: # Arbitrary check
            feedback.append("Content seems very short, potential for missing details.")
            is_quality_ok = False

        # Simulate ATS keyword check (basic)
        job_skills = job_description_data.get("skills", [])
        missing_keywords = [skill for skill in job_skills if skill.lower() not in formatted_cv_text.lower()]
        if missing_keywords:
            feedback.append(f"Missing potential ATS keywords: {', '.join(missing_keywords)}.")
            suggestions.append(f"Consider incorporating keywords like: {', '.join(missing_keywords)}.")
            is_quality_ok = False
            
        # Simulate consistency check (placeholder)
        # if "inconsistent_pattern" in formatted_cv_text:
        #     feedback.append("Potential inconsistency detected.")
        #     is_quality_ok = False


        quality_results = {
            "is_quality_ok": is_quality_ok,
            "feedback": " ".join(feedback) if feedback else "No major issues detected (simulated).",  # Join feedback messages
            "suggestions": suggestions
        }
        # ---------------------------------------------

        print("Completed: QualityAssuranceAgent (Simulated Checks)")
        return quality_results
