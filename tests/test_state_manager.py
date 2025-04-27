import unittest
from state_manager import (
    WorkflowStage,
    AgentIO,
    JobDescriptionData,
    VectorStoreConfig,
    CVData,
    SkillEntry,
    ExperienceEntry,
    ContentData,
    WorkflowState
)

class TestStateManager(unittest.TestCase):

    def test_workflow_stage_typeddict(self):
        """Test creation and access of WorkflowStage TypedDict."""
        stage_data: WorkflowStage = {
            "stage_name": "Parsing",
            "description": "Extracting data",
            "is_completed": False
        }
        self.assertEqual(stage_data["stage_name"], "Parsing")
        self.assertEqual(stage_data["description"], "Extracting data")
        self.assertEqual(stage_data["is_completed"], False)

    def test_agent_io_typeddict(self):
        """Test creation and access of AgentIO TypedDict."""
        io_data: AgentIO = {
            "input": {"text": str},
            "output": dict,
            "description": "Processes text."
        }
        self.assertEqual(io_data["input"], {"text": str})
        self.assertEqual(io_data["output"], dict)
        self.assertEqual(io_data["description"], "Processes text.")

    def test_job_description_data_class(self):
        """Test creation and access of JobDescriptionData class."""
        job_data = JobDescriptionData(
            raw_text="Job details.",
            skills=["Python", "Java"],
            experience_level="Senior",
            responsibilities=["Lead team"],
            industry_terms=["FinTech"],
            company_values=["Innovation"]
        )
        self.assertEqual(job_data.raw_text, "Job details.")
        self.assertEqual(job_data.skills, ["Python", "Java"])
        self.assertEqual(job_data.experience_level, "Senior")
        self.assertEqual(job_data.responsibilities, ["Lead team"])
        self.assertEqual(job_data.industry_terms, ["FinTech"])
        self.assertEqual(job_data.company_values, ["Innovation"])
        # Test __str__ method
        self.assertIn("JobDescriptionData(", str(job_data))
        self.assertIn("raw_text='Job details.'", str(job_data))

    def test_vector_store_config_dataclass(self):
        """Test creation and access of VectorStoreConfig dataclass."""
        config = VectorStoreConfig(dimension=1024, index_type="IndexFlatL2")
        self.assertEqual(config.dimension, 1024)
        self.assertEqual(config.index_type, "IndexFlatL2")

    def test_cv_data_typeddict(self):
        """Test creation and access of CVData TypedDict."""
        cv_data: CVData = {"raw_text": "My CV content here."}
        self.assertEqual(cv_data["raw_text"], "My CV content here.")

    def test_skill_entry_dataclass(self):
        """Test creation and access of SkillEntry dataclass."""
        skill = SkillEntry(text="Problem Solving")
        self.assertEqual(skill.text, "Problem Solving")

    def test_experience_entry_dataclass(self):
        """Test creation and access of ExperienceEntry dataclass."""
        experience = ExperienceEntry(text="Worked on project X")
        self.assertEqual(experience.text, "Worked on project X")

    def test_content_data_class(self):
        """Test creation and access of ContentData class (inherits from Dict)."""
        content = ContentData(
            summary="Summary text.",
            experience_bullets=["Bullet 1"],
            skills_section="Skills text.",
            projects=["Project A"],
            other_content={"Awards": "Award 1"}
        )
        self.assertEqual(content["summary"], "Summary text.")
        self.assertEqual(content["experience_bullets"], ["Bullet 1"])
        self.assertEqual(content["skills_section"], "Skills text.")
        self.assertEqual(content["projects"], ["Project A"])
        self.assertEqual(content["other_content"], {"Awards": "Award 1"})

        # Test with some fields missing or None
        content_partial = ContentData(summary="Only summary")
        self.assertEqual(content_partial["summary"], "Only summary")
        self.assertNotIn("experience_bullets", content_partial)

    def test_workflow_state_typeddict(self):
        """Test creation and access of WorkflowState TypedDict."""
        state_data: WorkflowState = {
            "job_description": {"skills": ["AI"]},
            "user_cv": {"raw_text": "CV text"},
            "extracted_skills": {"extracted": ["ML"]},
            "generated_content": {"summary": "Generated"},
            "feedback": ["Good"],
            "revision_history": ["Rev 1"],
            "current_stage": {"stage_name": "Done", "description": "Finished", "is_completed": True},
            "workflow_id": "abc-123"
        }
        self.assertEqual(state_data["job_description"], {"skills": ["AI"]})
        self.assertEqual(state_data["user_cv"], {"raw_text": "CV text"})
        self.assertEqual(state_data["extracted_skills"], {"extracted": ["ML"]})
        self.assertEqual(state_data["generated_content"], {"summary": "Generated"})
        self.assertEqual(state_data["feedback"], ["Good"])
        self.assertEqual(state_data["revision_history"], ["Rev 1"])
        self.assertEqual(state_data["current_stage"], {"stage_name": "Done", "description": "Finished", "is_completed": True})
        self.assertEqual(state_data["workflow_id"], "abc-123")


if __name__ == '__main__':
    unittest.main()
