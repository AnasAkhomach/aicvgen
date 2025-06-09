import unittest
import sys
import os
import json
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.state_manager import (
    WorkflowStage,
    AgentIO,
    JobDescriptionData,
    VectorStoreConfig,
    CVData,
    SkillEntry,
    ExperienceEntry,
    ContentData,
    WorkflowState,
    StructuredCV,
    Section,
    Subsection,
    Item,
    ItemStatus,
    ItemType,
    ContentPiece,
)


class TestStateManager(unittest.TestCase):

    def test_workflow_stage_typeddict(self):
        """Test creation and access of WorkflowStage TypedDict."""
        stage_data: WorkflowStage = {
            "stage_name": "Parsing",
            "description": "Extracting data",
            "is_completed": False,
        }
        self.assertEqual(stage_data["stage_name"], "Parsing")
        self.assertEqual(stage_data["description"], "Extracting data")
        self.assertEqual(stage_data["is_completed"], False)

    def test_agent_io_typeddict(self):
        """Test creation and access of AgentIO TypedDict."""
        io_data: AgentIO = {
            "input": {"text": str},
            "output": dict,
            "description": "Processes text.",
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
            company_values=["Innovation"],
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
            other_content={"Awards": "Award 1"},
        )
        self.assertEqual(content["summary"], "Summary text.")
        self.assertEqual(content["experience_bullets"], ["Bullet 1"])
        self.assertEqual(content["skills_section"], "Skills text.")
        self.assertEqual(content["projects"], ["Project A"])
        self.assertEqual(content["other_content"], {"Awards": "Award 1"})

        # Test with some fields missing or None
        content_partial = ContentData(summary="Only summary")
        self.assertEqual(content_partial["summary"], "Only summary")
        # Our implementation initializes all fields with default values
        self.assertEqual(content_partial["experience_bullets"], [])
        self.assertEqual(content_partial["skills_section"], "")
        self.assertEqual(content_partial["projects"], [])
        self.assertEqual(content_partial["other_content"], {})

    def test_workflow_state_typeddict(self):
        """Test creation and access of WorkflowState TypedDict."""
        state_data: WorkflowState = {
            "job_description": {"skills": ["AI"]},
            "user_cv": {"raw_text": "CV text"},
            "extracted_skills": {"extracted": ["ML"]},
            "generated_content": {"summary": "Generated"},
            "feedback": ["Good"],
            "revision_history": ["Rev 1"],
            "current_stage": {
                "stage_name": "Done",
                "description": "Finished",
                "is_completed": True,
            },
            "workflow_id": "abc-123",
        }
        self.assertEqual(state_data["job_description"], {"skills": ["AI"]})
        self.assertEqual(state_data["user_cv"], {"raw_text": "CV text"})
        self.assertEqual(state_data["extracted_skills"], {"extracted": ["ML"]})
        self.assertEqual(state_data["generated_content"], {"summary": "Generated"})
        self.assertEqual(state_data["feedback"], ["Good"])
        self.assertEqual(state_data["revision_history"], ["Rev 1"])
        self.assertEqual(
            state_data["current_stage"],
            {"stage_name": "Done", "description": "Finished", "is_completed": True},
        )
        self.assertEqual(state_data["workflow_id"], "abc-123")


    def test_item_creation_and_serialization(self):
        """Test creation and serialization of Item class."""
        item = Item(
            content="Developed scalable web applications",
            status=ItemStatus.GENERATED,
            item_type=ItemType.BULLET_POINT,
            metadata={"relevance_score": 0.85},
            user_feedback="Good content"
        )
        
        self.assertEqual(item.content, "Developed scalable web applications")
        self.assertEqual(item.status, ItemStatus.GENERATED)
        self.assertEqual(item.item_type, ItemType.BULLET_POINT)
        self.assertEqual(item.metadata["relevance_score"], 0.85)
        self.assertEqual(item.user_feedback, "Good content")
        
        # Test serialization
        item_dict = item.to_dict()
        self.assertEqual(item_dict["content"], "Developed scalable web applications")
        self.assertEqual(item_dict["status"], "generated")
        self.assertEqual(item_dict["item_type"], "bullet_point")
        
        # Test deserialization
        restored_item = Item.from_dict(item_dict)
        self.assertEqual(restored_item.content, item.content)
        self.assertEqual(restored_item.status, item.status)
        self.assertEqual(restored_item.item_type, item.item_type)

    def test_subsection_creation_and_serialization(self):
        """Test creation and serialization of Subsection class."""
        items = [
            Item(content="First bullet point", item_type=ItemType.BULLET_POINT),
            Item(content="Second bullet point", item_type=ItemType.BULLET_POINT)
        ]
        
        subsection = Subsection(
            name="Software Engineer at TechCorp",
            items=items,
            metadata={"company": "TechCorp", "dates": "2020-2023"},
            raw_text="Original job description text"
        )
        
        self.assertEqual(subsection.name, "Software Engineer at TechCorp")
        self.assertEqual(len(subsection.items), 2)
        self.assertEqual(subsection.metadata["company"], "TechCorp")
        
        # Test serialization
        subsection_dict = subsection.to_dict()
        self.assertEqual(len(subsection_dict["items"]), 2)
        
        # Test deserialization
        restored_subsection = Subsection.from_dict(subsection_dict)
        self.assertEqual(restored_subsection.name, subsection.name)
        self.assertEqual(len(restored_subsection.items), 2)

    def test_section_creation_and_serialization(self):
        """Test creation and serialization of Section class."""
        # Create items for direct section items
        section_items = [
            Item(content="Key qualification 1", item_type=ItemType.KEY_QUAL),
            Item(content="Key qualification 2", item_type=ItemType.KEY_QUAL)
        ]
        
        # Create subsections
        subsections = [
            Subsection(
                name="Job 1",
                items=[Item(content="Job 1 bullet", item_type=ItemType.BULLET_POINT)]
            )
        ]
        
        section = Section(
            name="Key Qualifications",
            content_type="DYNAMIC",
            items=section_items,
            subsections=subsections,
            order=1,
            status=ItemStatus.GENERATED
        )
        
        self.assertEqual(section.name, "Key Qualifications")
        self.assertEqual(section.content_type, "DYNAMIC")
        self.assertEqual(len(section.items), 2)
        self.assertEqual(len(section.subsections), 1)
        self.assertEqual(section.order, 1)
        self.assertEqual(section.status, ItemStatus.GENERATED)
        
        # Test serialization
        section_dict = section.to_dict()
        self.assertEqual(section_dict["status"], "generated")
        
        # Test deserialization
        restored_section = Section.from_dict(section_dict)
        self.assertEqual(restored_section.name, section.name)
        self.assertEqual(restored_section.status, section.status)

    def test_structured_cv_creation_and_operations(self):
        """Test creation and operations of StructuredCV class."""
        # Create sections
        key_quals_section = Section(
            name="Key Qualifications",
            content_type="DYNAMIC",
            items=[
                Item(content="Python expertise", item_type=ItemType.KEY_QUAL),
                Item(content="Machine learning experience", item_type=ItemType.KEY_QUAL)
            ],
            order=1
        )
        
        experience_section = Section(
            name="Professional Experience",
            content_type="DYNAMIC",
            subsections=[
                Subsection(
                    name="Senior Developer at ABC Corp",
                    items=[
                        Item(content="Led development team", item_type=ItemType.BULLET_POINT),
                        Item(content="Improved system performance", item_type=ItemType.BULLET_POINT)
                    ]
                )
            ],
            order=2
        )
        
        structured_cv = StructuredCV(
            sections=[key_quals_section, experience_section],
            metadata={"original_file": "resume.pdf", "timestamp": "2024-01-01"}
        )
        
        # Test basic properties
        self.assertEqual(len(structured_cv.sections), 2)
        self.assertEqual(structured_cv.metadata["original_file"], "resume.pdf")
        
        # Test section retrieval
        found_section = structured_cv.get_section_by_name("Key Qualifications")
        self.assertIsNotNone(found_section)
        self.assertEqual(found_section.name, "Key Qualifications")
        
        # Test item finding
        first_item = structured_cv.sections[0].items[0]
        found_item, found_section, found_subsection = structured_cv.find_item_by_id(first_item.id)
        self.assertIsNotNone(found_item)
        self.assertEqual(found_item.content, "Python expertise")
        self.assertIsNone(found_subsection)  # This item is directly in section
        
        # Test item content update
        success = structured_cv.update_item_content(first_item.id, "Advanced Python expertise")
        self.assertTrue(success)
        self.assertEqual(first_item.content, "Advanced Python expertise")
        
        # Test item status update
        success = structured_cv.update_item_status(first_item.id, ItemStatus.ACCEPTED)
        self.assertTrue(success)
        self.assertEqual(first_item.status, ItemStatus.ACCEPTED)
        
        # Test getting items by status
        accepted_items = structured_cv.get_items_by_status(ItemStatus.ACCEPTED)
        self.assertEqual(len(accepted_items), 1)
        self.assertEqual(accepted_items[0].content, "Advanced Python expertise")

    def test_structured_cv_serialization(self):
        """Test StructuredCV serialization and deserialization."""
        # Create a simple StructuredCV
        section = Section(
            name="Test Section",
            items=[Item(content="Test content", status=ItemStatus.GENERATED)]
        )
        
        structured_cv = StructuredCV(
            sections=[section],
            metadata={"test": "data"}
        )
        
        # Test serialization
        cv_dict = structured_cv.to_dict()
        self.assertEqual(len(cv_dict["sections"]), 1)
        self.assertEqual(cv_dict["metadata"]["test"], "data")
        
        # Test deserialization
        restored_cv = StructuredCV.from_dict(cv_dict)
        self.assertEqual(len(restored_cv.sections), 1)
        self.assertEqual(restored_cv.sections[0].name, "Test Section")
        self.assertEqual(restored_cv.metadata["test"], "data")

    def test_item_status_enum(self):
        """Test ItemStatus enum functionality."""
        # Test enum values
        self.assertEqual(ItemStatus.INITIAL.value, "initial")
        self.assertEqual(ItemStatus.GENERATED.value, "generated")
        self.assertEqual(ItemStatus.USER_EDITED.value, "user_edited")
        self.assertEqual(ItemStatus.TO_REGENERATE.value, "to_regenerate")
        self.assertEqual(ItemStatus.ACCEPTED.value, "accepted")
        self.assertEqual(ItemStatus.STATIC.value, "static")
        
        # Test string conversion
        self.assertEqual(str(ItemStatus.GENERATED), "generated")

    def test_item_type_enum(self):
        """Test ItemType enum functionality."""
        # Test enum values
        self.assertEqual(ItemType.BULLET_POINT.value, "bullet_point")
        self.assertEqual(ItemType.KEY_QUAL.value, "key_qual")
        self.assertEqual(ItemType.SUMMARY_PARAGRAPH.value, "summary_paragraph")
        self.assertEqual(ItemType.SECTION_TITLE.value, "section_title")
        
        # Test string conversion
        self.assertEqual(str(ItemType.BULLET_POINT), "bullet_point")

    def test_content_piece_creation(self):
        """Test ContentPiece creation and properties."""
        content_piece = ContentPiece(
            content="Test content piece",
            section_type="experience",
            piece_id="test-piece-123",
            status="generated"
        )
        
        self.assertEqual(content_piece.content, "Test content piece")
        self.assertEqual(content_piece.section_type, "experience")
        self.assertEqual(content_piece.piece_id, "test-piece-123")
        self.assertEqual(content_piece.status, "generated")


if __name__ == "__main__":
    unittest.main()
