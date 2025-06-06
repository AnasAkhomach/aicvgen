import unittest

# Import directly from the files
from state_manager import JobDescriptionData


class TestJobDescriptionData(unittest.TestCase):
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

    def test_to_dict_method(self):
        """Test the to_dict method of JobDescriptionData class."""
        # Create a JobDescriptionData instance
        job_data = JobDescriptionData(
            raw_text="Job description text",
            skills=["Python", "JavaScript"],
            experience_level="Mid-Senior",
            responsibilities=["Build web applications", "Lead team"],
            industry_terms=["SaaS", "Agile"],
            company_values=["Innovation", "Teamwork"],
            error="Test error",
        )

        # Convert to dictionary
        job_dict = job_data.to_dict()

        # Verify it's a dictionary
        self.assertIsInstance(job_dict, dict)

        # Verify all attributes are correctly included
        self.assertEqual(job_dict["raw_text"], "Job description text")
        self.assertEqual(job_dict["skills"], ["Python", "JavaScript"])
        self.assertEqual(job_dict["experience_level"], "Mid-Senior")
        self.assertEqual(
            job_dict["responsibilities"], ["Build web applications", "Lead team"]
        )
        self.assertEqual(job_dict["industry_terms"], ["SaaS", "Agile"])
        self.assertEqual(job_dict["company_values"], ["Innovation", "Teamwork"])
        self.assertEqual(job_dict["error"], "Test error")

        # Verify dictionary access works
        self.assertEqual(job_dict.get("skills"), ["Python", "JavaScript"])
        self.assertEqual(job_dict.get("non_existent_key", "default"), "default")


if __name__ == "__main__":
    unittest.main()
