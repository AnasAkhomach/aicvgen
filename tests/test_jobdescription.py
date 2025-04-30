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

if __name__ == '__main__':
    unittest.main() 