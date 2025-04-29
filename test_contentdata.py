import unittest
from state_manager import ContentData

class TestContentData(unittest.TestCase):
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
        # Our fixed ContentData class properly initializes all fields with defaults
        self.assertEqual(content_partial["experience_bullets"], [])
        self.assertEqual(content_partial["skills_section"], "")
        self.assertEqual(content_partial["projects"], [])
        self.assertEqual(content_partial["other_content"], {})

if __name__ == '__main__':
    unittest.main() 