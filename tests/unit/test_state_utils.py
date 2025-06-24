"""Unit tests for state_utils.py."""

import unittest
from unittest.mock import patch
from src.utils.state_utils import create_initial_agent_state
from src.models.data_models import JobDescriptionData, StructuredCV, MetadataModel
from src.orchestration.state import AgentState


class TestStateUtils(unittest.TestCase):
    """Test suite for state utility functions."""

    def test_create_initial_agent_state(self):
        """Verify that create_initial_agent_state correctly initializes AgentState."""
        # Arrange
        job_desc_raw = "We are looking for a Python developer."
        cv_text = "This is my CV."
        start_from_scratch = False

        # Act
        agent_state = create_initial_agent_state(
            job_description_raw=job_desc_raw,
            cv_text=cv_text,
            start_from_scratch=start_from_scratch,
        )

        # Assert
        self.assertIsInstance(agent_state, AgentState)
        self.assertIsInstance(agent_state.job_description_data, JobDescriptionData)
        self.assertEqual(agent_state.job_description_data.raw_text, job_desc_raw)

        self.assertIsInstance(agent_state.structured_cv, StructuredCV)
        self.assertIsInstance(agent_state.structured_cv.metadata, MetadataModel)
        self.assertEqual(
            agent_state.structured_cv.metadata.extra["original_cv_text"], cv_text
        )
        self.assertEqual(
            agent_state.structured_cv.metadata.extra["start_from_scratch"],
            start_from_scratch,
        )

        # Verify other fields are initialized with defaults
        self.assertIsNone(agent_state.user_feedback)
        self.assertEqual(agent_state.error_messages, [])
        self.assertTrue(agent_state.is_initial_generation)


if __name__ == "__main__":
    unittest.main()
