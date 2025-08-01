"""Key Qualifications Updater Agent.

This agent is responsible for taking the generated key qualifications list
and updating the structured CV with them. It follows the LangGraph pattern
of single responsibility per agent.
"""

from typing import Any, List
from pydantic import ValidationError

from src.agents.agent_base import AgentBase
from src.constants.agent_constants import AgentConstants
from src.error_handling.exceptions import AgentExecutionError
from src.models.cv_models import StructuredCV, Item, ItemStatus, ItemType
from src.utils.node_validation import ensure_pydantic_model


class KeyQualificationsUpdaterAgent(AgentBase):
    """Agent responsible for updating the structured CV with generated key qualifications.

    This agent consumes the list of key qualifications generated by KeyQualificationsWriterAgent
    and properly integrates them into the structured CV following the data model patterns.
    """

    def __init__(self, session_id: str, name: str = "KeyQualificationsUpdaterAgent"):
        """Initialize the KeyQualificationsUpdaterAgent.

        Args:
            name: The name of the agent
            session_id: The session identifier
        """
        super().__init__(
            name=name,
            description="Agent responsible for updating the structured CV with generated key qualifications",
            session_id=session_id,
        )
        self.logger.info(f"Initialized {self.name}")

    def _validate_inputs(self, kwargs: dict[str, Any]) -> None:
        """Validate required inputs for the agent.

        Args:
            kwargs: Input arguments containing structured_cv and generated_key_qualifications.

        Raises:
            AgentExecutionError: If required inputs are missing or invalid.
        """
        required_fields = ["structured_cv", "generated_key_qualifications"]
        for field in required_fields:
            if field not in kwargs or kwargs[field] is None:
                raise AgentExecutionError(self.name, f"Missing required input: {field}")

        # Pydantic validation for structured_cv is now handled by the decorator
        # At this point, structured_cv has already been converted to StructuredCV by the decorator

        # Validate generated_key_qualifications type
        if not isinstance(kwargs["generated_key_qualifications"], list):
            raise AgentExecutionError(
                self.name, "generated_key_qualifications must be a list"
            )

        if not kwargs["generated_key_qualifications"]:
            raise AgentExecutionError(
                self.name, "generated_key_qualifications cannot be empty"
            )

    @ensure_pydantic_model(
        ("structured_cv", StructuredCV),
    )
    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the key qualifications update logic.

        Takes the generated key qualifications and updates the structured CV.

        Args:
            **kwargs: Must contain 'structured_cv' and 'generated_key_qualifications'.

        Returns:
            dict containing the updated structured_cv.
        """
        try:
            # Validate inputs
            self._validate_inputs(kwargs)

            structured_cv: StructuredCV = kwargs["structured_cv"]
            generated_qualifications: List[str] = kwargs["generated_key_qualifications"]

            self.update_progress(
                AgentConstants.PROGRESS_MAIN_PROCESSING,
                "Updating CV with generated Key Qualifications.",
            )

            # Find the Key Qualifications section
            qual_section = None
            for section in structured_cv.sections:
                if section.name == "Key Qualifications":
                    qual_section = section
                    break

            if not qual_section:
                return {
                    "error_messages": [
                        "Key Qualifications section not found in structured_cv. It should be pre-initialized."
                    ]
                }

            # Clear existing items and add new ones
            qual_section.items = [
                Item(
                    content=qual,
                    status=ItemStatus.GENERATED,
                    item_type=ItemType.KEY_QUALIFICATION,
                )
                for qual in generated_qualifications
            ]

            self.logger.info(
                f"Updated Key Qualifications section with {len(generated_qualifications)} items"
            )

            self.update_progress(
                AgentConstants.PROGRESS_COMPLETE,
                "Key Qualifications update completed successfully.",
            )

            return {
                "structured_cv": structured_cv,
                "current_item_id": "key_qualifications_section",
            }

        except AgentExecutionError as e:
            self.logger.error(f"Agent execution error in {self.name}: {str(e)}")
            return {"error_messages": [str(e)]}
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            self.logger.error(f"Error updating Key Qualifications: {str(e)}")
            return {"error_messages": [f"Error updating Key Qualifications: {str(e)}"]}
        except Exception as e:
            self.logger.error(
                f"Unexpected error in {self.name}: {str(e)}", exc_info=True
            )
            return {
                "error_messages": [
                    f"Unexpected error during Key Qualifications update: {str(e)}"
                ]
            }
