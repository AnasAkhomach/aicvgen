from src.agents.agent_base import AgentBase
from src.core.state_manager import AgentIO
from typing import Any, Dict, List


class ToolsAgent(AgentBase):
    """
    Agent responsible for providing tool access for content processing and validation.
    """

    def __init__(self, name: str, description: str):
        """
        Initializes the ToolsAgent.

        Args:
            name: The name of the agent.
            description: A description of the agent.
        """
        # Define input and output schemas for the agent itself, though its methods will have specific schemas
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                input=Any,  # This agent's overall input might vary depending on the tool being used
                output=Any,  # This agent's overall output might vary
                description="Input for a specific tool function.",
            ),
            output_schema=AgentIO(
                input=Any,
                output=Any,
                description="Output from a specific tool function.",
            ),
        )

    def run(self, input_data: Any) -> Dict[str, Any]:
        """Run the tools agent."""
        raise NotImplementedError("ToolsAgent.run() must be implemented by subclasses")
    
    async def run_async(self, input_data: Any, context: 'AgentExecutionContext') -> 'AgentResult':
        """Async run method for consistency with enhanced agent interface."""
        from .agent_base import AgentResult
        from src.models.validation_schemas import validate_agent_input, ValidationError
        
        try:
            # Validate input data using Pydantic schemas
            try:
                validated_input = validate_agent_input('tools', input_data)
                # Convert validated Pydantic model back to dict for processing
                input_data = validated_input.model_dump()
                logger.info("Input validation passed for ToolsAgent")
            except ValidationError as ve:
                logger.error(f"Input validation failed for ToolsAgent: {ve.message}")
                return AgentResult(
                    success=False,
                    output_data={"error": f"Input validation failed: {ve.message}"},
                    confidence_score=0.0,
                    error_message=f"Input validation failed: {ve.message}",
                    metadata={"agent_type": "tools", "validation_error": True}
                )
            except Exception as e:
                logger.error(f"Input validation error for ToolsAgent: {str(e)}")
                return AgentResult(
                    success=False,
                    output_data={"error": f"Input validation error: {str(e)}"},
                    confidence_score=0.0,
                    error_message=f"Input validation error: {str(e)}",
                    metadata={"agent_type": "tools", "validation_error": True}
                )
            
            # Use the existing run method for the actual processing
            result = self.run(input_data)
            
            return AgentResult(
                success=True,
                output_data=result,
                confidence_score=1.0,
                metadata={"agent_type": "tools"}
            )
            
        except NotImplementedError as e:
            return AgentResult(
                success=False,
                output_data={"error": "ToolsAgent not implemented"},
                confidence_score=0.0,
                error_message=str(e),
                metadata={"agent_type": "tools"}
            )
        except Exception as e:
            return AgentResult(
                success=False,
                output_data={"error": "Tools agent execution failed"},
                confidence_score=0.0,
                error_message=str(e),
                metadata={"agent_type": "tools"}
            )

    def format_text(self, text: str, format_type: str = "markdown") -> str:
        """
        Simulates formatting text using a tool.

        Args:
            text: The text to format.
            format_type: The desired format (e.g., "markdown", "latex").

        Returns:
            The formatted text.
        """
        print(f"Simulating formatting text to {format_type}...")
        # Placeholder formatting logic
        if format_type == "markdown":
            return f"Formatted (Markdown): {text}"
        elif format_type == "latex":
            return f"Formatted (LaTeX): {text}"
        else:
            return f"Formatted ( desconocida): {text}"

    def validate_content(self, content: str, requirements: List[str]) -> Dict[str, Any]:
        """
        Simulates validating content against requirements.

        Args:
            content: The content to validate.
            requirements: A list of requirements to check against.

        Returns:
            A dictionary with validation results.
        """
        print(f"validate_content called with content: {content} and requirements: {requirements}")
        print("Simulating content validation...")
        # Placeholder validation logic
        validation_results = {
            "is_valid": True,  # Assume valid for simulation
            "feedback": "Content looks good (simulated).",
            "matched_requirements": [],
            "missing_requirements": [],
        }
        # Simulate checking for requirements (simple keyword match)
        for req in requirements:
            if req.lower() in content.lower():
                validation_results["matched_requirements"].append(req)
            else:
                validation_results["missing_requirements"].append(req)
                validation_results["is_valid"] = (
                    False  # Mark as invalid if any requirement is missing
                )
                validation_results["feedback"] = "Content is missing some requirements (simulated)."

        return validation_results

    # Add other tool-like methods here (e.g., grammar check, keyword optimization)
