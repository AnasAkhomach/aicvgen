"""Factory classes for creating reusable node components.

This module provides factory classes that abstract common agent logic
into reusable components, simplifying writer nodes in content_nodes.py.
"""

from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
from src.models.cv_models import StructuredCV
from src.config.logging_config import get_logger

if TYPE_CHECKING:
    from src.orchestration.state import GlobalState

logger = get_logger(__name__)


class AgentNodeFactory:
    """Factory for creating standardized agent nodes with input mapping and output updating.
    
    This factory abstracts the common pattern of:
    1. Mapping state to agent input
    2. Executing the agent
    3. Updating the CV with agent output
    4. Handling errors consistently
    """
    
    def __init__(
        self,
        agent: Any,
        input_mapper: Callable[[Dict[str, Any]], Any],
        output_updater: Callable[[StructuredCV, Dict[str, Any], Optional[str]], StructuredCV],
        node_name: str
    ):
        """Initialize the AgentNodeFactory.
        
        Args:
            agent: The agent instance to execute
            input_mapper: Function to map state to agent input
            output_updater: Function to update CV with agent output
            node_name: Name of the node for logging and error handling
        """
        self.agent = agent
        self.input_mapper = input_mapper
        self.output_updater = output_updater
        self.node_name = node_name
    
    async def execute_node(self, state: "GlobalState") -> "GlobalState":
        """Execute the agent node with standardized error handling.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with agent results
        """
        logger.info(f"Starting {self.node_name} node")
        
        try:
            # 1. Map state to agent input
            agent_input = self.input_mapper(state)
            
            # 2. Execute the agent
            agent_output = await self.agent._execute(**agent_input.model_dump())
            
            # 3. Use the specific updater to modify the CV
            current_cv = state["structured_cv"]
            item_id_to_update = state.get("current_item_id")
            updated_cv = self.output_updater(current_cv, agent_output, item_id_to_update)
            
            # 4. Return the updated state dictionary
            updated_state = {
                **state,
                "structured_cv": updated_cv,
                "last_executed_node": self.node_name.upper().replace(" ", "_")
            }
            
            logger.info(f"{self.node_name} node completed successfully")
            return updated_state
            
        except Exception as exc:
            logger.error(f"Error in {self.node_name} node: {exc}")
            error_messages = list(state.get("error_messages", []))
            error_messages.append(f"{self.node_name} execution failed: {str(exc)}")
            
            return {
                **state,
                "error_messages": error_messages,
                "last_executed_node": self.node_name.upper().replace(" ", "_")
            }


class WriterNodeFactory:
    """Specialized factory for writer nodes with additional validation.
    
    This factory extends AgentNodeFactory with writer-specific logic
    such as validation of required state fields and structured CV sections.
    """
    
    def __init__(
        self,
        agent: Any,
        input_mapper: Callable[[Dict[str, Any]], Any],
        output_updater: Callable[[StructuredCV, Dict[str, Any], Optional[str]], StructuredCV],
        node_name: str,
        required_sections: Optional[list] = None
    ):
        """Initialize the WriterNodeFactory.
        
        Args:
            agent: The agent instance to execute
            input_mapper: Function to map state to agent input
            output_updater: Function to update CV with agent output
            node_name: Name of the node for logging and error handling
            required_sections: List of required CV sections for validation
        """
        self.agent_factory = AgentNodeFactory(agent, input_mapper, output_updater, node_name)
        self.required_sections = required_sections or []
        self.node_name = node_name
    
    def _validate_state(self, state: "GlobalState") -> bool:
        """Validate that the state contains required fields and sections.
        
        Args:
            state: Current workflow state
            
        Returns:
            True if state is valid, False otherwise
        """
        # Check required state fields
        required_fields = ["structured_cv", "parsed_jd"]
        for field in required_fields:
            if field not in state or state[field] is None:
                logger.error(f"{self.node_name}: Missing required field '{field}' in state")
                return False
        
        # Check required CV sections
        if self.required_sections:
            structured_cv = state["structured_cv"]
            existing_sections = [section.name.lower() for section in structured_cv.sections]
            
            for required_section in self.required_sections:
                if required_section.lower() not in existing_sections:
                    logger.error(f"{self.node_name}: Missing required section '{required_section}' in CV")
                    return False
        
        return True
    
    async def execute_node(self, state: "GlobalState") -> "GlobalState":
        """Execute the writer node with validation and standardized error handling.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with agent results
        """
        # Validate state before execution
        if not self._validate_state(state):
            error_messages = list(state.get("error_messages", []))
            
            # Check for missing structured_cv specifically
            if "structured_cv" not in state or state["structured_cv"] is None:
                error_messages.append("Missing required field: structured_cv")
            else:
                error_messages.append(f"{self.node_name}: State validation failed")
            
            return {
                **state,
                "error_messages": error_messages,
                "last_executed_node": self.node_name.upper().replace(" ", "_")
            }
        
        # Delegate to the base agent factory
        return await self.agent_factory.execute_node(state)