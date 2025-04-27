from abc import ABC, abstractmethod
from typing import Any
from state_manager import AgentIO


class AgentBase(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name: str, description: str, input_schema: AgentIO, output_schema: AgentIO):
        """Initializes the AgentBase with the given attributes.

        Args:
            name: The name of the agent.
            description: The description of the agent.
            input_schema: The input schema of the agent.
            output_schema: The output schema of the agent.
        """        
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_schema = output_schema

    @abstractmethod
    def run(self, input: Any) -> Any:
        """Abstract method to be implemented by each agent.

        """
        raise NotImplementedError