from typing import List, Any
from agent_base import AgentBase
from vector_db import VectorDB
from state_manager import VectorStoreConfig, AgentIO, SkillEntry, ExperienceEntry

class VectorStoreAgent(AgentBase):
    """
    Agent for interacting with a vector store.
    """

    def __init__(self, name: str, description: str, model: Any, input_schema: AgentIO, output_schema: AgentIO, vector_db: VectorDB):
        """
        Initializes the vector store agent.

        Args:
            name: The name of the agent.
            description: A description of the agent.
            model: LLM model instance
            input_schema: input schema
            output_schema: output schema
            vector_db: the VectorDB
        """
        super().__init__(name=name, description=description, input_schema=input_schema, output_schema=output_schema)
        self.vector_db = vector_db
        self.model = model # Store the model

    def run_add_item(self, item: Any, text: str = None) -> None:
        """
        Adds an item to the vector database.

        Args:
            item: The item to add (e.g., SkillEntry, ExperienceEntry).
            text: The text to use for generating the embedding. If None, it will use item.text.
        """
        self.vector_db.add_item(item, text)

    def run(self, input: Any) -> Any:
        raise NotImplementedError("The run method is not implemented yet. Use run_add_item.")

    def search(self, query: str, k=3) -> List[Any]:
        """
        Searches for the k most similar items to the query.

        Args:
            query: The query to search for.
            k: The number of results to return.

        Returns:
            List of similar items.
        """
        return self.vector_db.search(query, k)