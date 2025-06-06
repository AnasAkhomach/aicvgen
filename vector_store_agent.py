from typing import List, Any, Dict
from agent_base import AgentBase
from vector_db import VectorDB
from state_manager import VectorStoreConfig, AgentIO, SkillEntry, ExperienceEntry


class VectorStoreAgent(AgentBase):
    """
    Agent for interacting with a vector store.
    """

    def __init__(
        self,
        name: str,
        description: str,
        model: Any,
        input_schema: AgentIO,
        output_schema: AgentIO,
        vector_db: VectorDB,
    ):
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
        super().__init__(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
        )
        self.vector_db = vector_db
        self.model = model  # Store the model
        self.last_saved_state = None

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

    def add_experiences(self, experiences: List[ExperienceEntry]) -> None:
        """
        Adds a list of experience entries to the vector store.

        Args:
            experiences: List of experience entries to add
        """
        if not experiences:
            return

        for experience in experiences:
            self.run_add_item(experience)

    def search_experiences(self, keywords: List[str], skills: List[str], k=5) -> List[Any]:
        """
        Search for relevant experiences based on job keywords and required skills.

        Args:
            keywords: List of job keywords
            skills: List of required skills
            k: Number of results to return per query

        Returns:
            List of relevant experience entries
        """
        results = []

        # Search by keywords
        if keywords:
            keyword_query = " ".join(keywords)
            keyword_results = self.search(keyword_query, k)
            results.extend(keyword_results)

        # Search by skills
        if skills:
            skills_query = " ".join(skills)
            skills_results = self.search(skills_query, k)
            results.extend(skills_results)

        # Remove duplicates while preserving order
        unique_results = []
        seen_ids = set()

        for result in results:
            if hasattr(result, "id") and result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)

        return unique_results

    def save_state(self, state: Dict) -> None:
        """
        Saves the current state for later retrieval.

        Args:
            state: The state dictionary to save
        """
        self.last_saved_state = state.copy() if state else None

    def get_last_saved_state(self, thread_id=None) -> Dict:
        """
        Retrieves the last saved state.

        Args:
            thread_id: Optional thread identifier (ignored in this implementation)

        Returns:
            The last saved state dictionary or None if no state was saved
        """
        # Ignore thread_id parameter as we're just using a simple in-memory state
        return self.last_saved_state
