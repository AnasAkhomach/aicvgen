import pytest
import asyncio
from unittest.mock import Mock
from src.core.state_manager import StateManager
from src.orchestration.state import AgentState
from src.services.llm_service import EnhancedLLMService
from src.services.llm_client import LLMClient
from src.services.llm_retry_handler import LLMRetryHandler
from src.models.data_models import StructuredCV


@pytest.mark.asyncio
async def test_state_manager_async_persistence(tmp_path):
    """Integration: StateManager async save/load roundtrip."""
    llm_client = LLMClient(llm=Mock())
    retry_handler = LLMRetryHandler(llm_client)
    llm_service = EnhancedLLMService(llm_client, retry_handler)
    state_manager = StateManager()
    state = AgentState(structured_cv=StructuredCV())
    state_manager.set_structured_cv(state.structured_cv)
    await state_manager.save_state()
    await state_manager.load_state()
    loaded_cv = state_manager.get_structured_cv()
    assert isinstance(loaded_cv, StructuredCV)
    assert loaded_cv == state.structured_cv
