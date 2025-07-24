"""Unit tests for AgentNodeFactory and WriterNodeFactory."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.orchestration.factories import AgentNodeFactory, WriterNodeFactory
from src.orchestration.state import GlobalState


class TestAgentNodeFactory:
    """Test cases for AgentNodeFactory."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = AsyncMock()
        agent._execute = AsyncMock(return_value={"test_output": "test_value"})
        return agent
    
    @pytest.fixture
    def mock_state(self):
        """Create a mock GlobalState."""
        return {
            "structured_cv": {"sections": []},
            "parsed_jd": {"job_title": "Test Job"},
            "current_item_id": "test_id",
            "research_data": {},
            "session_id": "test_session",
            "error_messages": []
        }
    
    def test_factory_initialization(self, mock_agent):
        """Test AgentNodeFactory initialization."""
        def mock_mapper(state):
            return {"test_input": "value"}
        
        def mock_updater(state, output):
            return {**state, "updated": True}
        
        factory = AgentNodeFactory(
            agent=mock_agent,
            input_mapper=mock_mapper,
            output_updater=mock_updater,
            node_name="Test Node"
        )
        
        assert factory.agent == mock_agent
        assert factory.input_mapper == mock_mapper
        assert factory.output_updater == mock_updater
        assert factory.node_name == "Test Node"
    
    @pytest.mark.asyncio
    async def test_execute_node_success(self, mock_agent, mock_state):
        """Test successful node execution."""
        def mock_mapper(state):
            return {"test_input": "value"}
        
        def mock_updater(state, output):
            return {**state, "updated": True, "last_executed_node": "TEST_NODE"}
        
        factory = AgentNodeFactory(
            agent=mock_agent,
            input_mapper=mock_mapper,
            output_updater=mock_updater,
            node_name="Test Node"
        )
        
        result = await factory.execute_node(mock_state)
        
        # Verify agent was called with mapped input
        mock_agent._execute.assert_called_once_with(test_input="value")
        
        # Verify state was updated
        assert result["updated"] is True
        assert result["last_executed_node"] == "TEST_NODE"
    
    @pytest.mark.asyncio
    async def test_execute_node_agent_error(self, mock_agent, mock_state):
        """Test node execution with agent error."""
        mock_agent._execute.side_effect = Exception("Agent failed")
        
        def mock_mapper(state):
            return {"test_input": "value"}
        
        def mock_updater(state, output):
            return {**state, "updated": True}
        
        factory = AgentNodeFactory(
            agent=mock_agent,
            input_mapper=mock_mapper,
            output_updater=mock_updater,
            node_name="Test Node"
        )
        
        result = await factory.execute_node(mock_state)
        
        # Verify error was handled
        assert "error_messages" in result
        assert "Test Node execution failed: Agent failed" in result["error_messages"]
        assert result["last_executed_node"] == "TEST_NODE"


class TestWriterNodeFactory:
    """Test cases for WriterNodeFactory."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = AsyncMock()
        agent._execute = AsyncMock(return_value={"generated_content": "test content"})
        return agent
    
    @pytest.fixture
    def mock_state_with_cv(self):
        """Create a mock GlobalState with structured CV."""
        return {
            "structured_cv": MagicMock(),
            "parsed_jd": {"job_title": "Test Job"},
            "current_item_id": "test_id",
            "research_data": {},
            "session_id": "test_session",
            "error_messages": []
        }
    
    def test_writer_factory_initialization(self, mock_agent):
        """Test WriterNodeFactory initialization."""
        def mock_mapper(state):
            return {"test_input": "value"}
        
        def mock_updater(state, output):
            return {**state, "updated": True}
        
        factory = WriterNodeFactory(
            agent=mock_agent,
            input_mapper=mock_mapper,
            output_updater=mock_updater,
            node_name="Test Writer",
            required_sections=["test_section"]
        )
        
        assert factory.agent == mock_agent
        assert factory.input_mapper == mock_mapper
        assert factory.output_updater == mock_updater
        assert factory.node_name == "Test Writer"
        assert factory.required_sections == ["test_section"]
    
    @pytest.mark.asyncio
    async def test_writer_execute_node_success(self, mock_agent, mock_state_with_cv):
        """Test successful writer node execution."""
        def mock_mapper(state):
            return {"test_input": "value"}
        
        def mock_updater(state, output):
            return {**state, "updated": True, "last_executed_node": "TEST_WRITER"}
        
        factory = WriterNodeFactory(
            agent=mock_agent,
            input_mapper=mock_mapper,
            output_updater=mock_updater,
            node_name="Test Writer",
            required_sections=["test_section"]
        )
        
        result = await factory.execute_node(mock_state_with_cv)
        
        # Verify agent was called
        mock_agent._execute.assert_called_once_with(test_input="value")
        
        # Verify state was updated
        assert result["updated"] is True
        assert result["last_executed_node"] == "TEST_WRITER"
    
    @pytest.mark.asyncio
    async def test_writer_missing_structured_cv(self, mock_agent):
        """Test writer node execution with missing structured_cv."""
        state_without_cv = {
            "parsed_jd": {"job_title": "Test Job"},
            "current_item_id": "test_id",
            "error_messages": []
        }
        
        def mock_mapper(state):
            return {"test_input": "value"}
        
        def mock_updater(state, output):
            return {**state, "updated": True}
        
        factory = WriterNodeFactory(
            agent=mock_agent,
            input_mapper=mock_mapper,
            output_updater=mock_updater,
            node_name="Test Writer",
            required_sections=["test_section"]
        )
        
        result = await factory.execute_node(state_without_cv)
        
        # Verify error was handled
        assert "error_messages" in result
        assert "Missing required field: structured_cv" in result["error_messages"]
        assert result["last_executed_node"] == "TEST_WRITER"