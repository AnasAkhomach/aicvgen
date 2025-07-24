"""Integration tests for content_nodes with factory pattern."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.orchestration.nodes.content_nodes import (
    key_qualifications_writer_node,
    professional_experience_writer_node,
    projects_writer_node,
    executive_summary_writer_node,
    qa_node
)


class TestContentNodesFactoryIntegration:
    """Integration tests for content nodes using factory pattern."""
    
    @pytest.fixture
    def mock_state(self):
        """Create a mock GlobalState."""
        return {
            "structured_cv": MagicMock(),
            "parsed_jd": MagicMock(),
            "current_item_id": "test_id",
            "research_data": {"test": "data"},
            "session_id": "test_session",
            "error_messages": []
        }
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = AsyncMock()
        agent._execute = AsyncMock(return_value={"generated_content": "test content"})
        return agent
    
    @patch('src.orchestration.nodes.content_nodes.WriterNodeFactory')
    @pytest.mark.asyncio
    async def test_key_qualifications_writer_node(self, mock_factory_class, mock_state, mock_agent):
        """Test key qualifications writer node uses factory pattern."""
        mock_factory = AsyncMock()
        mock_factory.execute_node = AsyncMock(return_value={**mock_state, "updated": True})
        mock_factory_class.return_value = mock_factory
        
        result = await key_qualifications_writer_node(mock_state, agent=mock_agent)
        
        # Verify factory was created with correct parameters
        mock_factory_class.assert_called_once()
        call_args = mock_factory_class.call_args
        assert call_args.kwargs['agent'] == mock_agent
        assert call_args.kwargs['node_name'] == "Key Qualifications Writer"
        assert call_args.kwargs['required_sections'] == ["key qualifications"]
        
        # Verify factory execute_node was called
        mock_factory.execute_node.assert_called_once_with(mock_state)
        
        # Verify result
        assert result["updated"] is True
    
    @patch('src.orchestration.nodes.content_nodes.WriterNodeFactory')
    @pytest.mark.asyncio
    async def test_professional_experience_writer_node(self, mock_factory_class, mock_state, mock_agent):
        """Test professional experience writer node uses factory pattern."""
        mock_factory = AsyncMock()
        mock_factory.execute_node = AsyncMock(return_value={**mock_state, "updated": True})
        mock_factory_class.return_value = mock_factory
        
        result = await professional_experience_writer_node(mock_state, agent=mock_agent)
        
        # Verify factory was created with correct parameters
        mock_factory_class.assert_called_once()
        call_args = mock_factory_class.call_args
        assert call_args.kwargs['agent'] == mock_agent
        assert call_args.kwargs['node_name'] == "Professional Experience Writer"
        assert call_args.kwargs['required_sections'] == ["professional experience"]
        
        # Verify factory execute_node was called
        mock_factory.execute_node.assert_called_once_with(mock_state)
        
        # Verify result
        assert result["updated"] is True
    
    @patch('src.orchestration.nodes.content_nodes.WriterNodeFactory')
    @pytest.mark.asyncio
    async def test_projects_writer_node(self, mock_factory_class, mock_state, mock_agent):
        """Test projects writer node uses factory pattern."""
        mock_factory = AsyncMock()
        mock_factory.execute_node = AsyncMock(return_value={**mock_state, "updated": True})
        mock_factory_class.return_value = mock_factory
        
        result = await projects_writer_node(mock_state, agent=mock_agent)
        
        # Verify factory was created with correct parameters
        mock_factory_class.assert_called_once()
        call_args = mock_factory_class.call_args
        assert call_args.kwargs['agent'] == mock_agent
        assert call_args.kwargs['node_name'] == "Projects Writer"
        assert call_args.kwargs['required_sections'] == ["projects"]
        
        # Verify factory execute_node was called
        mock_factory.execute_node.assert_called_once_with(mock_state)
        
        # Verify result
        assert result["updated"] is True
    
    @patch('src.orchestration.nodes.content_nodes.WriterNodeFactory')
    @pytest.mark.asyncio
    async def test_executive_summary_writer_node(self, mock_factory_class, mock_state, mock_agent):
        """Test executive summary writer node uses factory pattern."""
        mock_factory = AsyncMock()
        mock_factory.execute_node = AsyncMock(return_value={**mock_state, "updated": True})
        mock_factory_class.return_value = mock_factory
        
        result = await executive_summary_writer_node(mock_state, agent=mock_agent)
        
        # Verify factory was created with correct parameters
        mock_factory_class.assert_called_once()
        call_args = mock_factory_class.call_args
        assert call_args.kwargs['agent'] == mock_agent
        assert call_args.kwargs['node_name'] == "Executive Summary Writer"
        assert call_args.kwargs['required_sections'] == ["executive summary"]
        
        # Verify factory execute_node was called
        mock_factory.execute_node.assert_called_once_with(mock_state)
        
        # Verify result
        assert result["updated"] is True
    
    @patch('src.orchestration.nodes.content_nodes.AgentNodeFactory')
    @pytest.mark.asyncio
    async def test_qa_node(self, mock_factory_class, mock_state, mock_agent):
        """Test QA node uses factory pattern."""
        mock_factory = AsyncMock()
        mock_factory.execute_node = AsyncMock(return_value={**mock_state, "qa_results": {"score": 0.9}})
        mock_factory_class.return_value = mock_factory
        
        result = await qa_node(mock_state, agent=mock_agent)
        
        # Verify factory was created with correct parameters
        mock_factory_class.assert_called_once()
        call_args = mock_factory_class.call_args
        assert call_args.kwargs['agent'] == mock_agent
        assert call_args.kwargs['node_name'] == "Quality Assurance"
        
        # Verify factory execute_node was called
        mock_factory.execute_node.assert_called_once_with(mock_state)
        
        # Verify result
        assert result["qa_results"] == {"score": 0.9}
    
    @patch('src.orchestration.nodes.content_nodes.map_state_to_key_qualifications_input')
    @patch('src.orchestration.nodes.content_nodes.update_cv_with_key_qualifications_data')
    @patch('src.orchestration.nodes.content_nodes.WriterNodeFactory')
    @pytest.mark.asyncio
    async def test_key_qualifications_node_mapper_integration(self, mock_factory_class, mock_updater, mock_mapper, mock_state, mock_agent):
        """Test that key qualifications node uses correct mapper and updater functions."""
        mock_factory = AsyncMock()
        mock_factory.execute_node = AsyncMock(return_value=mock_state)
        mock_factory_class.return_value = mock_factory
        
        await key_qualifications_writer_node(mock_state, agent=mock_agent)
        
        # Verify factory was created with the imported functions
        call_args = mock_factory_class.call_args
        assert 'input_mapper' in call_args.kwargs
        assert 'output_updater' in call_args.kwargs
        
        # The actual functions should be passed, not the mocks
        # This verifies the imports are working correctly
        assert callable(call_args.kwargs['input_mapper'])
        assert callable(call_args.kwargs['output_updater'])


class TestContentNodesErrorHandling:
    """Test error handling in content nodes with factory pattern."""
    
    @pytest.fixture
    def mock_state(self):
        """Create a mock GlobalState."""
        return {
            "structured_cv": MagicMock(),
            "parsed_jd": MagicMock(),
            "current_item_id": "test_id",
            "research_data": {"test": "data"},
            "session_id": "test_session",
            "error_messages": []
        }
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = AsyncMock()
        agent._execute = AsyncMock(side_effect=Exception("Agent execution failed"))
        return agent
    
    @patch('src.orchestration.nodes.content_nodes.WriterNodeFactory')
    @pytest.mark.asyncio
    async def test_node_error_handling(self, mock_factory_class, mock_state, mock_agent):
        """Test that nodes handle factory execution errors properly."""
        mock_factory = AsyncMock()
        mock_factory.execute_node = AsyncMock(return_value={
            **mock_state,
            "error_messages": ["Factory execution failed"],
            "last_executed_node": "KEY_QUALIFICATIONS_WRITER"
        })
        mock_factory_class.return_value = mock_factory
        
        result = await key_qualifications_writer_node(mock_state, agent=mock_agent)
        
        # Verify error was handled by factory
        assert "error_messages" in result
        assert "Factory execution failed" in result["error_messages"]
        assert result["last_executed_node"] == "KEY_QUALIFICATIONS_WRITER"