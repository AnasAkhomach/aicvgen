"""Unit tests for the abstract agent base class interface."""

import pytest
from abc import ABC
from unittest.mock import AsyncMock, MagicMock
from src.agents.agent_base import EnhancedAgentBase, AgentResult, AgentIO
from src.orchestration.state import AgentState
from src.models.data_models import JobDescriptionData, StructuredCV
from src.services.llm_service import EnhancedLLMService


class TestEnhancedAgentBase:
    """Test cases for the EnhancedAgentBase abstract class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that EnhancedAgentBase cannot be instantiated directly."""
        # Arrange & Act & Assert
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            EnhancedAgentBase(name="test", description="test", llm_service=MagicMock())
    
    def test_concrete_class_must_implement_run_as_node(self):
        """Test that concrete classes must implement run_as_node method."""
        # Arrange
        class IncompleteAgent(EnhancedAgentBase):
            """Agent that doesn't implement run_as_node."""
            pass
        
        # Act & Assert
        with pytest.raises(TypeError, match="Can't instantiate abstract class.*run_as_node"):
            IncompleteAgent(name="incomplete", description="test", llm_service=MagicMock())
    
    def test_concrete_class_with_run_as_node_can_be_instantiated(self):
        """Test that concrete classes implementing run_as_node can be instantiated."""
        # Arrange
        class CompleteAgent(EnhancedAgentBase):
            """Agent that properly implements both abstract methods."""
            
            async def run_as_node(self, state: AgentState) -> dict:
                return {"test_field": "test_value"}
            
            async def run_async(self, input_data, context):
                return AgentResult(success=True, data={"test": "data"}, metadata={})
        
        # Act
        agent = CompleteAgent(
            name="complete", 
            description="test", 
            input_schema=AgentIO(description="test input", schema={}), 
            output_schema=AgentIO(description="test output", schema={})
        )
        
        # Assert
        assert agent.name == "complete"
        assert agent.description == "test"
        assert hasattr(agent, 'run_as_node')
    
    @pytest.mark.asyncio
    async def test_run_as_node_signature_enforcement(self):
        """Test that run_as_node has the correct signature."""
        # Arrange
        class TestAgent(EnhancedAgentBase):
            async def run_as_node(self, state: AgentState) -> dict:
                return {"processed": True}
            
            async def run_async(self, input_data, context):
                return AgentResult(success=True, data={"processed": True}, metadata={})
        
        agent = TestAgent(
            name="test", 
            description="test", 
            input_schema=AgentIO(description="test input", schema={}), 
            output_schema=AgentIO(description="test output", schema={})
        )
        test_state = AgentState(
            structured_cv=StructuredCV(),
            job_description_data=JobDescriptionData(raw_text="test job")
        )
        
        # Act
        result = await agent.run_as_node(test_state)
        
        # Assert
        assert isinstance(result, dict)
        assert result["processed"] is True
    
    def test_inheritance_from_abc(self):
        """Test that EnhancedAgentBase properly inherits from ABC."""
        # Act & Assert
        assert issubclass(EnhancedAgentBase, ABC)
        assert hasattr(EnhancedAgentBase, '__abstractmethods__')
        assert 'run_as_node' in EnhancedAgentBase.__abstractmethods__


class TestAgentInterfaceCompliance:
    """Test that existing agents comply with the standardized interface."""
    
    def test_parser_agent_compliance(self):
        """Test that ParserAgent implements the required interface."""
        from src.agents.parser_agent import ParserAgent
        
        # Arrange
        mock_llm = MagicMock(spec=EnhancedLLMService)
        
        # Act
        agent = ParserAgent(name="parser", description="test", llm_service=mock_llm)
        
        # Assert
        assert isinstance(agent, EnhancedAgentBase)
        assert hasattr(agent, 'run_as_node')
        assert callable(getattr(agent, 'run_as_node'))
    
    def test_content_writer_agent_compliance(self):
        """Test that EnhancedContentWriterAgent implements the required interface."""
        from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
        
        # Act
        agent = EnhancedContentWriterAgent()
        
        # Assert
        assert isinstance(agent, EnhancedAgentBase)
        assert hasattr(agent, 'run_as_node')
        assert callable(getattr(agent, 'run_as_node'))
    
    def test_qa_agent_compliance(self):
        """Test that QualityAssuranceAgent implements the required interface."""
        from src.agents.quality_assurance_agent import QualityAssuranceAgent
        
        # Arrange
        mock_llm = MagicMock(spec=EnhancedLLMService)
        
        # Act
        agent = QualityAssuranceAgent(name="qa", description="test", llm_service=mock_llm)
        
        # Assert
        assert isinstance(agent, EnhancedAgentBase)
        assert hasattr(agent, 'run_as_node')
        assert callable(getattr(agent, 'run_as_node'))
    
    def test_research_agent_compliance(self):
        """Test that ResearchAgent implements the required interface."""
        from src.agents.research_agent import ResearchAgent
        
        # Arrange
        mock_llm = MagicMock(spec=EnhancedLLMService)
        mock_vector_db = MagicMock()
        
        # Act
        agent = ResearchAgent(
            name="research", 
            description="test", 
            llm_service=mock_llm,
            vector_db=mock_vector_db
        )
        
        # Assert
        assert isinstance(agent, EnhancedAgentBase)
        assert hasattr(agent, 'run_as_node')
        assert callable(getattr(agent, 'run_as_node'))
    
    def test_formatter_agent_compliance(self):
        """Test that FormatterAgent implements the required interface."""
        from src.agents.formatter_agent import FormatterAgent
        
        # Act
        agent = FormatterAgent(name="formatter", description="test")
        
        # Assert
        assert isinstance(agent, EnhancedAgentBase)
        assert hasattr(agent, 'run_as_node')
        assert callable(getattr(agent, 'run_as_node'))