#!/usr/bin/env python3
"""
Test script to verify all critical imports work correctly.
Part of T002: Codebase Health Verification
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_critical_imports():
    """Test that all critical modules can be imported successfully."""
    
    print("Testing critical imports...")
    
    try:
        # Test constants imports
        from src.constants.llm_constants import LLMConstants
        from src.constants.config_constants import ConfigConstants
        from src.constants.agent_constants import AgentConstants
        print("✅ Constants imports successful")
        
        # Test config imports
        from src.config.settings import LLMSettings, AgentSettings
        from src.config.logging_config import get_structured_logger
        print("✅ Config imports successful")
        
        # Test model imports
        from src.models.data_models import RateLimitLog, RateLimitState
        from src.models.agent_models import AgentResult
        print("✅ Model imports successful")
        
        # Test error handling imports
        from src.error_handling.exceptions import NetworkError, RateLimitError
        print("✅ Error handling imports successful")
        
        # Test agent imports
        from src.agents.professional_experience_writer_agent import ProfessionalExperienceWriterAgent
        from src.agents.quality_assurance_agent import QualityAssuranceAgent
        print("✅ Agent imports successful")
        
        # Test service imports
        from src.services.rate_limiter import RateLimiter
        from src.services.session_manager import SessionManager
        print("✅ Service imports successful")
        
        print("\n🎉 All critical imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key components."""
    
    print("\nTesting basic functionality...")
    
    try:
        # Test logger creation
        from src.config.logging_config import get_structured_logger
        logger = get_structured_logger("test")
        logger.info("Test log message")
        print("✅ Logging functionality works")
        
        # Test constants access
        from src.constants.llm_constants import LLMConstants
        assert hasattr(LLMConstants, 'MAX_TOKENS_GENERATION')
        print("✅ Constants access works")
        
        # Test settings creation
        from src.config.settings import LLMSettings
        settings = LLMSettings()
        assert hasattr(settings, 'default_model')
        print("✅ Settings creation works")
        
        print("\n🎉 Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("AICVGEN IMPORT & FUNCTIONALITY TEST")
    print("=" * 50)
    
    imports_ok = test_critical_imports()
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 50)
    if imports_ok and functionality_ok:
        print("✅ ALL TESTS PASSED - Codebase is healthy!")
        sys.exit(0)
    else:
        print("❌ TESTS FAILED - Issues need to be resolved")
        sys.exit(1)