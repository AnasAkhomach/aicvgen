#!/usr/bin/env python3
"""
Simple test script to debug lifecycle manager cv_parser issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_lifecycle_manager_only():
    """Test only the lifecycle manager cv_parser issue."""
    print("=== Testing Lifecycle Manager Only ===")
    
    try:
        from src.core.agent_lifecycle_manager import get_agent_lifecycle_manager
        
        manager = get_agent_lifecycle_manager()
        print(f"Lifecycle manager created: {type(manager).__name__}")
        
        # Check statistics
        stats = manager.get_statistics()
        print(f"Pool statistics keys: {list(stats.get('pool_statistics', {}).keys())}")
        
        # Check if cv_parser is registered
        pool_stats = stats.get('pool_statistics', {})
        if 'cv_parser' in pool_stats:
            print(f"✓ cv_parser pool found: {pool_stats['cv_parser']}")
        else:
            print(f"✗ cv_parser not in pool statistics")
            print(f"Available pools: {list(pool_stats.keys())}")
        
        # Try to get cv_parser agent
        print("\nAttempting to get cv_parser agent...")
        agent = manager.get_agent('cv_parser')
        
        if agent:
            print(f"✓ cv_parser agent retrieved: {type(agent).__name__}")
            print(f"  Agent ID: {agent.agent_id}")
            print(f"  Agent type: {agent.agent_type}")
            return True
        else:
            print("✗ cv_parser agent returned None")
            print("This means cv_parser is not registered with lifecycle manager")
            return False
            
    except Exception as e:
        print(f"✗ Error in lifecycle manager test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_orchestrator_registration():
    """Test if orchestrator properly registers cv_parser with lifecycle manager."""
    print("\n=== Testing Orchestrator Registration ===")
    
    try:
        from src.orchestration.agent_orchestrator import get_agent_orchestrator
        
        # Create orchestrator (this should register agents)
        orchestrator = get_agent_orchestrator()
        print(f"Orchestrator created: {type(orchestrator).__name__}")
        
        # Now check lifecycle manager again
        manager = orchestrator.lifecycle_manager
        stats = manager.get_statistics()
        pool_stats = stats.get('pool_statistics', {})
        
        print(f"Pools after orchestrator creation: {list(pool_stats.keys())}")
        
        if 'cv_parser' in pool_stats:
            print(f"✓ cv_parser pool registered: {pool_stats['cv_parser']}")
            
            # Try to get agent
            agent = manager.get_agent('cv_parser')
            if agent:
                print(f"✓ cv_parser agent available: {type(agent).__name__}")
                return True
            else:
                print("✗ cv_parser pool exists but no agent available")
                return False
        else:
            print("✗ cv_parser still not registered after orchestrator creation")
            return False
            
    except Exception as e:
        print(f"✗ Error in orchestrator registration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Lifecycle Manager CV Parser Issue...\n")
    
    # Test lifecycle manager directly
    lifecycle_direct = test_lifecycle_manager_only()
    
    # Test after orchestrator registration
    lifecycle_after_orchestrator = test_orchestrator_registration()
    
    print("\n=== Results ===")
    print(f"Lifecycle Manager (direct): {'✓' if lifecycle_direct else '✗'}")
    print(f"Lifecycle Manager (after orchestrator): {'✓' if lifecycle_after_orchestrator else '✗'}")
    
    if lifecycle_after_orchestrator:
        print("\n🎉 Issue resolved! cv_parser works after orchestrator initialization.")
    elif not lifecycle_direct and not lifecycle_after_orchestrator:
        print("\n❌ cv_parser not available in lifecycle manager even after orchestrator init.")
    else:
        print("\n⚠️ Mixed results - needs further investigation.")