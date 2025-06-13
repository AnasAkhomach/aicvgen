#!/usr/bin/env python3
"""
Simple test to debug the cv_parser agent issue without complex imports.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_simple_cv_parser():
    """Simple test to check cv_parser agent availability."""
    print("=== Simple CV Parser Debug Test ===")
    
    try:
        # Import only what we need
        from src.orchestration.agent_orchestrator import get_agent_orchestrator
        
        # Get the default orchestrator
        orchestrator = get_agent_orchestrator()
        
        print("\n1. Checking orchestrator initialization:")
        print(f"   Session ID: {orchestrator.session_id}")
        print(f"   Lifecycle manager: {orchestrator.lifecycle_manager is not None}")
        
        print("\n2. Checking registered agent types:")
        lifecycle_manager = orchestrator.lifecycle_manager
        registered_types = list(lifecycle_manager._pools.keys())
        print(f"   Registered types: {registered_types}")
        
        print("\n3. Checking cv_parser pool:")
        if "cv_parser" in lifecycle_manager._pools:
            pool = lifecycle_manager._pools["cv_parser"]
            print(f"   Pool size: {len(pool)}")
            print(f"   Pool strategy: {lifecycle_manager._pool_configs['cv_parser'].strategy}")
            print(f"   Min instances: {lifecycle_manager._pool_configs['cv_parser'].min_instances}")
            print(f"   Max instances: {lifecycle_manager._pool_configs['cv_parser'].max_instances}")
            
            # Check agent states
            if pool:
                for i, agent in enumerate(pool):
                    print(f"   Agent {i}: {agent.state} - Available: {agent.is_available()}")
            else:
                print("   Pool is empty - this might be the issue!")
        else:
            print("   cv_parser not registered!")
        
        print("\n4. Attempting to get cv_parser agent:")
        try:
            managed_agent = orchestrator.get_agent("cv_parser")
            if managed_agent:
                print(f"   ✓ Got agent: {managed_agent.instance.__class__.__name__}")
                print(f"   Agent state: {managed_agent.state}")
                print(f"   Agent available: {managed_agent.is_available()}")
            else:
                print("   ✗ Failed to get agent - this is the root cause!")
                
                # Let's try to create one manually
                print("\n5. Attempting manual agent creation:")
                try:
                    agent = lifecycle_manager.create_agent("cv_parser", "test_session")
                    if agent:
                        print(f"   ✓ Manual creation successful: {agent.instance.__class__.__name__}")
                        print(f"   Agent state: {agent.state}")
                    else:
                        print("   ✗ Manual creation also failed")
                except Exception as e:
                    print(f"   ✗ Manual creation error: {e}")
                    
        except Exception as e:
            print(f"   ✗ Error getting agent: {e}")
        
        print("\n=== Test completed ===")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_cv_parser()