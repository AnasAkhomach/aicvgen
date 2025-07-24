"""Debug script to test singleton behavior."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.container import Container, ContainerSingleton, get_container

print("Testing Container singleton behavior...")

# Test 1: Direct instantiation should fail
print("\n1. Testing direct instantiation:")
try:
    container = Container()
    print("ERROR: Direct instantiation succeeded when it should have failed!")
except RuntimeError as e:
    print(f"SUCCESS: Direct instantiation failed as expected: {e}")

# Test 2: get_container should work
print("\n2. Testing get_container():")
try:
    container1 = get_container()
    print(f"SUCCESS: get_container() returned: {type(container1)} at {id(container1)}")
except Exception as e:
    print(f"ERROR: get_container() failed: {e}")

# Test 3: Multiple calls should return same instance
print("\n3. Testing singleton behavior:")
try:
    container2 = get_container()
    print(f"Second call returned: {type(container2)} at {id(container2)}")
    print(f"Same instance? {container1 is container2}")
except Exception as e:
    print(f"ERROR: Second get_container() call failed: {e}")

# Test 4: Reset and create new
print("\n4. Testing reset:")
try:
    ContainerSingleton.reset_instance()
    container3 = get_container()
    print(f"After reset: {type(container3)} at {id(container3)}")
    print(f"Different from original? {container1 is not container3}")
except Exception as e:
    print(f"ERROR: Reset test failed: {e}")

print("\nDone.")