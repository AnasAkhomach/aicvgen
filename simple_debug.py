import asyncio
import sys
sys.path.append('.')

from src.orchestration.state import AgentState
from src.models.cv_models import StructuredCV, JobDescriptionData

# Create minimal test data
structured_cv = StructuredCV(sections=[])
job_data = JobDescriptionData(raw_text="test job")

# Create state
state = AgentState(
    structured_cv=structured_cv,
    job_description_data=job_data,
    cv_text="test"
)

# Test model_dump
state_dict = state.model_dump()
print(f"Keys: {list(state_dict.keys())}")
print(f"structured_cv in dict: {'structured_cv' in state_dict}")
print(f"job_description_data in dict: {'job_description_data' in state_dict}")

if 'structured_cv' in state_dict:
    print(f"structured_cv type: {type(state_dict['structured_cv'])}")
if 'job_description_data' in state_dict:
    print(f"job_description_data type: {type(state_dict['job_description_data'])}")