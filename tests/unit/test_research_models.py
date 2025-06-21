import sys
import os
import pytest
from datetime import datetime

# Ensure project root (containing src/) is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.research_models import (
    ResearchFindings,
    ResearchMetadataModel,
    ResearchStatus,
)


def test_research_metadata_model():
    meta = ResearchMetadataModel(
        source="web", analyst="Alice", notes="Initial scrape", extra={"foo": 1}
    )
    assert meta.source == "web"
    assert meta.analyst == "Alice"
    assert meta.notes == "Initial scrape"
    assert meta.extra["foo"] == 1


def test_research_findings_enforces_metadata_type():
    rf = ResearchFindings(
        status=ResearchStatus.SUCCESS,
        research_timestamp=datetime.now(),
        metadata=ResearchMetadataModel(source="api", analyst="Bob"),
    )
    assert isinstance(rf.metadata, ResearchMetadataModel)
    assert rf.metadata.source == "api"
    assert rf.metadata.analyst == "Bob"


def test_research_findings_serialization():
    rf = ResearchFindings.create_empty()
    d = rf.to_dict()
    assert isinstance(d, dict)
    rf2 = ResearchFindings.from_dict(d)
    assert isinstance(rf2, ResearchFindings)
