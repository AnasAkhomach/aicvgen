import sys
import os
import pytest
from datetime import datetime

# Ensure project root (containing src/) is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.quality_models import (
    QualityCheck,
    QualityCheckType,
    QualityStatus,
    QualityCheckMetadataModel,
    QualityCheckResults,
    QualityCriteriaModel,
    SectionQualityResult,
    ItemQualityResult,
)
from src.models.data_models import MetadataModel


def test_quality_check_metadata_model():
    meta = QualityCheckMetadataModel(
        check_id="chk1", source="unit-test", extra={"foo": "bar"}
    )
    assert meta.check_id == "chk1"
    assert meta.source == "unit-test"
    assert meta.extra["foo"] == "bar"


def test_quality_criteria_model():
    crit = QualityCriteriaModel(
        min_length=10, max_length=100, required_keywords=["python"], extra={"bar": 1}
    )
    assert crit.min_length == 10
    assert crit.max_length == 100
    assert crit.required_keywords == ["python"]
    assert crit.extra["bar"] == 1


def test_quality_check_enforces_metadata_type():
    qc = QualityCheck(
        check_type=QualityCheckType.GRAMMAR_CHECK,
        status=QualityStatus.PASS,
        message="All good.",
        score=1.0,
        suggestions=[],
        metadata=QualityCheckMetadataModel(check_id="chk2"),
    )
    assert isinstance(qc.metadata, QualityCheckMetadataModel)
    assert qc.metadata.check_id == "chk2"


def test_quality_check_results_enforces_types():
    qcr = QualityCheckResults(
        session_id="sess1",
        trace_id="trace1",
        quality_criteria=QualityCriteriaModel(min_length=5),
        metadata=MetadataModel(company="TestCorp"),
    )
    assert isinstance(qcr.quality_criteria, QualityCriteriaModel)
    assert isinstance(qcr.metadata, MetadataModel)
    assert qcr.quality_criteria.min_length == 5
    assert qcr.metadata.company == "TestCorp"


def test_quality_check_results_serialization():
    qcr = QualityCheckResults.create_empty()
    d = qcr.to_dict()
    assert isinstance(d, dict)
    qcr2 = QualityCheckResults.from_dict(d)
    assert isinstance(qcr2, QualityCheckResults)


def test_section_and_item_quality_result():
    item = ItemQualityResult(
        item_id="item1",
        section="Experience",
        content_preview="Worked at X",
        overall_status=QualityStatus.PASS,
        checks=[],
        overall_score=1.0,
        timestamp=datetime.now(),
    )
    section = SectionQualityResult(section_name="Experience")
    section.add_item_result(item)
    assert section.total_items == 1
    assert section.passed_items == 1
    assert section.failed_items == 0
    assert section.warning_items == 0
