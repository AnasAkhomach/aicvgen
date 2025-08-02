"""Unit tests for agent output models serialization."""

import json
import pytest
from datetime import datetime
from typing import Dict, Any

from src.models.agent_output_models import (
    ParserAgentOutput,
    EnhancedContentWriterOutput,
    CleaningAgentOutput,
    CVAnalysisResult,
    CVAnalyzerAgentOutput,
    ItemQualityResultModel,
    SectionQualityResultModel,
    OverallQualityCheckResultModel,
    QualityAssuranceAgentOutput,
    FormatterAgentOutput,
    CompanyInsight,
    IndustryInsight,
    RoleInsight,
    ResearchMetadataModel,
    ResearchFindings,
    ResearchAgentOutput,
    ResearchStatus,
)


class TestAgentOutputModelsSerialization:
    """Test serialization for all agent output models."""

    def test_parser_agent_output_serialization(self):
        """Test ParserAgentOutput serialization."""
        from src.models.cv_models import JobDescriptionData, StructuredCV

        job_data = JobDescriptionData(
            raw_text="Software Engineer position at Test Company. Test job description with requirements.",
            job_title="Software Engineer",
            company_name="Test Company",
        )

        structured_cv = StructuredCV()

        model = ParserAgentOutput(
            job_description_data=job_data, structured_cv=structured_cv
        )

        # Test serialization cycle
        data = model.to_dict()
        restored = ParserAgentOutput.from_dict(data)
        assert restored.job_description_data is not None
        assert restored.structured_cv is not None

        # Test JSON serialization
        json_str = model.model_dump_json()
        json.loads(json_str)  # Should not raise

    def test_enhanced_content_writer_output_serialization(self):
        """Test EnhancedContentWriterOutput serialization."""
        # Create a minimal StructuredCV for testing
        from src.models.cv_models import StructuredCV

        structured_cv = StructuredCV()

        model = EnhancedContentWriterOutput(
            updated_structured_cv=structured_cv,
            item_id="test_item_123",
            generated_content="Enhanced content for the CV",
        )

        # Test serialization cycle
        data = model.to_dict()
        restored = EnhancedContentWriterOutput.from_dict(data)
        assert restored.item_id == model.item_id
        assert restored.generated_content == model.generated_content

        # Test JSON serialization
        json_str = model.model_dump_json()
        json.loads(json_str)  # Should not raise

    def test_cleaning_agent_output_serialization(self):
        """Test CleaningAgentOutput serialization."""
        model = CleaningAgentOutput(
            cleaned_data={"skills": ["Python", "SQL"], "experience": "5 years"},
            modifications_made=["Removed duplicates", "Fixed formatting"],
            raw_output="Raw unprocessed data",
            output_type="skills_list",
        )

        # Test serialization cycle
        data = model.to_dict()
        restored = CleaningAgentOutput.from_dict(data)
        assert restored.cleaned_data == model.cleaned_data
        assert restored.modifications_made == model.modifications_made
        assert restored.output_type == model.output_type

        # Test JSON serialization
        json_str = model.model_dump_json()
        json.loads(json_str)  # Should not raise

    def test_cv_analysis_result_serialization(self):
        """Test CVAnalysisResult serialization."""
        model = CVAnalysisResult(
            summary="Analysis summary",
            key_skills=["Python", "SQL"],
            strengths=["strength1", "strength2"],
            gaps_identified=["weakness1"],
            recommendations=["recommendation1"],
            match_score=0.85,
        )

        # Test serialization cycle
        data = model.to_dict()
        restored = CVAnalysisResult.from_dict(data)
        assert restored.summary == model.summary
        assert restored.strengths == model.strengths

        # Test JSON serialization
        json_str = model.model_dump_json()
        json.loads(json_str)  # Should not raise

    def test_quality_models_serialization(self):
        """Test quality-related models serialization."""
        item_quality = ItemQualityResultModel(
            item_id="test_item",
            passed=True,
            issues=["minor issue"],
            suggestions=["suggestion1"],
        )

        section_quality = SectionQualityResultModel(
            section_name="experience",
            passed=True,
            issues=[],
            item_checks=[item_quality],
        )

        overall_quality = OverallQualityCheckResultModel(
            check_name="overall_check", passed=True, details="All checks passed"
        )

        qa_output = QualityAssuranceAgentOutput(
            section_results=[section_quality],
            overall_passed=True,
            recommendations=["Keep up the good work"],
        )

        # Test serialization for all models
        for model in [item_quality, section_quality, overall_quality, qa_output]:
            data = model.to_dict()
            restored = type(model).from_dict(data)
            assert restored.model_dump() == model.model_dump()

            # Test JSON serialization
            json_str = model.model_dump_json()
            json.loads(json_str)  # Should not raise

    def test_formatter_agent_output_serialization(self):
        """Test FormatterAgentOutput serialization."""
        model = FormatterAgentOutput(output_path="/path/to/formatted/cv.pdf")

        # Test serialization cycle
        data = model.to_dict()
        restored = FormatterAgentOutput.from_dict(data)
        assert restored.output_path == model.output_path

        # Test JSON serialization
        json_str = model.model_dump_json()
        json.loads(json_str)  # Should not raise

    def test_research_models_serialization(self):
        """Test research-related models serialization."""
        company_insight = CompanyInsight(
            company_name="Test Company",
            industry="Technology",
            size="Large",
            culture="Innovative",
            recent_news=["News 1", "News 2"],
            key_values=["Innovation", "Quality"],
            confidence_score=0.9,
        )

        industry_insight = IndustryInsight(
            industry_name="Technology",
            trends=["AI", "Cloud"],
            key_skills=["Python", "AWS"],
            growth_areas=["Machine Learning"],
            challenges=["Talent shortage"],
            confidence_score=0.85,
        )

        role_insight = RoleInsight(
            role_title="Software Engineer",
            required_skills=["Python", "SQL"],
            preferred_qualifications=["Bachelor's degree"],
            responsibilities=["Code development"],
            career_progression=["Senior Engineer"],
            salary_range="$80k-120k",
            confidence_score=0.8,
        )

        metadata = ResearchMetadataModel(
            source="web_search",
            analyst="AI Agent",
            notes="Comprehensive research",
            extra={"version": "1.0", "timestamp": datetime.now().isoformat()},
        )

        research_findings = ResearchFindings(
            company_insights=company_insight,
            industry_insights=industry_insight,
            role_insights=role_insight,
            research_timestamp=datetime.now(),
            status=ResearchStatus.SUCCESS,
            confidence_score=0.85,
        )

        research_output = ResearchAgentOutput(
            research_findings=research_findings, metadata=metadata
        )

        # Test serialization for all research models
        models = [
            company_insight,
            industry_insight,
            role_insight,
            metadata,
            research_findings,
            research_output,
        ]

        for model in models:
            data = model.to_dict()
            restored = type(model).from_dict(data)

            # For datetime fields, compare ISO strings
            if hasattr(model, "research_timestamp") and model.research_timestamp:
                assert (
                    restored.research_timestamp.isoformat()
                    == model.research_timestamp.isoformat()
                )
            else:
                assert restored.model_dump() == model.model_dump()

            # Test JSON serialization
            json_str = model.model_dump_json()
            json.loads(json_str)  # Should not raise

    def test_datetime_serialization(self):
        """Test datetime serialization across all models."""
        test_datetime = datetime.now()

        # Test with ResearchFindings which has datetime field
        research_findings = ResearchFindings(
            research_timestamp=test_datetime,
            status=ResearchStatus.SUCCESS,
            confidence_score=0.8,
        )

        # Test JSON serialization preserves datetime as ISO string
        json_str = research_findings.model_dump_json()
        data = json.loads(json_str)
        assert test_datetime.isoformat() in json_str

        # Test deserialization
        restored = ResearchFindings.from_dict(data)
        assert isinstance(restored.research_timestamp, datetime)

    def test_complex_nested_serialization(self):
        """Test serialization of complex nested structures."""
        # Create a complex nested structure
        research_findings = ResearchFindings(
            company_insights=CompanyInsight(
                company_name="Test Corp", industry="Tech", confidence_score=0.9
            ),
            industry_insights=IndustryInsight(
                industry_name="Technology",
                trends=["AI", "Cloud"],
                confidence_score=0.85,
            ),
            role_insights=RoleInsight(
                role_title="Developer", required_skills=["Python"], confidence_score=0.8
            ),
            research_timestamp=datetime.now(),
            status=ResearchStatus.SUCCESS,
            confidence_score=0.85,
        )

        research_output = ResearchAgentOutput(
            research_findings=research_findings,
            metadata=ResearchMetadataModel(
                source="comprehensive_search", extra={"nested": {"data": "value"}}
            ),
        )

        # Test full serialization cycle
        data = research_output.to_dict()
        restored = ResearchAgentOutput.from_dict(data)

        # Verify nested structures are preserved
        assert restored.research_findings.company_insights is not None
        assert restored.research_findings.company_insights.company_name == "Test Corp"
        assert restored.research_findings.industry_insights is not None
        assert (
            restored.research_findings.industry_insights.industry_name == "Technology"
        )
        assert restored.research_findings.role_insights is not None
        assert restored.research_findings.role_insights.role_title == "Developer"

        # Test JSON serialization of complex structure
        json_str = research_output.model_dump_json()
        json.loads(json_str)  # Should not raise
        assert "Test Corp" in json_str
        assert "Technology" in json_str
        assert "Developer" in json_str
