"""Test that AgentState correctly stores and returns cv_analysis_results (CB-01).
Uses a minimal StructuredCV instance to satisfy validation."""

import pytest
from src.orchestration.state import AgentState
from src.models.cv_analysis_result import CVAnalysisResult
from src.models.data_models import StructuredCV


def test_agent_state_cv_analysis_results():
    # Create a dummy CVAnalysisResult
    analysis = CVAnalysisResult(
        skill_matches=["Python", "AI"],
        experience_relevance=0.95,
        gaps_identified=["No Java"],
        strengths=["Deep Learning"],
        recommendations=["Learn Java"],
        match_score=0.88,
        analysis_timestamp="2025-06-21T12:00:00Z",
    )
    # Create minimal StructuredCV
    cv = StructuredCV()
    # Create AgentState with cv_analysis_results
    state = AgentState(structured_cv=cv, cv_analysis_results=analysis)
    assert state.cv_analysis_results is not None
    assert state.cv_analysis_results.skill_matches == ["Python", "AI"]
    assert state.cv_analysis_results.match_score == 0.88
