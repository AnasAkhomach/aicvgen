import pytest
from datetime import datetime, timedelta
from src.services.session_manager import SessionManager
from src.models.vector_store_and_session_models import SessionInfoModel


class DummySession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.user_id = "user1"
        self.status = type("Status", (), {"value": "active"})()
        self.created_at = datetime.now() - timedelta(hours=1)
        self.updated_at = datetime.now()
        self.expires_at = datetime.now() + timedelta(hours=1)
        self.current_stage = type("Stage", (), {"value": "stage1"})()
        self.total_items = 1
        self.completed_items = 1
        self.failed_items = 0
        self.metadata = {"foo": "bar"}
        self.processing_time = 1.0
        self.llm_calls = 2
        self.tokens_used = 10

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "current_stage": self.current_stage.value,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "metadata": self.metadata,
            "processing_time": self.processing_time,
            "llm_calls": self.llm_calls,
            "tokens_used": self.tokens_used,
        }


def test_session_manager_summary_returns_pydantic_models(monkeypatch):
    manager = SessionManager()
    dummy = DummySession("sess1")
    manager.active_sessions = {"sess1": dummy}
    summary = manager.get_session_summary()
    assert "sessions" in summary
    assert all(
        isinstance(SessionInfoModel(**s), SessionInfoModel) for s in summary["sessions"]
    )
    assert summary["sessions"][0]["session_id"] == "sess1"
