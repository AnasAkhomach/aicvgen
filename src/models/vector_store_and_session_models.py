from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime


class VectorStoreItemMetadata(BaseModel):
    # Add fields as needed for metadata
    # Example: job_id: Optional[str] = None
    pass


class VectorStoreSearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    id: Optional[str] = None
    distance: Optional[float] = None


class SessionInfoModel(BaseModel):
    session_id: str
    user_id: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    current_stage: str
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    metadata: Dict[str, Any] = {}
    processing_time: float = 0.0
    llm_calls: int = 0
    tokens_used: int = 0


class SessionSummaryModel(BaseModel):
    sessions: List[SessionInfoModel]
