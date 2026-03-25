from typing import Any, Optional

from pydantic import BaseModel, Field


class CreatePipelineRunResponse(BaseModel):
    run_id: str


class JobDescriptionIn(BaseModel):
    job_description: str


class WeightsPhase6In(BaseModel):
    w_resume: float = Field(default=0.5, ge=0, le=1)
    w_github: float = Field(default=0.3, ge=0, le=1)
    w_cgpa: float = Field(default=0.2, ge=0, le=1)
    threshold: float = Field(default=70.0, ge=0, le=100)


class StartPhase7In(BaseModel):
    weights: WeightsPhase6In
    # Base URL to build per-candidate test links
    test_link_base: str = Field(default="https://your-app-domain.com/tests/take?token=")


class UploadTestResultsResponse(BaseModel):
    qualified_count: int


class UploadTestResultsIn(BaseModel):
    test_threshold: float = Field(default=60.0, ge=0, le=100)


class LeaderboardRow(BaseModel):
    external_candidate_id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    cgpa: Optional[float] = None
    resume_ai_score: Optional[int] = None
    github_technical_score: Optional[int] = None
    overall_score: Optional[float] = None
    test_la: Optional[int] = None
    test_code: Optional[int] = None
    test_performance_score: Optional[float] = None
    final_score: Optional[float] = None

    is_qualified: bool = False

class LeaderboardResponse(BaseModel):
    run_id: str
    rows: list[LeaderboardRow]


class ScheduleIn(BaseModel):
    start_datetime: str = Field(default="2026-03-23T10:00:00Z")
    slot_minutes: int = Field(default=30, ge=5, le=240)
    timezone: str = Field(default="UTC")
    calendar_id: str = Field(default="primary")

