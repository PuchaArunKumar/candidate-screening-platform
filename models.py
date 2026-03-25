import uuid

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base


Base = declarative_base()


def new_uuid_str() -> str:
    return str(uuid.uuid4())


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(String(36), primary_key=True, default=new_uuid_str)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

    status = Column(String(50), nullable=False, default="created")
    error = Column(Text, nullable=True)

    job_description = Column(Text, nullable=True)
    shortlist_threshold = Column(Float, nullable=True)

    # Weighted formula params for Phase 6.
    w_resume = Column(Float, nullable=False, default=0.5)
    w_github = Column(Float, nullable=False, default=0.3)
    w_cgpa = Column(Float, nullable=False, default=0.2)

    # Test weights for Phase 8.
    w_test_la = Column(Float, nullable=False, default=0.4)
    w_test_code = Column(Float, nullable=False, default=0.6)
    w_pipeline = Column(Float, nullable=False, default=0.7)
    w_test = Column(Float, nullable=False, default=0.3)


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(String(36), primary_key=True, default=new_uuid_str)
    run_id = Column(String(36), ForeignKey("pipeline_runs.id"), nullable=False, index=True)

    external_candidate_id = Column(String(64), nullable=True, index=True)
    name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True, index=True)
    college = Column(String(255), nullable=True)
    branch = Column(String(255), nullable=True)
    cgpa = Column(Float, nullable=True)
    best_ai_project = Column(Text, nullable=True)
    research_work = Column(Text, nullable=True)
    github_profile_url = Column(Text, nullable=True)
    resume_link_url = Column(Text, nullable=True)


class ResumeExtraction(Base):
    __tablename__ = "resume_extractions"

    candidate_id = Column(String(36), ForeignKey("candidates.id"), primary_key=True)
    resume_text = Column(Text, nullable=True)
    extracted_at = Column(DateTime, nullable=True)


class AiEvaluation(Base):
    __tablename__ = "ai_evaluations"

    candidate_id = Column(String(36), ForeignKey("candidates.id"), primary_key=True)
    resume_ai_score = Column(Integer, nullable=True)
    resume_ai_explanation = Column(Text, nullable=True)
    evaluated_at = Column(DateTime, nullable=True)


class GithubAnalysis(Base):
    __tablename__ = "github_analyses"

    candidate_id = Column(String(36), ForeignKey("candidates.id"), primary_key=True)
    github_technical_score = Column(Integer, nullable=True)
    github_summary = Column(Text, nullable=True)
    analyzed_at = Column(DateTime, nullable=True)


class Ranking(Base):
    __tablename__ = "rankings"

    candidate_id = Column(String(36), ForeignKey("candidates.id"), primary_key=True)
    overall_score = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=True)


class TestResult(Base):
    __tablename__ = "test_results"

    candidate_id = Column(String(36), ForeignKey("candidates.id"), primary_key=True)
    test_la = Column(Integer, nullable=True)
    test_code = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime, nullable=True)


class FinalRanking(Base):
    __tablename__ = "final_rankings"

    candidate_id = Column(String(36), ForeignKey("candidates.id"), primary_key=True)
    test_performance_score = Column(Float, nullable=True)
    final_score = Column(Float, nullable=True)
    computed_at = Column(DateTime, nullable=True)


class TestLink(Base):
    __tablename__ = "test_links"

    candidate_id = Column(String(36), ForeignKey("candidates.id"), primary_key=True)
    token = Column(String(255), nullable=False)
    test_link_url = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=True)


class InterviewEvent(Base):
    __tablename__ = "interview_events"

    candidate_id = Column(String(36), ForeignKey("candidates.id"), primary_key=True)
    calendar_event_id = Column(String(255), nullable=True)
    meet_link = Column(Text, nullable=True)
    scheduled_start = Column(DateTime, nullable=True)
    scheduled_end = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=True)


class EmailSendLog(Base):
    __tablename__ = "email_send_logs"

    id = Column(String(36), primary_key=True, default=new_uuid_str)
    candidate_id = Column(String(36), ForeignKey("candidates.id"), index=True, nullable=False)
    run_id = Column(String(36), ForeignKey("pipeline_runs.id"), index=True, nullable=False)

    channel = Column(String(20), nullable=False, default="smtp")
    message_id = Column(String(255), nullable=True)
    sent_at = Column(DateTime, nullable=True)


