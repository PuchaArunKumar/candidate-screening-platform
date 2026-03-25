from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import get_settings


settings = get_settings()


def _connect_args():
    # Needed only for SQLite local development.
    return {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}


engine = create_engine(settings.DATABASE_URL, connect_args=_connect_args(), pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

