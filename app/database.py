from __future__ import annotations

import enum
import os
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Enum as SAEnum, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./pawai.db')

_connect_args = {'check_same_thread': False} if DATABASE_URL.startswith('sqlite') else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class JobStatus(str, enum.Enum):
    pending = 'pending'
    processing = 'processing'
    done = 'done'
    failed = 'failed'


class PredictionJob(Base):
    __tablename__ = 'prediction_jobs'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(SAEnum(JobStatus), default=JobStatus.pending, nullable=False)
    top_k_requested = Column(Integer, default=3)
    result = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
