from sqlalchemy import Integer, ForeignKey, Date, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base

class Visit(Base):
    __tablename__ = "visits"

    id: Mapped[int] = mapped_column(primary_key=True)
    child_id: Mapped[int] = mapped_column(ForeignKey("children.id", ondelete="CASCADE"), index=True)
    visit_number: Mapped[int] = mapped_column(Integer, nullable=False)
    visit_date: Mapped[Date] = mapped_column(Date)
    age_months: Mapped[int] = mapped_column(Integer)

    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    child = relationship("Child", back_populates="visits")
    videos = relationship("Video", back_populates="visit", cascade="all, delete-orphan")
    questionnaire = relationship("Questionnaire", back_populates="visit", uselist=False, cascade="all, delete-orphan")
    ml_prediction = relationship("MLPrediction", back_populates="visit", uselist=False, cascade="all, delete-orphan")
