from sqlalchemy import String, Float, ForeignKey, DateTime, func, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base

class MLPrediction(Base):
    __tablename__ = "ml_predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    visit_id: Mapped[int] = mapped_column(ForeignKey("visits.id", ondelete="CASCADE"), unique=True, index=True)

    asd_risk_bucket: Mapped[str] = mapped_column(String(16))  # low/moderate/moderate-high/high
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)  # raw model probability 0.0-1.0
    explanations: Mapped[list] = mapped_column(JSON, default=list)

    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    visit = relationship("Visit", back_populates="ml_prediction")
