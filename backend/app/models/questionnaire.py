from sqlalchemy import Boolean, Text, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base

class Questionnaire(Base):
    __tablename__ = "questionnaires"

    id: Mapped[int] = mapped_column(primary_key=True)
    visit_id: Mapped[int] = mapped_column(ForeignKey("visits.id", ondelete="CASCADE"), unique=True, index=True)

    # Legacy boolean fields (kept for ML pipeline compatibility)
    regression: Mapped[bool] = mapped_column(Boolean, default=False)
    seizures: Mapped[bool] = mapped_column(Boolean, default=False)
    motor_delay: Mapped[bool] = mapped_column(Boolean, default=False)
    global_delay: Mapped[bool] = mapped_column(Boolean, default=False)
    family_history_asd_ndd: Mapped[bool] = mapped_column(Boolean, default=False)
    dysmorphic_features: Mapped[bool] = mapped_column(Boolean, default=False)
    macrocephaly: Mapped[bool] = mapped_column(Boolean, default=False)
    microcephaly: Mapped[bool] = mapped_column(Boolean, default=False)

    # Likert-scale questionnaire responses (question_key -> "always"|"often"|"sometimes"|"rarely"|"never")
    responses: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Family history matrix: {condition: {family_member: bool}}
    family_history: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    visit = relationship("Visit", back_populates="questionnaire")
