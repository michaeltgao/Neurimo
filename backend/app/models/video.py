from sqlalchemy import String, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base

class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True)
    visit_id: Mapped[int] = mapped_column(ForeignKey("visits.id", ondelete="CASCADE"), index=True)

    task_type: Mapped[str] = mapped_column(String(32), index=True)  # imitation, joint_attention, free_play
    storage_path: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(32), default="uploaded")  # uploaded, processed, failed

    annotations_path: Mapped[str] = mapped_column(String(256), nullable=True)      # e.g. data/annotations/42.json
    annotations_version: Mapped[str] = mapped_column(String(32), nullable=True)   # e.g. "v1"

    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    visit = relationship("Visit", back_populates="videos")
