from sqlalchemy import String, Date, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base

class Child(Base):
    __tablename__ = "children"

    id: Mapped[int] = mapped_column(primary_key=True)
    pseudo_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    birthdate: Mapped[Date] = mapped_column(Date)
    sex: Mapped[str] = mapped_column(String(16))
    clinic_id: Mapped[str] = mapped_column(String(64), default="clinic-001")

    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    visits = relationship("Visit", back_populates="child", cascade="all, delete-orphan")
