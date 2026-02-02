from datetime import date, datetime
from pydantic import BaseModel

class ChildCreate(BaseModel):
    pseudo_id: str
    birthdate: date
    sex: str
    clinic_id: str | None = "default"

class ChildOut(BaseModel):
    id: int
    pseudo_id: str
    birthdate: date
    sex: str
    clinic_id: str
    created_at: datetime

    class Config:
        from_attributes = True
