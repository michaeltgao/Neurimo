from datetime import date, datetime
from pydantic import BaseModel

class VisitCreate(BaseModel):
    visit_date: date
    age_months: int

class VisitOut(BaseModel):
    id: int
    child_id: int
    visit_number: int
    visit_date: date
    age_months: int
    created_at: datetime

    class Config:
        from_attributes = True
