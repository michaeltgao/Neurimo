from datetime import date
from pydantic import BaseModel

class VisitInfo(BaseModel):
    id: int
    child_id: int
    visit_date: date
    age_months: int

class PriorVisit(BaseModel):
    id: int
    age_months: int
    visit_date: date
    asd_risk_bucket: str

class ReportOut(BaseModel):
    visit: VisitInfo
    asd_risk_bucket: str
    explanations: list[str]
    prior_visits: list[PriorVisit]
