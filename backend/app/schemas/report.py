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
    risk_score: float | None = None
    visit_number: int
    is_current: bool

class ExplanationsByTask(BaseModel):
    joint_attention: list[str] = []
    imitation: list[str] = []
    free_play: list[str] = []
    questionnaire: list[str] = []
    general: list[str] = []

class ReportOut(BaseModel):
    visit: VisitInfo
    asd_risk_bucket: str
    risk_score: float | None = None
    explanations: list[str]
    explanations_by_task: ExplanationsByTask | None = None
    prior_visits: list[PriorVisit]
