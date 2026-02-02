from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.visit import Visit
from app.models.questionnaire import Questionnaire
from app.schemas.questionnaire import QuestionnaireCreate, QuestionnaireOut

router = APIRouter(prefix="/visits", tags=["questionnaires"])

@router.post("/{visit_id}/questionnaire", response_model=QuestionnaireOut)
def upsert_questionnaire(
    visit_id: str,
    payload: QuestionnaireCreate,
    db: Session = Depends(get_db),
):
    # Parse visit_id format: "child_id-visit_number" (e.g., "22-1")
    try:
        child_id_str, visit_number_str = visit_id.split("-")
        child_id = int(child_id_str)
        visit_number = int(visit_number_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid visit_id format. Expected 'child_id-visit_number' (e.g., '22-1')")

    visit = db.query(Visit).filter(
        Visit.child_id == child_id,
        Visit.visit_number == visit_number
    ).first()
    if not visit:
        raise HTTPException(status_code=404, detail="Visit not found")

    existing = db.query(Questionnaire).filter(Questionnaire.visit_id == visit.id).first()
    if existing:
        for k, v in payload.model_dump().items():
            setattr(existing, k, v)
        db.commit()
        db.refresh(existing)
        return existing

    q = Questionnaire(visit_id=visit.id, **payload.model_dump())
    db.add(q)
    db.commit()
    db.refresh(q)
    return q

@router.get("/{visit_id}/questionnaire", response_model=QuestionnaireOut)
def get_questionnaire(visit_id: str, db: Session = Depends(get_db)):
    # Parse visit_id format: "child_id-visit_number" (e.g., "22-1")
    try:
        child_id_str, visit_number_str = visit_id.split("-")
        child_id = int(child_id_str)
        visit_number = int(visit_number_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid visit_id format. Expected 'child_id-visit_number' (e.g., '22-1')")

    visit = db.query(Visit).filter(
        Visit.child_id == child_id,
        Visit.visit_number == visit_number
    ).first()
    if not visit:
        raise HTTPException(status_code=404, detail="Visit not found")

    q = db.query(Questionnaire).filter(Questionnaire.visit_id == visit.id).first()
    if not q:
        raise HTTPException(status_code=404, detail="Questionnaire not found")
    return q
