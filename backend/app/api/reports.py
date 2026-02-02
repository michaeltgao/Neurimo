from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.visit import Visit
from app.schemas.report import ReportOut
from app.services.report_service import build_report

router = APIRouter(prefix="/visits", tags=["reports"])

@router.get("/{visit_id}/report", response_model=ReportOut)
def get_report(visit_id: str, db: Session = Depends(get_db)):
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

    report = build_report(db, visit.id)
    if not report:
        raise HTTPException(status_code=404, detail="Visit not found")
    return report
