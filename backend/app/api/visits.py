from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.visit import Visit
from app.models.child import Child
from app.schemas.visit import VisitCreate, VisitOut

router = APIRouter(tags=["visits"])


@router.get("/visits/{visit_id}", response_model=VisitOut)
def get_visit(visit_id: str, db: Session = Depends(get_db)):
    # Parse visit_id in format "child_id-visit_number" (e.g., "21-1")
    parts = visit_id.split("-")
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Invalid visit ID format. Expected: child_id-visit_number")

    try:
        child_id = int(parts[0])
        visit_number = int(parts[1])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid visit ID format. Expected: child_id-visit_number")

    visit = db.query(Visit).filter(Visit.child_id == child_id, Visit.visit_number == visit_number).first()
    if not visit:
        raise HTTPException(status_code=404, detail="Visit not found")
    return visit

@router.post("/children/{child_id}/visits", response_model=VisitOut)
def create_visit(child_id: int, payload: VisitCreate, db: Session = Depends(get_db)):
    child = db.query(Child).filter(Child.id == child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    # Calculate the next visit number for this child
    max_visit = db.query(Visit).filter(Visit.child_id == child_id).order_by(Visit.visit_number.desc()).first()
    next_visit_number = (max_visit.visit_number + 1) if max_visit else 1

    visit = Visit(child_id=child_id, visit_number=next_visit_number, **payload.model_dump())
    db.add(visit)
    db.commit()
    db.refresh(visit)
    return visit

@router.get("/children/{child_id}/visits", response_model=list[VisitOut])
def list_visits(child_id: int, db: Session = Depends(get_db)):
    return (
        db.query(Visit)
        .filter(Visit.child_id == child_id)
        .order_by(Visit.visit_date.asc())
        .all()
    )
