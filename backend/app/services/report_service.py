from sqlalchemy.orm import Session
from app.models.visit import Visit
from app.models.ml_prediction import MLPrediction
from app.models.video import Video

REQUIRED_TASKS = {"imitation", "joint_attention", "free_play"}

def build_report(db: Session, visit_id: int) -> dict:
    visit = db.query(Visit).filter(Visit.id == visit_id).first()
    if not visit:
        return {}

    # Check uploaded tasks
    videos = db.query(Video).filter(Video.visit_id == visit_id).all()
    tasks_present = {v.task_type for v in videos}
    has_all_videos = REQUIRED_TASKS.issubset(tasks_present)

    pred = db.query(MLPrediction).filter(MLPrediction.visit_id == visit_id).first()

    if not has_all_videos:
        risk_bucket = "insufficient_data"
        explanations = ["Upload all 3 behavioral videos to compute ASD risk."]
    elif not pred:
        risk_bucket = "pending"
        explanations = ["Processing in progress. Please refresh in a few minutes."]
    else:
        risk_bucket = pred.asd_risk_bucket
        explanations = pred.explanations or []

    return {
        "visit": {
            "id": visit.id,
            "child_id": visit.child_id,
            "visit_date": visit.visit_date,
            "age_months": visit.age_months,
        },
        "asd_risk_bucket": risk_bucket,
        "explanations": explanations,
        "prior_visits": [],
    }
