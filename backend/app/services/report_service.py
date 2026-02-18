from sqlalchemy.orm import Session, joinedload
from app.models.visit import Visit
from app.models.ml_prediction import MLPrediction
from app.models.video import Video
from app.services.explanation_service import format_explanations

REQUIRED_TASKS = {"imitation", "joint_attention", "free_play"}


def _categorize_explanation(explanation: str) -> str:
    """Categorize an explanation by its source task type."""
    exp_lower = explanation.lower()

    # Joint attention indicators
    ja_keywords = [
        "attention bid", "gaze-following", "follow point", "orienting to name",
        "social cues", "stillness during joint attention", "joint attention"
    ]
    for kw in ja_keywords:
        if kw in exp_lower:
            return "joint_attention"

    # Imitation indicators
    imit_keywords = [
        "imitation", "arm-raise", "clapping", "demonstrated action"
    ]
    for kw in imit_keywords:
        if kw in exp_lower:
            return "imitation"

    # Free play indicators
    fp_keywords = [
        "free play", "repetitive motion", "hand-to-face", "eye contact during free",
        "social engagement"
    ]
    for kw in fp_keywords:
        if kw in exp_lower:
            return "free_play"

    # Questionnaire/general
    if "caregiver" in exp_lower or "reported" in exp_lower:
        return "questionnaire"

    return "general"


def _categorize_explanations(explanations: list) -> dict:
    """Categorize explanations by task type."""
    categorized = {
        "joint_attention": [],
        "imitation": [],
        "free_play": [],
        "questionnaire": [],
        "general": [],
    }
    for exp in explanations:
        task_type = _categorize_explanation(exp)
        categorized[task_type].append(exp)
    return categorized


def build_report(db: Session, visit_id: int) -> dict:
    visit = db.query(Visit).filter(Visit.id == visit_id).first()
    if not visit:
        return {}

    # Check uploaded tasks
    videos = db.query(Video).filter(Video.visit_id == visit_id).all()
    tasks_present = {v.task_type for v in videos}
    has_all_videos = REQUIRED_TASKS.issubset(tasks_present)

    pred = db.query(MLPrediction).filter(MLPrediction.visit_id == visit_id).first()

    risk_score = None
    if not has_all_videos:
        risk_bucket = "insufficient_data"
        explanations = ["Upload all 3 behavioral videos to compute ASD risk."]
    elif not pred:
        risk_bucket = "pending"
        explanations = ["Processing in progress. Please refresh in a few minutes."]
    else:
        risk_bucket = pred.asd_risk_bucket
        risk_score = pred.probability
        explanations = format_explanations(pred.explanations or [])

    # Fetch all visits for this child that have predictions (for timeline)
    all_visits = (
        db.query(Visit)
        .options(joinedload(Visit.ml_prediction))
        .filter(Visit.child_id == visit.child_id)
        .order_by(Visit.age_months.asc())
        .all()
    )

    prior_visits = []
    for v in all_visits:
        if v.ml_prediction is None:
            continue
        prior_visits.append({
            "id": v.id,
            "age_months": v.age_months,
            "visit_date": v.visit_date,
            "asd_risk_bucket": v.ml_prediction.asd_risk_bucket,
            "risk_score": v.ml_prediction.probability,
            "visit_number": v.visit_number,
            "is_current": v.id == visit.id,
        })

    # Categorize explanations by task type for frontend display
    categorized = _categorize_explanations(explanations) if pred else {}

    return {
        "visit": {
            "id": visit.id,
            "child_id": visit.child_id,
            "visit_date": visit.visit_date,
            "age_months": visit.age_months,
        },
        "asd_risk_bucket": risk_bucket,
        "risk_score": risk_score,
        "explanations": explanations,
        "explanations_by_task": categorized,  # Categorized by task type
        "prior_visits": prior_visits,
    }
