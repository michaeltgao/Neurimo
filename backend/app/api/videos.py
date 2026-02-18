from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.visit import Visit
from app.models.video import Video
from app.schemas import video
from app.schemas.video import VideoOut, VideoWithVisitOut
from app.services.file_storage import save_visit_video, ALLOWED_TASKS
from app.services.file_storage import save_json, annotations_filename
from app.services.annotations import make_placeholder_annotations
from app.services.file_storage import load_json
from app.services.guided_review import build_guided_review_data, get_overlay_data

router = APIRouter(tags=["videos"])

@router.post("/visits/{visit_id}/videos", response_model=VideoOut)
async def upload_video(
    visit_id: str,
    task_type: str = Query(..., description="imitation | joint_attention | free_play"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if task_type not in ALLOWED_TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task_type. Must be one of {sorted(ALLOWED_TASKS)}")

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

    try:
        storage_path = await save_visit_video(visit_id, task_type, file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check for existing video with same visit_id and task_type
    existing = db.query(Video).filter(
        Video.visit_id == visit.id,
        Video.task_type == task_type
    ).first()

    if existing:
        # Update existing record
        existing.storage_path = storage_path
        existing.status = "uploaded"
        db.commit()
        db.refresh(existing)
        return existing

    # Create new record
    video = Video(
        visit_id=visit.id,
        task_type=task_type,
        storage_path=storage_path,
        status="uploaded",
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    rel = f"annotations/{annotations_filename(video.id)}"
    ann_path = save_json(rel, make_placeholder_annotations(video.id, task_type))
    video.annotations_path = ann_path
    video.annotations_version = "v1"
    db.commit()

    return video

@router.get("/visits/{visit_id}/videos", response_model=list[VideoOut])
def list_videos_for_visit(visit_id: str, db: Session = Depends(get_db)):
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
        return []

    videos = (
        db.query(Video)
        .filter(Video.visit_id == visit.id)
        .order_by(Video.created_at.asc())
        .all()
    )
    return videos

@router.get("/children/{child_id}/videos", response_model=list[VideoOut])
def list_videos_for_child(child_id: int, db: Session = Depends(get_db)):
    # join visits so we only pull this child's videos
    videos = (
        db.query(Video)
        .join(Visit, Video.visit_id == Visit.id)
        .filter(Visit.child_id == child_id)
        .order_by(Visit.visit_date.asc(), Video.created_at.asc())
        .all()
    )
    return videos

@router.get("/videos/{video_id}/annotations")
def get_video_annotations(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if not video.annotations_path:
        raise HTTPException(status_code=404, detail="Annotations not found for this video")

    try:
        return load_json(video.annotations_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Annotations file missing on disk",
        )
    
@router.get("/videos/{video_id}", response_model=VideoWithVisitOut)
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    visit = db.query(Visit).filter(Visit.id == video.visit_id).first()
    if not visit:
        raise HTTPException(status_code=404, detail="Visit not found")

    return VideoWithVisitOut(
        id=video.id,
        visit_id=video.visit_id,
        task_type=video.task_type,
        storage_path=video.storage_path,
        status=video.status,
        created_at=video.created_at,
        child_id=visit.child_id,
        visit_number=visit.visit_number,
    )


@router.get("/videos/{video_id}/guided-review")
def get_guided_review_data(
    video_id: int,
    duration_ms: float = Query(10000, description="Video duration in milliseconds"),
    db: Session = Depends(get_db)
):
    """
    Get guided review data for a video.

    Transforms ML pipeline outputs (audio events, pose tracks, behavioral events)
    into the GuidedReviewData format expected by the frontend.

    Query params:
        duration_ms: Video duration in milliseconds (frontend probes this)
    """
    # Get video and its visit
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    visit = db.query(Visit).filter(Visit.id == video.visit_id).first()
    if not visit:
        raise HTTPException(status_code=404, detail="Visit not found")

    # Compute age bucket from visit age_months
    age_months = visit.age_months or 18
    if age_months < 12:
        age_bucket = "0-12 months"
    elif age_months < 24:
        age_bucket = "12-24 months"
    elif age_months < 36:
        age_bucket = "24-36 months"
    else:
        age_bucket = "36+ months"

    # Build video URL
    import os
    base_url = os.getenv("VITE_API_BASE_URL", "http://localhost:8000")
    storage_path = video.storage_path

    # Handle different path formats
    if storage_path.startswith("/static/"):
        video_url = f"{base_url}{storage_path}"
    elif "data/" in storage_path:
        idx = storage_path.find("data/")
        rel = storage_path[idx + len("data/"):]
        video_url = f"{base_url}/static/{rel}"
    else:
        video_url = f"{base_url}/static/{storage_path}"

    # Build guided review data from ML pipeline
    try:
        data = build_guided_review_data(
            video_id=video_id,
            visit_id=visit.id,
            task_type=video.task_type,
            video_url=video_url,
            duration_ms=duration_ms,
            age_bucket=age_bucket,
        )
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build guided review data: {str(e)}"
        )


@router.get("/videos/{video_id}/overlay-data")
def get_video_overlay_data(
    video_id: int,
    duration_ms: float = Query(10000, description="Video duration in milliseconds"),
    db: Session = Depends(get_db)
):
    """
    Get frame-level overlay data for video annotation visualization.

    Returns pose landmarks, bounding boxes, head orientation, and event data
    for rendering real-time overlays on the video player.

    Query params:
        duration_ms: Video duration in milliseconds
    """
    # Get video and its visit
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    visit = db.query(Visit).filter(Visit.id == video.visit_id).first()
    if not visit:
        raise HTTPException(status_code=404, detail="Visit not found")

    try:
        data = get_overlay_data(
            video_id=video_id,
            visit_id=visit.id,
            task_type=video.task_type,
            duration_ms=duration_ms,
        )
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get overlay data: {str(e)}"
        )
