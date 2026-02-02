from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.visit import Visit
from app.models.video import Video
from app.schemas import video
from app.schemas.video import VideoOut
from app.services.file_storage import save_visit_video, ALLOWED_TASKS
from app.services.file_storage import save_json, annotations_filename
from app.services.annotations import make_placeholder_annotations
from app.services.file_storage import load_json

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
    
@router.get("/videos/{video_id}", response_model=VideoOut)
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video
