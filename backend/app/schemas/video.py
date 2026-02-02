from datetime import datetime
from pydantic import BaseModel

class VideoOut(BaseModel):
    id: int
    visit_id: int
    task_type: str
    storage_path: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True
