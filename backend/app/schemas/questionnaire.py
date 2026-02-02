from pydantic import BaseModel

class QuestionnaireCreate(BaseModel):
    regression: bool = False
    seizures: bool = False
    motor_delay: bool = False
    global_delay: bool = False
    family_history_asd_ndd: bool = False
    dysmorphic_features: bool = False
    macrocephaly: bool = False
    microcephaly: bool = False
    notes: str | None = None

class QuestionnaireOut(QuestionnaireCreate):
    id: int
    visit_id: int

    class Config:
        from_attributes = True
