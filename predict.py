import pickle
from pathlib import Path
from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from model import normalize_record
import os

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).with_name("pipeline.bin"))))
THRESHOLD = 0.41

with MODEL_PATH.open('rb') as model_file:
    pipeline, defaults = pickle.load(model_file)

app = FastAPI(title = "Student Dropout Predictor")

class StudentData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: str = "NA"
    nationality: str = "NA"
    family_income_level: str = "NA"
    course_type: str = "NA"
    study_mode: str = "NA"
    age: float | None = None
    first_generation: float | None = None
    disability_status: float | None = None
    scholarship_holder: float | None = None
    tuition_up_to_date: float | None = None
    part_time_job: float | None = None
    weekly_work_hours: float | None = None
    semester: float | None = None
    international_student: float | None = None
    admission_grade: float | None = None
    previous_qualification_grade: float | None = None
    units_enrolled: float | None = None
    units_passed: float | None = None
    units_failed: float | None = None
    gpa_semester_1: float | None = None
    gpa_semester_2: float | None = None
    attendance_rate: float | None = None
    assignments_submitted_rate: float | None = None
    library_visits_per_month: float | None = None
    lms_logins_per_week: float | None = None
    extracurricular_activities: float | None = None
    missed_deadlines: float | None = None
    stress_level: float | None = None
    mental_health_support: float | None = None
    satisfaction_score: float | None = None
    commute_time_mins: float | None = None
    sleep_hours: float | None = None


@app.post("/predict")
def predict(student: StudentData) -> dict[str, Any]:
    record = normalize_record(student.model_dump(), defaults)
    probability = float(pipeline.predict_proba([record])[0, 1])

    return {
        "dropout_probability": round(probability, 6),
        "dropout": probability >= THRESHOLD
    }