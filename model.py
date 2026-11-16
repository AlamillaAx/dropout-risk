from pathlib import Path
from typing import Any

CATEGORICAL_FEATURES = (
    'gender',
    'nationality',
    'family_income_level',
    'course_type',
    'study_mode'
)

NUMERICAL_FEATURES = (
    'age',
    'first_generation',
    'disability_status',
    'scholarship_holder',
    'tuition_up_to_date',
    'part_time_job',
    'weekly_work_hours',
    'semester',
    'international_student',
    'admission_grade',
    'previous_qualification_grade',
    'units_enrolled',
    'units_passed',
    'units_failed',
    'gpa_semester_1',
    'gpa_semester_2',
    'attendance_rate',
    'assignments_submitted_rate',
    'library_visits_per_month',
    'lms_logins_per_week',
    'extracurricular_activities',
    'missed_deadlines',
    'stress_level',
    'mental_health_support',
    'satisfaction_score',
    'commute_time_mins',
    'sleep_hours'
)

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

def normalize_record(raw: dict[str, Any], numeric_defaults: dict[str, float] | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {}

    for feature in CATEGORICAL_FEATURES:
        value = raw.get(feature)

        if value is None or value == "":
            record[feature] = "NA"
        else:
            record[feature] = str(value).strip().lower().replace(" ", "_")

    for feature in NUMERICAL_FEATURES:
        value = raw.get(feature)

        if value is None or value == "":
            record[feature] = numeric_defaults[feature]
        else:
            record[feature] = float(value)

    return record