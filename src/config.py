from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = PROJECT_ROOT / "models"

DEFAULT_RAW_DATA = DATA_RAW / "diabetes.csv"
DEFAULT_CLEAN_DATA = DATA_PROCESSED / "cleaned_diabetes.csv"
DEFAULT_SPLIT_DATA = MODELS / "train_test_data.joblib"
DEFAULT_METRICS = REPORTS / "metrics.json"

TARGET_CANDIDATES = ("Outcome", "target", "label", "class", "diagnosis")
GLUCOSE_CANDIDATES = ("Glucose", "glucose")
BP_CANDIDATES = (
    "BloodPressure",
    "Blood_Pressure",
    "blood_pressure",
    "bloodpressure",
    "BP",
    "bp",
)
