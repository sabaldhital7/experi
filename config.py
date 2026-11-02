
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"   # put the UCI file here
MODEL_PATH = BASE_DIR / "model" / "best_model.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
PLOTS_DIR = BASE_DIR / "static" / "plots"

TARGET = "G3"  # final grade (0-20)

# Features used by the model. (label, min, max, default) are used by the web form.
FEATURES = {
    "studytime": ("Weekly study time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)", 1, 4, 2),
    "failures":  ("Past class failures (0-3)", 0, 3, 0),
    "absences":  ("Number of absences", 0, 93, 4),
    "G1":        ("First terminal  (0-20)", 0, 20, 10),
    "G2":        ("Second termianl  (0-20)", 0, 20, 10),
    "age":       ("Age", 15, 22, 17),
    "health":    ("Health status (1: very bad ... 5: very good)", 1, 5, 3),
}

FEATURE_NAMES = list(FEATURES.keys())
RANDOM_STATE = 42
TEST_SIZE = 0.2
