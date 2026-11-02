
import json
import sys

import joblib
import pandas as pd
from flask import Flask, render_template, request

from config import FEATURE_NAMES, FEATURES, METRICS_PATH, MODEL_PATH

app = Flask(__name__)

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    sys.exit("Model not found. Run  python train.py  first.")

model = joblib.load(MODEL_PATH)
with open(METRICS_PATH) as f:
    metrics = json.load(f)


def parse_form(form):
    """Validate form input. Returns (values_dict, error_message)."""
    values = {}
    for name, (label, lo, hi, _default) in FEATURES.items():
        raw = form.get(name, "").strip()
        try:
            val = float(raw)
        except ValueError:
            return form, f"'{label}' must be a number."
        if not lo <= val <= hi:
            return form, f"'{label}' must be between {lo} and {hi}."
        values[name] = val
    return values, None


@app.route("/", methods=["GET", "POST"])
def index():
    prediction, error, status = None, None, None
    values = {k: v[3] for k, v in FEATURES.items()}  # defaults

    if request.method == "POST":
        values, error = parse_form(request.form)
        if error is None:
            row = pd.DataFrame([values], columns=FEATURE_NAMES)
            prediction = round(float(model.predict(row)[0]), 1)
            prediction = min(max(prediction, 0.0), 20.0)  # grades are 0-20
            status = "Pass (>= 10)" if prediction >= 10 else "At risk (< 10)"

    return render_template("index.html", features=FEATURES, values=values,
                           prediction=prediction, status=status, error=error,
                           metrics=metrics)


if __name__ == "__main__":
    app.run(debug=False)
