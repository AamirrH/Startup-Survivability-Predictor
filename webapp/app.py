from datetime import date
import math
import os
import numpy as np
from flask import Flask, jsonify, render_template, request
from waitress import serve
from webapp.model import NUMERIC_COLUMNS, load_model, predict_startup


def validate_inputs(data, metadata):
    features = {}
    errors = {}
    warnings = []
    current_year = date.today().year
    for field in NUMERIC_COLUMNS:
        value = data.get(field)
        if value is None or value == "":
            features[field] = np.nan
            continue
        try:
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError
            number = float(value)
            if not math.isfinite(number):
                raise ValueError
            if field == "funding_total_usd":
                valid = 0 <= number <= 1e12
            elif field == "funding_rounds":
                valid = number.is_integer() and 1 <= number <= 1000
            else:
                valid = number.is_integer() and 1800 <= number <= current_year
            if not valid:
                raise ValueError
            features[field] = number
        except (ValueError, TypeError, OverflowError):
            if field == "funding_total_usd":
                errors[field] = "Enter a funding amount from 0 to 1 trillion USD, or leave it blank."
            elif field == "funding_rounds":
                errors[field] = "Enter a whole number from 1 to 1,000, or leave it blank."
            else:
                errors[field] = f"Enter a whole year from 1800 to {current_year}, or leave it blank."

    allowed_countries = {country["code"] for country in metadata["countries"]}
    choices = {"country_code": allowed_countries, "first_category": set(metadata["categories"])}
    for field, allowed in choices.items():
        value = data.get(field)
        if value is None or value == "":
            features[field] = np.nan
        elif not isinstance(value, str) or value not in allowed:
            errors[field] = "Choose an available option, or select Other / unknown."
        else:
            features[field] = value

    name = data.get("startup_name", "")
    if not isinstance(name, str) or len(name) > 80:
        errors["startup_name"] = "Use a startup name of 80 characters or fewer."
    if errors:
        return features, errors, warnings

    if features["first_funding_at_year"] > features["last_funding_at_year"]:
        errors["last_funding_at_year"] = "Last funding year must be the same as or later than first funding year."
    if features["founded_at_year"] > features["last_funding_at_year"]:
        errors["founded_at_year"] = "Check the founding year, or leave it unknown if funding predates incorporation."
    known = sum(not (isinstance(value, float) and math.isnan(value)) for value in features.values())
    if known == 0:
        errors["form"] = "Enter at least one startup detail before running a prediction."
    elif known < 3:
        warnings.append("Most details are unknown. This result relies heavily on typical training values.")
    if any(features[field] > metadata["last_training_year"] for field in NUMERIC_COLUMNS if field.endswith("_year")):
        warnings.append("These dates are newer than the 2015 training snapshot. Performance on newer startups has not been validated.")
    if features["funding_total_usd"] > metadata["max_funding"] or features["funding_rounds"] > metadata["max_rounds"]:
        warnings.append("Funding is outside the range seen in training. Interpret this result cautiously.")
    return features, errors, warnings


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    model, metadata = load_model()

    @app.get("/")
    def index():
        return render_template("index.html", metadata=metadata, current_year=date.today().year)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "Random Forest"}

    @app.post("/predict")
    def predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(errors={"form": "Send the startup details as a JSON object."}), 400
        features, errors, warnings = validate_inputs(data, metadata)
        if errors:
            return jsonify(errors=errors), 400
        result = predict_startup(model, features)
        result["startup_name"] = data.get("startup_name", "").strip() or "Your startup"
        result["warnings"] = warnings
        result["known_features"] = sum(not (isinstance(value, float) and math.isnan(value)) for value in features.values())
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(errors={"form": "That request is too large. Enter only the startup details."}), 413

    @app.after_request
    def response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        if request.path == "/predict":
            response.headers["Cache-Control"] = "no-store"
        return response

    return app


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"Startup outlook is running at http://{host}:{port}", flush=True)
    serve(create_app(), host=host, port=port, threads=4)
