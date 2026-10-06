import io
from dataclasses import asdict
from flask import Flask, render_template, request, jsonify, send_file
from generator.engine import generate_dataset
from generator.presets import PRESETS
from generator.ml_evaluator import train_and_evaluate_baseline

app = Flask(__name__)


def serialize_column(col):
    """Convert a Column dataclass to a JSON-friendly dict."""
    d = asdict(col)
    d['type'] = d.pop('dtype')
    return d


def serialize_preset(preset):
    """Convert a preset dict (with Column objects) to a JSON-friendly dict."""
    return {
        "name": preset["name"],
        "description": preset["description"],
        "target_churn_rate": preset["target_churn_rate"],
        "noise_level": preset["noise_level"],
        "columns": [serialize_column(c) for c in preset["columns"]],
        "relationships": preset["relationships"],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/presets", methods=["GET"])
def get_presets():
    preset_list = [
        {"name": key, "description": val["description"]}
        for key, val in PRESETS.items()
    ]
    return jsonify(preset_list)


@app.route("/api/presets/<name>", methods=["GET"])
def get_preset(name):
    name_lower = name.lower()
    if name_lower in PRESETS:
        return jsonify(serialize_preset(PRESETS[name_lower]))
    return jsonify({"error": "Preset not found"}), 404


MAX_GENERATE_ROWS = 500_000

@app.route("/api/preview", methods=["POST"])
def preview():
    config = request.get_json() or {}
    
    # Selectable preview rows from 5 to 200
    requested_preview = config.get("preview_rows") or config.get("num_rows") or 50
    try:
        preview_rows = max(5, min(int(requested_preview), 200))
    except (ValueError, TypeError):
        preview_rows = 50
        
    config["num_rows"] = preview_rows

    try:
        df = generate_dataset(config)
        columns = df.columns.tolist()
        data = df.to_dict(orient="records")
        actual_churn_rate = float(df["churn"].mean()) if "churn" in df.columns and len(df) > 0 else 0.0
        rule_stats = df.attrs.get("rule_stats", [])
        invariants_report = df.attrs.get("invariants_report", {})

        return jsonify({
            "columns": columns,
            "data": data,
            "stats": {
                "actual_churn_rate": actual_churn_rate,
                "num_rows": len(df),
                "num_cols": len(columns),
                "rule_stats": rule_stats,
                "invariants_report": invariants_report,
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "status": 400}), 400


@app.route("/api/train_baseline", methods=["POST"])
def train_baseline():
    config = request.get_json() or {}
    
    try:
        # Sample size for fast baseline training (clamped strictly between 200 and 10,000)
        raw_train_rows = config.get("train_rows", 2500)
        try:
            train_rows = max(200, min(int(raw_train_rows), 10000))
        except (ValueError, TypeError):
            return jsonify({"error": f"Invalid 'train_rows': {raw_train_rows}. Must be an integer between 200 and 10000.", "status": 400}), 400

        model_type = config.get("model_type", "hist_gb")
        if model_type not in ["hist_gb", "logistic_regression"]:
            return jsonify({
                "error": f"Invalid 'model_type' '{model_type}'. Supported models: ['hist_gb', 'logistic_regression'].",
                "status": 400
            }), 400
        
        gen_config = config.copy()
        gen_config["num_rows"] = train_rows
        
        df = generate_dataset(gen_config)
        evaluation = train_and_evaluate_baseline(df, model_type=model_type)
        return jsonify(evaluation)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "status": 400}), 400


@app.route("/api/generate", methods=["POST"])
def generate():
    config = request.get_json() or {}
    try:
        raw_rows = config.get("num_rows", 10000)
        try:
            num_rows = int(raw_rows)
        except (ValueError, TypeError):
            return jsonify({"error": f"Invalid 'num_rows': {raw_rows}. Must be an integer.", "status": 400}), 400

        if num_rows > MAX_GENERATE_ROWS:
            return jsonify({
                "error": f"Requested row count ({num_rows:,}) exceeds the maximum in-memory export ceiling ({MAX_GENERATE_ROWS:,}).",
                "status": 400
            }), 400

        df = generate_dataset(config)

        # In-memory streaming: avoids Windows file locking and orphaned temp files
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        preset_tag = config.get("preset_name", "custom")
        download_name = f"{preset_tag}_churn_dataset.csv"

        return send_file(
            csv_buffer,
            mimetype="text/csv",
            as_attachment=True,
            download_name=download_name
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "status": 400}), 400


if __name__ == "__main__":
    app.run(debug=True)

