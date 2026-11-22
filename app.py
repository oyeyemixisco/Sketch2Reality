"""Sketch2Future — Local Flask UI with mock or Kaggle API backend.

Launch: python app.py
Then visit: http://127.0.0.1:5000
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import secrets
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory, send_file
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

import config
from services.history import add_run, clear_history, read_history
from services.mock_pipeline import MODEL_LABELS, process_mock
from services.kaggle_api import KaggleAPIError, get_health, send_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
config.MEDIA_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    if config.BACKEND_MODE == "mock":
        return jsonify({
            "status": "online", "mode": "local_demo", "backend_connected": False,
            "gpu_connected": False, "inference_enabled": False,
            "message": "Local mock mode. Kaggle is not being used.",
        })
    try:
        remote = get_health()
        return jsonify({
            "status": "online", "mode": remote.get("mode", "kaggle_api_demo"),
            "backend_connected": True,
            "gpu_connected": bool(remote.get("gpu_available", False)),
            "inference_enabled": bool(remote.get("inference_enabled", False)),
            "message": remote.get("message", "Connected to Kaggle API."),
            "lora_available": bool(remote.get("lora_available", False)),
            "busy": bool(remote.get("busy", False)),
        })
    except KaggleAPIError as exc:
        return jsonify({
            "status": "offline", "mode": "kaggle_offline", "backend_connected": False,
            "gpu_connected": False, "inference_enabled": False,
            "message": str(exc),
        })


def _number(field: str, default, low, high, value_type=float):
    raw = request.form.get(field, str(default))
    try:
        value = value_type(raw)
        if not low <= value <= high:
            raise ValueError
        return value
    except (ValueError, TypeError):
        raise ValueError(f"Invalid {field}: expected value between {low} and {high}.")


def _parse_request():
    source = request.files.get("sketch")
    if source is None or not source.filename:
        raise ValueError("Please upload or draw a sketch first.")
    if Path(source.filename).suffix.lower() not in {".png", ".jpeg", ".jpg", ".webp"}:
        raise ValueError("Only PNG, JPG and WEBP images are supported.")
    data = source.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Sketch exceeds the {config.MAX_UPLOAD_MB} MB limit.")
    try:
        with Image.open(io.BytesIO(data)) as raw:
            if raw.width * raw.height > 30_000_000:
                raise ValueError("Image is too large. Use an image under 30 megapixels.")
            image = ImageOps.exif_transpose(raw).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc
    image.thumbnail((config.MAX_SOURCE_DIMENSION, config.MAX_SOURCE_DIMENSION), Image.Resampling.LANCZOS)

    approach = request.form.get("approach", "a1")
    if approach not in MODEL_LABELS:
        raise ValueError("Invalid model selection.")
    preprocessing = request.form.get("preprocessing", "auto")
    if preprocessing not in {"auto", "canny", "scribble", "lineart"}:
        raise ValueError("Invalid preprocessing selection.")
    preset = request.form.get("preset", "standard")
    if preset not in {"standard", "detailed", "custom"}:
        raise ValueError("Invalid quality preset.")
    prompt = request.form.get("prompt", "").strip()
    negative_prompt = request.form.get("negative_prompt", "").strip()
    if len(prompt) > 700 or len(negative_prompt) > 500:
        raise ValueError("Prompt is too long.")
    seed = _number("seed", 42, -1, 2147483647, int)
    if seed == -1:
        seed = secrets.randbelow(2147483647)
    settings = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "preset": preset,
        "seed": seed,
        "steps": _number("steps", 30, 10, 60, int),
        "guidance_scale": _number("guidance_scale", 7.5, 1.0, 15.0),
        "controlnet_scale": _number("controlnet_scale", 0.9, 0.0, 2.0),
        "lora_scale": _number("lora_scale", 0.7, 0.0, 1.5),
    }
    return image, approach, preprocessing, settings


def _save_run(image, approach, preprocessing, settings):
    run_id = uuid.uuid4().hex[:16]
    # Mock is intentionally visual only; do not pretend that a learned Lineart model ran.
    mock_mode = "canny" if preprocessing == "lineart" else preprocessing
    processed = process_mock(image, approach, mock_mode, settings)
    if preprocessing == "lineart":
        processed["explanation"]["preprocessing_reason"] = (
            "Local mock mode: the requested Lineart model is NOT executed; illustrative Canny edges were used."
        )
    source_name = f"{run_id}_sketch.png"
    edge_name = f"{run_id}_guidance.png"
    result_name = f"{run_id}_demo.png"
    image.save(config.MEDIA_DIR / source_name)
    processed["conditioning"].save(config.MEDIA_DIR / edge_name)
    processed["result"].save(config.MEDIA_DIR / result_name)
    record = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "mock",
        "approach": approach,
        "model_label": MODEL_LABELS[approach],
        "source_url": f"/api/media/{source_name}",
        "conditioning_url": f"/api/media/{edge_name}",
        "output_url": f"/api/media/{result_name}",
        "metadata": processed["metadata"],
        "explanation": processed["explanation"],
    }
    add_run(config.HISTORY_FILE, record, config.MAX_HISTORY_ITEMS)
    return record


def _decode_remote_png(value: str) -> Image.Image:
    """Reject invalid, oversized or non-image data from the remote service."""
    try:
        if not isinstance(value, str) or len(value) > 18_000_000:
            raise ValueError("Kaggle returned an invalid image payload.")
        content = base64.b64decode(value, validate=True)
        if len(content) > 13_000_000:
            raise ValueError("Kaggle returned an oversized image.")
        with Image.open(io.BytesIO(content)) as opened:
            if opened.format != "PNG" or opened.width * opened.height > 30_000_000:
                raise ValueError("Kaggle returned an unsupported image.")
            return opened.convert("RGB")
    except (binascii.Error, OSError, UnidentifiedImageError, ValueError) as exc:
        raise KaggleAPIError("Kaggle returned a corrupt or unsupported image payload.") from exc


def _save_remote_run(result, expected_approach=None):
    if not isinstance(result, dict):
        raise KaggleAPIError("Kaggle returned an invalid generation response.")
    approach = result.get("approach")
    if approach not in MODEL_LABELS or (expected_approach and approach != expected_approach):
        raise KaggleAPIError("Kaggle returned an unexpected model selection.")
    if not isinstance(result.get("metadata"), dict) or not isinstance(result.get("explanation"), dict):
        raise KaggleAPIError("Kaggle returned incomplete explanation metadata.")
    try:
        images = {
            "sketch": _decode_remote_png(result["source_png_b64"]),
            "guidance": _decode_remote_png(result["conditioning_png_b64"]),
            "demo": _decode_remote_png(result["output_png_b64"]),
        }
    except KeyError as exc:
        raise KaggleAPIError("Kaggle response is missing a generated image.") from exc
    run_id = uuid.uuid4().hex[:16]
    names = {role: f"{run_id}_{role}.png" for role in images}
    for role, image in images.items():
        image.save(config.MEDIA_DIR / names[role])
    record = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": result.get("mode", "kaggle_api_demo"),
        "approach": approach,
        "model_label": MODEL_LABELS[approach],
        "source_url": f"/api/media/{names['sketch']}",
        "conditioning_url": f"/api/media/{names['guidance']}",
        "output_url": f"/api/media/{names['demo']}",
        "metadata": result["metadata"],
        "explanation": result["explanation"],
        "lab_label": result.get("lab_label"),
    }
    add_run(config.HISTORY_FILE, record, config.MAX_HISTORY_ITEMS)
    return record


@app.post("/api/generate")
def generate():
    try:
        image, approach, preprocessing, settings = _parse_request()
        if config.BACKEND_MODE == "kaggle":
            remote = send_image(image, approach, preprocessing, settings)
            return jsonify(_save_remote_run(remote, approach))
        return jsonify(_save_run(image, approach, preprocessing, settings))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except KaggleAPIError as exc:
        return jsonify({"error": str(exc)}), exc.status_code


@app.post("/api/compare")
def compare():
    try:
        image, _approach, preprocessing, settings = _parse_request()
        if config.BACKEND_MODE == "kaggle":
            remote = send_image(image, _approach, preprocessing, settings, compare=True)
            if not isinstance(remote, dict) or not isinstance(remote.get("runs"), list):
                raise KaggleAPIError("Kaggle comparison response was invalid.")
            if [item.get("approach") for item in remote["runs"]] != list(MODEL_LABELS):
                raise KaggleAPIError("Kaggle did not return the expected three approaches.")
            runs = [_save_remote_run(item, approach) for item, approach in zip(remote["runs"], MODEL_LABELS)]
            return jsonify({"mode": remote.get("mode", "kaggle_api_demo"), "runs": runs,
                            "notice": remote.get("notice", "Check the conditioning method for each model.")})
        runs = [_save_run(image, approach, preprocessing, settings) for approach in MODEL_LABELS]
        return jsonify({"mode": "mock", "runs": runs,
                        "notice": "Comparison shows illustrative placeholders, not model performance."})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except KaggleAPIError as exc:
        return jsonify({"error": str(exc)}), exc.status_code


@app.post("/api/ablation")
def ablation():
    """Four-run LoRA experiment; no fabricated outputs when Kaggle is unavailable."""
    try:
        image, _, preprocessing, settings = _parse_request()
        if config.BACKEND_MODE != "kaggle":
            return jsonify({"error": "LoRA Lab requires the Kaggle GPU backend; mock mode cannot evaluate real LoRA weights."}), 503
        if preprocessing == "lineart" or preprocessing in {"auto", "canny", "scribble"}:
            remote = send_image(image, "a2", preprocessing, settings, ablation=True)
        else:
            raise ValueError("Invalid LoRA Lab preprocessing mode.")
        if not isinstance(remote, dict) or not isinstance(remote.get("runs"), list):
            raise KaggleAPIError("Kaggle returned invalid LoRA Lab data.")
        expected = ["A · Text only", "B · Text + LoRA", "C · Text + ControlNet", "D · Text + ControlNet + LoRA"]
        if [run.get("lab_label") for run in remote["runs"]] != expected:
            raise KaggleAPIError("Kaggle returned unexpected LoRA experiment runs.")
        runs = [_save_remote_run(result, "a2") for result in remote["runs"]]
        from PIL import ImageChops, ImageStat

        def delta(a, b):
            # 0..255 RGB mean absolute pixel difference; differences are not quality scores.
            file_a = config.MEDIA_DIR / a["output_url"].rsplit("/", 1)[-1]
            file_b = config.MEDIA_DIR / b["output_url"].rsplit("/", 1)[-1]
            with Image.open(file_a) as im_a, Image.open(file_b) as im_b:
                if im_a.size != im_b.size:
                    return None
                diff = ImageChops.difference(im_a.convert("RGB"), im_b.convert("RGB"))
                return round(sum(ImageStat.Stat(diff).mean) / 3, 4)

        differences = {
            "A_vs_B": delta(runs[0], runs[1]),
            "C_vs_D": delta(runs[2], runs[3]),
            "B_vs_D": delta(runs[1], runs[3]),
        }
        return jsonify({"mode": "ai_inference", "runs": runs,
                        "pixel_differences": differences,
                        "notice": remote.get("notice", "Pixel differences measure change, not visual quality.")})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except KaggleAPIError as exc:
        return jsonify({"error": str(exc)}), exc.status_code


@app.get("/api/history")
def history():
    runs = read_history(config.HISTORY_FILE)
    return jsonify({"runs": runs, "count": len(runs)})


@app.get("/api/runs/<run_id>")
def get_run(run_id):
    for run in read_history(config.HISTORY_FILE):
        if run.get("run_id") == run_id:
            return jsonify(run)
    abort(404)


@app.get("/api/runs/<run_id>/metadata")
def export_metadata(run_id):
    for run in read_history(config.HISTORY_FILE):
        if run.get("run_id") == run_id:
            payload = json.dumps(run, indent=2).encode("utf-8")
            return send_file(io.BytesIO(payload), mimetype="application/json",
                             as_attachment=True, download_name=f"sketch2future_{run_id}.json")
    abort(404)


@app.get("/api/media/<path:filename>")
def media(filename):
    return send_from_directory(config.MEDIA_DIR, filename)


@app.delete("/api/history")
def delete_history():
    clear_history(config.HISTORY_FILE, config.MEDIA_DIR)
    return jsonify({"status": "cleared"})


@app.errorhandler(RequestEntityTooLarge)
def too_large(_err):
    return jsonify({"error": f"Upload too large: maximum {config.MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
