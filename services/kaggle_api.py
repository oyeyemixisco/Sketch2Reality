"""Server-to-server Kaggle inference API client. Never called from frontend JS."""
from __future__ import annotations

import io
import requests
from PIL import Image

import config


class KaggleAPIError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        self.status_code = status_code
        super().__init__(message)


def is_configured() -> bool:
    return bool(config.KAGGLE_API_URL and config.KAGGLE_API_KEY)


def _base_url() -> str:
    if not is_configured():
        raise KaggleAPIError("Kaggle API is not configured. Check KAGGLE_API_URL and KAGGLE_API_KEY in your local .env.", 503)
    return config.KAGGLE_API_URL.rstrip("/")


def _headers(auth: bool = True):
    headers = {"ngrok-skip-browser-warning": "true"}
    if auth:
        headers["X-API-Key"] = config.KAGGLE_API_KEY
    return headers


def _decode_response(response):
    try:
        data = response.json()
    except ValueError as exc:
        raise KaggleAPIError(
            "Kaggle returned a non-JSON response. Check that the ngrok URL is current and the tunnel is running.", 502
        ) from exc
    if response.status_code in (401, 403):
        raise KaggleAPIError("Kaggle authentication failed. Match the API key in Kaggle Secrets and local .env.", 502)
    if not response.ok:
        message = data.get("error", "Kaggle request failed") if isinstance(data, dict) else "Kaggle request failed"
        raise KaggleAPIError(f"Kaggle API: {message}", 502)
    return data


def _request(method: str, route: str, **kwargs):
    url = f"{_base_url()}/{route.lstrip('/')}"
    try:
        response = requests.request(
            method,
            url,
            headers=_headers(),
            timeout=(config.KAGGLE_CONNECT_TIMEOUT, config.KAGGLE_READ_TIMEOUT),
            allow_redirects=False,
            **kwargs,
        )
    except requests.exceptions.Timeout as exc:
        raise KaggleAPIError(
            "Kaggle did not respond before the timeout. Large SDXL comparisons can take several minutes; check the notebook session and retry.",
            504,
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise KaggleAPIError("Cannot reach Kaggle through ngrok. Check the URL and that the Kaggle notebook is running.", 503) from exc
    return _decode_response(response)


def get_health():
    if not is_configured():
        raise KaggleAPIError("Kaggle connection is not configured. Set both KAGGLE_API_URL and KAGGLE_API_KEY in your local .env.", 503)
    try:
        response = requests.get(
            f"{config.KAGGLE_API_URL.rstrip('/')}/health",
            headers=_headers(auth=False),
            timeout=(5, 8),
            allow_redirects=False,
        )
    except requests.exceptions.RequestException as exc:
        raise KaggleAPIError("Kaggle API is offline or unreachable.", 503) from exc
    result = _decode_response(response)
    if result.get("status") != "online":
        raise KaggleAPIError("Kaggle health response does not indicate an online service.", 503)
    return result


def send_image(image: Image.Image, approach: str, preprocessing: str, settings: dict, *, compare: bool = False, ablation: bool = False):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    fields = {"approach": approach, "preprocessing": preprocessing}
    fields.update({key: str(value) for key, value in settings.items()})
    return _request(
        "POST",
        "ablation" if ablation else ("compare" if compare else "generate"),
        data=fields,
        files={"sketch": ("sketch.png", buffer, "image/png")},
    )
