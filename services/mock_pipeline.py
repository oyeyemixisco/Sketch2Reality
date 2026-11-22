"""Honest Phase 1 demo processing — NOT Stable Diffusion, ControlNet or LoRA.

The notebook's real preprocessing and model calls are connected in Phase 2/3.
This module gives local frontend developers a working image-upload, preview,
metadata and explanation pipeline without requiring a GPU.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

MODEL_LABELS = {
    "a1": "Approach 1 · SD 1.5 + ControlNet",
    "a2": "Approach 2 · SD 1.5 + ControlNet + LoRA",
    "a3": "Approach 3 · SDXL + ControlNet",
}

COLOR_PALETTES = {
    "a1": ("#162b4b", "#8cb7ca", "#f4e6c7"),
    "a2": ("#362650", "#cb8da3", "#f9dfb9"),
    "a3": ("#203e44", "#86b6a3", "#f7e2b8"),
}


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _image_properties(image: Image.Image) -> dict[str, float]:
    """Cheap, transparent heuristic used only for the UI demonstration."""
    gray = ImageOps.grayscale(image)
    gray.thumbnail((256, 256), Image.Resampling.LANCZOS)
    hist = gray.histogram()
    count = max(1, sum(hist))
    dark = sum(hist[:115]) / count
    mid = sum(hist[65:215]) / count
    return {"dark_pixel_ratio": round(dark, 3), "midtone_ratio": round(mid, 3)}


def make_conditioning(image: Image.Image, requested: str) -> tuple[Image.Image, str, str, dict[str, float]]:
    props = _image_properties(image)
    if requested == "auto":
        # Only a pedagogical approximation of the earlier notebook's heuristic.
        selected = "canny" if props["midtone_ratio"] > 0.29 else "scribble"
        rationale = (
            "Automatic preview selected edge-style processing because the uploaded drawing "
            "contains substantial midtones or shading. This is a simplified Phase 1 heuristic."
            if selected == "canny"
            else "Automatic preview selected scribble-style processing because the drawing "
            "has relatively few midtones. This is a simplified Phase 1 heuristic."
        )
    else:
        selected = requested
        rationale = f"You manually selected {selected.title()} processing; automatic selection was bypassed."

    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    if selected == "canny":
        # Pillow FIND_EDGES is a mock edge preview; it is NOT OpenCV Canny.
        blur = gray.filter(ImageFilter.GaussianBlur(radius=0.9))
        conditioning = blur.filter(ImageFilter.FIND_EDGES)
        conditioning = ImageOps.autocontrast(conditioning)
        conditioning = conditioning.point(lambda pixel: 255 if pixel > 68 else 0)
    else:
        # Inverse threshold, mimicking the appearance of white-on-black scribble guidance.
        conditioning = gray.point(lambda pixel: 255 if pixel < 170 else 0)

    return conditioning.convert("RGB"), selected, rationale, props


def make_demo_render(image: Image.Image, model: str) -> Image.Image:
    """Stylized version of the submitted image with an unambiguous DEMO watermark.

    Do not present the output as AI generated. It is only an illustrative UI result.
    """
    dark, mid, light = COLOR_PALETTES[model]
    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    shadow = ImageOps.colorize(gray, black=dark, white=light, mid=mid, blackpoint=12, whitepoint=255)

    # Add a gentle tonal wash without inventing new scene objects.
    wash = Image.new("RGB", image.size, light)
    draw = ImageDraw.Draw(wash)
    w, h = image.size
    for y in range(h):
        t = y / max(1, h - 1)
        warm = (243, 226, 198)
        cool = (174, 204, 202)
        rgb = tuple(round((1-t)*cool[i] + t*warm[i]) for i in range(3))
        draw.line((0, y, w, y), fill=rgb)
    tinted = Image.blend(shadow, wash, 0.16)
    tinted = ImageEnhance.Contrast(tinted).enhance(1.09)
    tinted = ImageEnhance.Sharpness(tinted).enhance(1.28)

    # White-on-dark label, large enough to remain visible on desktop/mobile.
    overlay = Image.new("RGBA", tinted.size, (0, 0, 0, 0))
    box = ImageDraw.Draw(overlay)
    pad = max(12, round(w * 0.016))
    font_size = max(12, min(23, round(w * 0.024)))
    label = "DEMO PREVIEW  |  NOT AI GENERATED"
    font = _font(font_size)
    bb = box.textbbox((0, 0), label, font=font)
    label_w, label_h = bb[2]-bb[0], bb[3]-bb[1]
    x0, y0 = pad, pad
    box.rounded_rectangle((x0, y0, x0 + label_w + 24, y0 + label_h + 18), radius=10, fill=(22, 32, 49, 225))
    box.text((x0+12, y0+7), label, font=font, fill=(255, 255, 255, 255))
    return Image.alpha_composite(tinted.convert("RGBA"), overlay).convert("RGB")


def process_mock(image: Image.Image, model: str, requested_mode: str, settings: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    conditioning, selected, reason, features = make_conditioning(image, requested_mode)
    result = make_demo_render(image, model)
    duration_ms = round((time.perf_counter()-started)*1000)

    return {
        "conditioning": conditioning,
        "result": result,
        "metadata": {
            "execution_mode": "mock",
            "model_selected": MODEL_LABELS[model],
            "model_executed": "No diffusion model (Phase 1 only)",
            "preprocessing_requested": requested_mode,
            "preprocessing_preview": selected,
            "input_width": image.width,
            "input_height": image.height,
            "features": features,
            "settings": settings,
            "mock_processing_ms": duration_ms,
        },
        "explanation": {
            "summary": "This is a demonstration preview based on your uploaded drawing. "
                       "No Stable Diffusion, ControlNet, or LoRA model has run yet.",
            "preprocessing_reason": reason,
            "sketch_role": "Your uploaded sketch provided all visible shapes in this mock preview.",
            "prompt_role": "Your text prompt has been saved, but it has NOT changed the demo image. "
                           "Prompts will influence generation after Kaggle inference is connected.",
            "model_role": f"You selected {MODEL_LABELS[model]}. The model is not connected in Phase 1.",
            "limitation": "The displayed map is a Pillow-based preview, not the exact ControlNet "
                          "conditioning tensor. Interpret no image-quality or AI-fidelity scores from this demo.",
        },
    }
