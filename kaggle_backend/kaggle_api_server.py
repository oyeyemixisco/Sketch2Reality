"""Sketch2Future XAI Studio: Kaggle GPU inference HTTP API.

Single source of truth. The accompanying notebook writes this exact code to
/kaggle/working/kaggle_api_server.py and imports the module before starting ngrok.

A1: SD1.5 + ControlNet (Auto: Canny / Scribble / Lineart)
A2: SD1.5 + ControlNet + existing Kohya LoRA (same Auto modes)
A3: SDXL + ControlNet (working standalone Auto: Canny / Scribble)
LoRA Lab: A text only; B text+LoRA; C CN only; D CN+LoRA.
"""
from __future__ import annotations

import base64
import gc
import hmac
import io
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

SD15_BASE = os.getenv('SKETCH2FUTURE_SD15_BASE', 'stable-diffusion-v1-5/stable-diffusion-v1-5')
SDXL_BASE = os.getenv('SKETCH2FUTURE_SDXL_BASE', 'stabilityai/stable-diffusion-xl-base-1.0')
SD15_CONTROLNET = {
    'canny': 'lllyasviel/control_v11p_sd15_canny',
    'scribble': 'lllyasviel/control_v11p_sd15_scribble',
    'lineart': 'lllyasviel/control_v11p_sd15_lineart',
}
SDXL_CONTROLNET = {
    'canny': 'diffusers/controlnet-canny-sdxl-1.0',
    'scribble': 'xinsir/controlnet-scribble-sdxl-1.0',
}
MODEL_LABELS = {
    'a1': 'Approach 1 · SD 1.5 + ControlNet',
    'a2': 'Approach 2 · SD 1.5 + ControlNet + LoRA',
    'a3': 'Approach 3 · SDXL + ControlNet',
}
DEFAULT_NEGATIVE = 'low quality, blurry, distorted geometry, duplicate objects, unrealistic image'
HF_TOKEN = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
LORA_PATH = os.getenv('SKETCH2FUTURE_LORA_PATH', '').strip()
MAX_UPLOAD_MB = 10
MAX_SOURCE_PIXELS = 30_000_000
_LINEART_DETECTOR = None
INFERENCE_LOCK = threading.Lock()


def _encode(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, 'PNG', optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def _resize_sd15(image: Image.Image, long_side: int = 768) -> Image.Image:
    """Preserves the size policy of Approach 1/2 procedural notebooks."""
    image = image.convert('RGB')
    w, h = image.size
    scale = min(1.0, long_side / max(w, h))
    nw = max(64, int(round((w * scale) / 8)) * 8)
    nh = max(64, int(round((h * scale) / 8)) * 8)
    return image.resize((nw, nh), Image.Resampling.LANCZOS)


def _select_sd15(image: Image.Image, requested: str):
    gray = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    metrics = {
        'stroke_density': round(float((otsu > 0).mean()), 4),
        'midtone_ratio': round(float(((gray > 35) & (gray < 220)).mean()), 4),
        'intensity_std': round(float(gray.std()), 2),
        'edge_ratio': round(float((cv2.Canny(gray, 100, 200) > 0).mean()), 4),
    }
    if requested != 'auto':
        return requested, metrics, f'You manually selected {requested.title()}; Auto selection was bypassed.'
    if metrics['midtone_ratio'] > .22 and metrics['stroke_density'] <= .65:
        return 'lineart', metrics, 'Auto chose Lineart: midtone ratio > 0.22 and stroke density <= 0.65. This is a heuristic, not a learned quality prediction.'
    if metrics['midtone_ratio'] > .22 or metrics['stroke_density'] > .24 or metrics['intensity_std'] > 48:
        return 'canny', metrics, 'Auto chose Canny: shading, stroke density, or intensity variation exceeded the configured thresholds.'
    return 'scribble', metrics, 'Auto chose Scribble: the sketch fell below the configured shading/density thresholds.'


def _condition_sd15(image: Image.Image, method: str) -> Image.Image:
    global _LINEART_DETECTOR
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    if method == 'canny':
        return Image.fromarray(cv2.Canny(gray, 100, 200)).convert('RGB')
    if method == 'scribble':
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
        threshold_type = cv2.THRESH_BINARY_INV if np.median(border) >= 128 else cv2.THRESH_BINARY
        _, mask = cv2.threshold(blur, 0, 255, threshold_type + cv2.THRESH_OTSU)
        return Image.fromarray(mask).convert('RGB')
    if method == 'lineart':
        if _LINEART_DETECTOR is None:
            from controlnet_aux import LineartDetector
            _LINEART_DETECTOR = LineartDetector.from_pretrained('lllyasviel/Annotators')
        return _LINEART_DETECTOR(image, coarse=False).convert('RGB')
    raise ValueError('Unsupported SD1.5 preprocessing.')


def _resize_sdxl(image: Image.Image, long_side: int = 1024) -> Image.Image:
    """Preserves resize behaviour from working standalone Approach 3 notebook."""
    image = ImageOps.exif_transpose(image).convert('RGB')
    w, h = image.size
    scale = long_side / max(w, h)
    nw = max(8, int(round((w * scale) / 8)) * 8)
    nh = max(8, int(round((h * scale) / 8)) * 8)
    if min(nw, nh) < 128:
        raise ValueError('Sketch is extremely wide or tall. Crop it to include the main subject.')
    return image.resize((nw, nh), Image.Resampling.LANCZOS)


def _prepare_sdxl(image: Image.Image, requested: str):
    """Identical method/threshold policy to validated standalone SDXL notebook."""
    resized = _resize_sdxl(image)
    gray = cv2.cvtColor(np.asarray(resized), cv2.COLOR_RGB2GRAY)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    light_background = np.median(border) >= 128
    blurred_otsu = cv2.GaussianBlur(gray, (3, 3), .8)
    mode = cv2.THRESH_BINARY_INV if light_background else cv2.THRESH_BINARY
    _, scribble = cv2.threshold(blurred_otsu, 0, 255, mode + cv2.THRESH_OTSU)
    blurred_canny = cv2.GaussianBlur(gray, (5, 5), 1.6)
    canny = cv2.Canny(blurred_canny, 65, 150, L2gradient=True)
    metrics = {
        'midtone_fraction': round(float(((gray >= 65) & (gray <= 235)).mean()), 4),
        'scribble_density': round(float((scribble > 0).mean()), 4),
        'canny_density': round(float((canny > 0).mean()), 4),
        'light_background': bool(light_background),
        'processed_resolution': list(resized.size),
    }
    if requested in {'canny', 'scribble'}:
        selected, reason = requested, 'You manually selected this method; Auto selection was bypassed.'
    elif requested == 'auto':
        dense = (metrics['scribble_density'] > .18 or
                 (metrics['midtone_fraction'] > .18 and metrics['scribble_density'] > .08))
        if dense and metrics['canny_density'] >= .001:
            selected, reason = 'canny', 'Shading/filled regions detected: prefer sparse Canny edges.'
        elif metrics['scribble_density'] >= .001:
            selected, reason = 'scribble', 'Clean or relatively sparse line drawing: prefer Otsu Scribble.'
        elif metrics['canny_density'] >= .001:
            selected, reason = 'canny', 'Otsu produced very few strokes: using Canny edges.'
        else:
            raise ValueError('No meaningful lines found. Try a higher-contrast sketch.')
    else:
        raise ValueError('SDXL supports Auto, Canny or Scribble, not Lineart.')
    candidates = {'canny': Image.fromarray(canny).convert('RGB'),
                  'scribble': Image.fromarray(scribble).convert('RGB')}
    return resized, candidates[selected], selected, metrics, reason


def preprocess(image: Image.Image, requested: str, approach: str, *, comparing: bool = False):
    if requested not in {'auto', 'canny', 'scribble', 'lineart'}:
        raise ValueError('Invalid preprocessing mode.')
    fallback = False
    if approach == 'a3':
        if requested == 'lineart':
            if not comparing:
                raise ValueError('Lineart is supported for Approaches 1 and 2 only; use Auto, Canny, or Scribble for SDXL.')
            requested_sdxl = 'auto'
            fallback = True
        else:
            requested_sdxl = requested
        source, control, selected, metrics, reason = _prepare_sdxl(image, requested_sdxl)
        if fallback:
            reason = 'Lineart is SD1.5-only. SDXL used its own Auto Canny/Scribble selection. ' + reason
        return source, control, selected, metrics, reason, fallback
    if approach not in {'a1', 'a2'}:
        raise ValueError('Invalid approach.')
    source = _resize_sd15(image)
    selected, metrics, reason = _select_sd15(image, requested)
    return source, _condition_sd15(source, selected), selected, metrics, reason, False


@dataclass
class PipelineBundle:
    key: str
    pipe: Any
    base_model: str
    controlnet_model: Optional[str]
    lora_loaded: bool = False


class PipelineManager:
    """One active pipeline to keep Kaggle GPU VRAM usage bounded."""
    def __init__(self):
        self.bundle: Optional[PipelineBundle] = None

    def clear(self):
        if self.bundle is not None:
            try:
                self.bundle.pipe.maybe_free_model_hooks()
            except Exception:
                pass
            self.bundle = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def _device_and_kwargs():
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        kwargs = {'torch_dtype': torch.float16 if device == 'cuda' else torch.float32}
        if HF_TOKEN:
            kwargs['token'] = HF_TOKEN
        return device, kwargs

    @staticmethod
    def _prepare_pipe(pipe, device):
        from diffusers import UniPCMultistepScheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        if device == 'cuda':
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

    def get(self, key: str, *, approach: str, preprocessing: Optional[str] = None) -> PipelineBundle:
        if self.bundle and self.bundle.key == key:
            return self.bundle
        self.clear()
        from diffusers import (
            AutoencoderKL, ControlNetModel, StableDiffusionPipeline,
            StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline,
        )
        device, kwargs = self._device_and_kwargs()
        if approach == 'a3':
            controlnet_id = SDXL_CONTROLNET[preprocessing]
            # Match working standalone SDXL notebook: no variant='fp16' for ControlNet.
            cn = ControlNetModel.from_pretrained(controlnet_id, use_safetensors=True, **kwargs)
            vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', use_safetensors=True, **kwargs)
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                SDXL_BASE, controlnet=cn, vae=vae, use_safetensors=True, **kwargs
            )
            self._prepare_pipe(pipe, device)  # UniPC, same as standalone notebook.
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            bundle = PipelineBundle(key, pipe, SDXL_BASE, controlnet_id)
        else:
            cn_id = SD15_CONTROLNET[preprocessing] if preprocessing else None
            if cn_id:
                cn = ControlNetModel.from_pretrained(cn_id, **kwargs)
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    SD15_BASE, controlnet=cn, safety_checker=None,
                    requires_safety_checker=False, **kwargs,
                )
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    SD15_BASE, safety_checker=None, requires_safety_checker=False, **kwargs,
                )
            self._prepare_pipe(pipe, device)
            bundle = PipelineBundle(key, pipe, SD15_BASE, cn_id)
        self.bundle = bundle
        return bundle

    def enable_lora(self, bundle: PipelineBundle, scale: float):
        path = Path(LORA_PATH)
        if not LORA_PATH or not path.is_file():
            raise FileNotFoundError(
                'LoRA checkpoint missing. Set SKETCH2FUTURE_LORA_PATH to the existing .safetensors file in Kaggle.'
            )
        if not bundle.lora_loaded:
            # Use the working procedural Approach 2 V2 loading path, including Kohya LoRA keys.
            bundle.pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name='sketch2future')
            bundle.lora_loaded = True
        bundle.pipe.set_adapters(['sketch2future'], adapter_weights=[float(scale)])


MANAGER = PipelineManager()


def _generator(seed: int):
    import torch
    return torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(seed)


def _invoke(bundle: PipelineBundle, *, prompt: str, negative_prompt: str,
            conditioning: Optional[Image.Image], settings: dict) -> Image.Image:
    args = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'num_inference_steps': int(settings['steps']),
        'guidance_scale': float(settings['guidance_scale']),
        'generator': _generator(int(settings['seed'])),
        'num_images_per_prompt': 1,
    }
    if conditioning is not None:
        args['image'] = conditioning
        args['width'] = conditioning.width
        args['height'] = conditioning.height
        args['controlnet_conditioning_scale'] = float(settings['controlnet_scale'])
    else:
        args['width'] = int(settings['width'])
        args['height'] = int(settings['height'])
    import torch
    with torch.inference_mode():
        return bundle.pipe(**args).images[0].convert('RGB')


def _record(source, conditioning, output, *, approach, preprocessing_requested,
            preprocessing_selected, metrics, reason, bundle, settings,
            elapsed_ms, fallback=False, lab_label=None, lora_active=False):
    mode = 'ai_inference'
    effective = dict(settings)
    effective['prompt'] = settings['prompt'] or 'photorealistic image guided by the uploaded sketch'
    meta = {
        'execution_mode': mode,
        'model_executed': bundle.base_model,
        'controlnet_model': bundle.controlnet_model,
        'preprocessing_requested': preprocessing_requested,
        'preprocessing_selected': preprocessing_selected or 'none',
        'preprocessing_auto_metrics': metrics,
        'preprocessing_fallback': bool(fallback),
        'preprocessor_model': 'lllyasviel/Annotators (LineartDetector)' if preprocessing_selected == 'lineart' else None,
        'input_width': source.width,
        'input_height': source.height,
        'output_width': output.width,
        'output_height': output.height,
        'generation_ms': elapsed_ms,
        'settings': effective,
        'effective_controlnet_scale': float(settings['controlnet_scale']) if conditioning is not None else None,
        'lora_loaded': lora_active,
        'lora_checkpoint_name': Path(LORA_PATH).name if lora_active else None,
        'lora_scale_effective': float(settings['lora_scale']) if lora_active else 0,
        'scheduler': type(bundle.pipe.scheduler).__name__,
        'preprocessing_preview': f'{preprocessing_selected} control image' if conditioning is not None else 'No ControlNet (text-to-image)',
        'lab_label': lab_label,
        'backend': 'Kaggle GPU Flask API',
    }
    exp = {
        'summary': ('This is a controlled LoRA ablation run. ' if lab_label else '') +
                   'The sketch was converted to a guidance map where ControlNet was used; the text prompt supplied visual context.',
        'preprocessing_reason': reason,
        'prompt_role': 'The prompt and effective negative prompt guide semantic content and appearance; the diffusion model may add or alter details.',
        'model_role': (f'Base: {bundle.base_model}; ControlNet: {bundle.controlnet_model or "none"}; '
                       f'LoRA enabled: {lora_active}; adapter weight: {settings["lora_scale"] if lora_active else 0}.'),
        'limitation': ('Auto selection is a heuristic, not a learned certainty score. '
                       'This is process-level explainability, not an explanation of every internal diffusion decision. '
                       'LoRA activity does not by itself prove quality improvement.'),
    }
    return {
        'approach': approach,
        'model_label': MODEL_LABELS[approach],
        'mode': mode,
        'source_png_b64': _encode(source),
        'conditioning_png_b64': _encode(conditioning if conditioning is not None else source),
        'output_png_b64': _encode(output),
        'metadata': meta,
        'explanation': exp,
        'lab_label': lab_label,
    }


def generate_result(image, approach: str, preprocessing: str, settings: dict, *, comparing=False):
    started = time.perf_counter()
    source, conditioning, selected, metrics, reason, fallback = preprocess(
        image, preprocessing, approach, comparing=comparing
    )
    bundle = MANAGER.get(f'{approach}:{selected}', approach=approach, preprocessing=selected)
    if approach == 'a2':
        MANAGER.enable_lora(bundle, settings['lora_scale'])
    prompt = settings['prompt'] or 'photorealistic image guided by the uploaded sketch'
    output = _invoke(bundle, prompt=prompt, negative_prompt=settings['negative_prompt'],
                     conditioning=conditioning, settings=settings)
    return _record(source, conditioning, output, approach=approach,
                   preprocessing_requested=preprocessing, preprocessing_selected=selected,
                   metrics=metrics, reason=reason, bundle=bundle, settings=settings,
                   elapsed_ms=round((time.perf_counter() - started) * 1000),
                   fallback=fallback, lora_active=approach == 'a2')


def run_lora_lab(image: Image.Image, preprocessing: str, settings: dict, *, baselines=True):
    """Four-run LoRA ablation, using a matched seed & settings in every run."""
    source, conditioning, selected, metrics, reason, _ = preprocess(image, preprocessing, 'a2')
    run_settings = dict(settings)
    run_settings['width'], run_settings['height'] = source.size
    prompt = settings['prompt'] or 'photorealistic image guided by the uploaded sketch'
    runs = []

    def run_one(key, *, bundle, cond=None, lora_active=False, note=''):
        started = time.perf_counter()
        output = _invoke(bundle, prompt=prompt, negative_prompt=settings['negative_prompt'],
                         conditioning=cond, settings=run_settings)
        record = _record(source, cond, output, approach='a2',
                         preprocessing_requested=preprocessing,
                         preprocessing_selected=selected if cond is not None else None,
                         metrics=metrics if cond is not None else {},
                         reason=reason if cond is not None else 'Text-to-image only: no sketch/control image was supplied to a ControlNet.',
                         bundle=bundle, settings=run_settings,
                         elapsed_ms=round((time.perf_counter() - started) * 1000),
                         lab_label=key, lora_active=lora_active)
        record['model_label'] = key
        runs.append(record)

    try:
        text_bundle = MANAGER.get('lab:text', approach='a2', preprocessing=None)
        if baselines:
            run_one('A · Text only', bundle=text_bundle)
        MANAGER.enable_lora(text_bundle, settings['lora_scale'])
        run_one('B · Text + LoRA', bundle=text_bundle, lora_active=True)

        cn_bundle = MANAGER.get(f'lab:cn:{selected}', approach='a2', preprocessing=selected)
        if baselines:
            run_one('C · Text + ControlNet', bundle=cn_bundle, cond=conditioning)
        MANAGER.enable_lora(cn_bundle, settings['lora_scale'])
        run_one('D · Text + ControlNet + LoRA', bundle=cn_bundle, cond=conditioning, lora_active=True)
    finally:
        MANAGER.clear()
    return {
        'mode': 'ai_inference',
        'runs': runs,
        'preprocessing_selected': selected,
        'notice': ('A vs B and C vs D isolate LoRA with/without ControlNet. '
                   'B vs D shows the effect of adding ControlNet while LoRA is enabled. '
                   'Pixel differences indicate change, not image quality.'),
    }


def _validate_upload():
    sketch = request.files.get('sketch')
    if sketch is None or not sketch.filename:
        raise ValueError('Sketch is required.')
    if Path(sketch.filename).suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
        raise ValueError('Only PNG, JPG and WEBP are supported.')
    raw = sketch.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError('Sketch exceeds 10 MB.')
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.width * opened.height > MAX_SOURCE_PIXELS:
                raise ValueError('Sketch dimensions exceed 30 megapixels.')
            image = ImageOps.exif_transpose(opened).convert('RGB')
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError('Image file cannot be decoded.') from exc
    image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)

    approach = request.form.get('approach', 'a1')
    if approach not in MODEL_LABELS:
        raise ValueError('Invalid approach.')
    preprocessing = request.form.get('preprocessing', 'auto')
    if preprocessing not in {'auto', 'canny', 'scribble', 'lineart'}:
        raise ValueError('Invalid preprocessing.')
    prompt = request.form.get('prompt', '').strip()
    negative = request.form.get('negative_prompt', '').strip()
    if len(prompt) > 700 or len(negative) > 500:
        raise ValueError('Prompt too long.')

    def number(name, default, lo, hi, cast=float):
        try:
            n = cast(request.form.get(name, default))
            if not lo <= n <= hi:
                raise ValueError()
            return n
        except (TypeError, ValueError):
            raise ValueError(f'{name} must be between {lo} and {hi}.')

    seed = number('seed', 42, -1, 2147483647, int)
    if seed == -1:
        seed = secrets.randbelow(2147483647)
    settings = {
        'prompt': prompt,
        'negative_prompt': negative or DEFAULT_NEGATIVE,
        'seed': seed,
        'steps': number('steps', 30, 10, 60, int),
        'guidance_scale': number('guidance_scale', 7.5, 1, 15),
        'controlnet_scale': number('controlnet_scale', .9, 0, 2),
        'lora_scale': number('lora_scale', .7, 0, 1.5),
    }
    return image, approach, preprocessing, settings


def create_app(api_key: str):
    if len(api_key) < 24:
        raise ValueError('Kaggle API key must be at least 24 characters.')
    api = Flask(__name__)
    api.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024 + 65536

    @api.before_request
    def check_auth():
        if request.path == '/health':
            return None
        if not hmac.compare_digest(request.headers.get('X-API-Key', ''), api_key):
            return jsonify({'error': 'Unauthorized request.'}), 401

    @api.get('/health')
    def health():
        import torch
        return jsonify({
            'status': 'online', 'mode': 'ai_inference', 'inference_enabled': True,
            'gpu_available': torch.cuda.is_available(),
            'message': 'Kaggle API ready (SD1.5 / LoRA / SDXL / LoRA Lab).',
            'lora_available': bool(LORA_PATH and Path(LORA_PATH).is_file()),
            'preprocessing_supported': {'a1': ['auto', 'canny', 'scribble', 'lineart'],
                                        'a2': ['auto', 'canny', 'scribble', 'lineart'],
                                        'a3': ['auto', 'canny', 'scribble']},
            'active_pipeline': MANAGER.bundle.key if MANAGER.bundle else None,
            'busy': INFERENCE_LOCK.locked(),
        })

    def guarded(work):
        if not INFERENCE_LOCK.acquire(blocking=False):
            return jsonify({'error': 'GPU already processing a request. Retry after the current run finishes.'}), 409
        try:
            return jsonify(work())
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({'error': str(exc)}), 400
        except Exception as exc:
            import traceback
            traceback.print_exc()
            try:
                import torch
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    MANAGER.clear()
                    return jsonify({'error': 'GPU out of memory. Try fewer simultaneous workloads or reduce image size.'}), 507
            except ImportError:
                pass
            return jsonify({'error': f'Generation failed ({type(exc).__name__}): {exc}'}), 500
        finally:
            INFERENCE_LOCK.release()

    @api.post('/generate')
    def generate():
        return guarded(lambda: generate_result(*_generate_arguments()))

    def _generate_arguments():
        image, approach, mode, settings = _validate_upload()
        return image, approach, mode, settings

    @api.post('/compare')
    def compare():
        def work():
            image, _, mode, settings = _validate_upload()
            runs = [generate_result(image, approach, mode, settings, comparing=True)
                    for approach in MODEL_LABELS]
            return {'mode': 'ai_inference', 'runs': runs,
                    'notice': 'SDXL uses its own Auto Canny/Scribble when Lineart is selected for the SD1.5 approaches.'
                    if mode == 'lineart' else 'Each approach records its own executed preprocessing and checkpoints.'}
        return guarded(work)

    @api.post('/ablation')
    def ablation():
        def work():
            image, _, mode, settings = _validate_upload()
            if not LORA_PATH or not Path(LORA_PATH).is_file():
                raise FileNotFoundError('LoRA Lab needs your .safetensors file on Kaggle; configure SKETCH2FUTURE_LORA_PATH.')
            return run_lora_lab(image, mode, settings)
        return guarded(work)

    @api.post('/clear-cache')
    def clear_cache():
        def work():
            MANAGER.clear()
            return {'status': 'cleared'}
        return guarded(work)

    @api.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify({'error': 'Upload too large (10 MB).'}), 413

    return api


__all__ = ['create_app', 'generate_result', 'preprocess', 'run_lora_lab', 'MANAGER']
