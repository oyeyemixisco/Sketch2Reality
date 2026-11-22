# Sketch2Future — Explainable AI Studio

**Runnable local Flask interface + Kaggle GPU inference + ngrok.** Verified Image generation notebook-model settings, transparent execution metadata and an optional four-run LoRA experiment. GPU and model downloads are required for real generation.

## Architecture

```text
Browser (upload or draw sketch)
     |  local HTTP (no secret in browser)
Sketch2Future_Local_Flask/app.py
     |  HTTPS + NGROK, Kaggle, HuggingFace API-Keys
Kaggle notebook → ngrok → kaggle_api_server.py
     ├─ Approach 1: SD 1.5 + Canny/Scribble/Lineart ControlNet
     ├─ Approach 2: SD 1.5 + Canny/Scribble/Lineart ControlNet + trained LoRA
     ├─ Approach 3: SDXL + Canny/Scribble ControlNet
     └─ LoRA Lab: text-only, text+LoRA, text+CN, text+CN+LoRA
     |  base64 PNG + effective settings + explanation
Local Flask saves PNGs / metadata → browser Result / Explain / Compare / LoRA Lab
```

## Start Kaggle

1. Open `Sketch2Future_Kaggle_GPU_API_ngrok.ipynb` as a **new Kaggle Notebook**.
2. Enable **GPU** and **Internet** in notebook settings. The first run downloads base models, ControlNets, VAE and (when requested) the Lineart annotator.
3. In Kaggle **Add-ons → Secrets**, configure a secret called `NGROK_TOKEN` containing your personal ngrok authtoken. For optional Hugging Face credentials, use `HF_Token`.
4. In the notebook configuration cell, edit `LORA_WEIGHTS_PATH` to your actual `.safetensors` under `/kaggle/input/...`. The default placeholder will make Approach 2 and LoRA Lab unavailable until changed; Approaches 1 and 3 work without the LoRA.
5. Execute cells in sequence. If packages have been upgraded and Kaggle advises it, **restart the kernel and continue from the configuration cell**.
6. Generate a 24 radmon varchar on Kaggle to uses as  **`KAGGLE_API_KEY`** (secret).  **`KAGGLE_API_URL`** (HTTPS ngrok address). Leave the Kaggle kernel running during use.
7. The `/health` cell confirms the tunnel and reports `gpu_available`, `lora_available`, and `busy`. This is a **connectivity check, not a completed image generation**.

## 2 — Start the local Flask web app

```bash
# activate environment in terminal
python -m venv .venv
# install requirements
pip install -r requirements.txt

```

Edit `.env`:

```ini
SKETCH2FUTURE_BACKEND=kaggle
KAGGLE_API_URL=https://YOUR-NGROK-TUNNEL.ngrok-free.app
KAGGLE_API_KEY=PASTE-KEY-PRINTED-BY-KAGGLE-NOTEBOOK
KAGGLE_READ_TIMEOUT=1800
```

Start:

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser. You should see **Kaggle API connected**. If you see **offline**, check the notebook session, tunnel URL and API key. The browser never directly talks to Kaggle.

A developer-only preview mode is available by setting `SKETCH2FUTURE_BACKEND=mock` in `.env` and restarting local Flask. Mock is only a visual placeholder and does **not** run diffusion or LoRA; LoRA Lab explicitly rejects mock mode.

## Use the interface

- **Create:** Upload or draw a sketch; add prompt/optional negative prompt. Select an approach and preprocessing mode under **Advanced settings**. Auto chooses Canny/Scribble/Lineart for A1/A2; Auto chooses Canny/Scribble for A3. Lineart is disabled for A3.
- **Result:** Save your generated image. The loading message names the actual selected approach rather than always referring to SDXL.
- **Explain:** Inspect original sketch, conditioning map and output. Review selection reason, measured stroke statistics, model and ControlNet checkpoint IDs, LoRA active status and weight, full effective prompt settings, scheduler, seed, and execution time. Download JSON for reproducibility.
- **Compare:** Three actual runs sequentially on Kaggle using the same sketch, prompt, seed and user settings. If manual Lineart is selected, A3 uses its own Auto Canny/Scribble selection and the fallback is clearly labelled. Different model output sizes and preprocessing methods are recorded, not implied to be identical.
- **LoRA Lab:** Four controlled SD 1.5 generations. **A→B** isolates LoRA without ControlNet, **C→D** isolates LoRA with ControlNet, **B→D** adds ControlNet while LoRA is enabled. Mean absolute RGB pixel differences are computed locally (scale 0–255); they are **difference measurements, not quality scores**.
- **History:** Local single-user generation history. Images and metadata are saved in `Sketch2Future_Local_Flask/instance/`, which is Git-ignored.

## Actual models and preprocessing

| Approach | Base model | ControlNet | Auto method |
|---|---|---|---|
| 1 | `stable-diffusion-v1-5/stable-diffusion-v1-5` | `lllyasviel/control_v11p_sd15_*` | midtone/stroke heuristic: Canny/Scribble/Lineart |
| 2 | Same as 1, plus user `.safetensors` LoRA | Same as 1 | Same as 1 |
| 3 | `stabilityai/stable-diffusion-xl-base-1.0` | `diffusers/controlnet-canny-sdxl-1.0` or `xinsir/controlnet-scribble-sdxl-1.0` | Standalone SDXL Canny/Otsu heuristic |

Approach 3 preserves **the verified standalone SDXL checkpoint loading**: standard SDXL ControlNet checkpoint files, the `madebyollin/sdxl-vae-fp16-fix` VAE, **UniPC scheduler**, a maximum long side of 1024 with aspect-ratio-preserving scaling, Canny thresholds 65/150 with Gaussian smoothing, and the original ControlNet strength **without a hidden 0.65 cap**. No `variant='fp16'` is passed to the `xinsir` ControlNet.

Approaches 1 and 2 use the lineart detector `lllyasviel/Annotators` when required. ControlNet and learned-line-extraction checkpoints are paired by method; Lineart is not routed into SDXL.

## Runtime considerations

- GPU inference runs **sequentially**. The Kaggle API returns HTTP 409 if another request is already running, rather than overlapping pipelines and exhausting VRAM.
- The backend caches **one model pipeline at a time** and frees the previous pipeline when switching approaches. SDXL downloads can consume several gigabytes and initial generation can take considerably longer.
- Set `KAGGLE_READ_TIMEOUT=600` for comparisons and four-run ablations. Long runs require the Kaggle session and ngrok tunnel to remain active. A Kaggle disconnect invalidates the tunnel URL.
- If a LoRA checkpoint is incompatible, verify that it was trained for **Stable Diffusion 1.5** and preferably the same base checkpoint; no weights are retrained or silently substituted.
- If `torchao` causes the optional import issue in Kaggle, the notebook installation cell uninstalls it. Restart the kernel if import errors remain after dependency changes.
- If SDXL says `diffusion_pytorch_model.fp16.safetensors` is missing, ensure you are running **this notebook** (not an older Phase 3 backend) and reran its server start cell.
- For reproducibility keep the supplied seed, prompt, output dimensions and preprocessing mode fixed when testing two strategies. Auto is **heuristic**, not proven optimal.
