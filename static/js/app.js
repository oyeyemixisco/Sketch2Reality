/* Sketch2Future XAI Studio: local UI, real GPU inference, Explain/Compare/LoRA Lab. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const state = { sourceMode: 'upload', selectedFile: null, previewObjectUrl: null,
    drawingDirty: false, erasing: false, drawing: false, preset: 'standard',
    activeRun: null, comparisons: [], labRuns: [], busy: false, toastTimer: null };

  const canvas = $('drawingCanvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: false });
  const MAX_BYTES = 10 * 1024 * 1024;
  const MODELS = {
    a1: { short: 'Approach 1', label: 'SD 1.5 + ControlNet' },
    a2: { short: 'Approach 2', label: 'SD 1.5 + ControlNet + LoRA' },
    a3: { short: 'Approach 3', label: 'SDXL + ControlNet' },
  };

  function showToast(message) {
    const node = $('toast');
    node.textContent = message;
    node.classList.add('show');
    window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(() => node.classList.remove('show'), 3900);
  }
  function show(node, visible = true) { node.classList.toggle('hidden', !visible); }
  function setBusy(busy) {
    state.busy = busy;
    $('generateButton').disabled = busy;
    $('compareButton').disabled = busy;
    $('rerunCompare').disabled = busy;
    $('labButton').disabled = busy;
    $('labRerun').disabled = busy;
    $('generateButton').querySelector('span:nth-child(2)').textContent = busy ? 'Sending request...' : 'Generate image';
  }

  async function checkHealth() {
    const chip = $('backendStatus');
    chip.classList.remove('online', 'offline');
    try {
      const res = await fetch('/api/health', { cache: 'no-store' });
      if (!res.ok) throw Error('Local Flask is not responding');
      const health = await res.json();
      const remote = health.mode !== 'local_demo';
      const connected = Boolean(health.backend_connected);
      chip.classList.add(health.status === 'offline' ? 'offline' : 'online');
      $('backendStatusText').textContent = remote
        ? (connected ? 'Kaggle API connected' : 'Kaggle API offline')
        : 'Local demo mode';
      chip.title = health.message || 'Backend check completed';
      if ($('environmentLabel')) $('environmentLabel').textContent = connected ? 'KAGGLE GPU' : (remote ? 'KAGGLE OFFLINE' : 'LOCAL DEMO');
      if (connected) {
        $('backendBannerText').textContent = 'Connected to the Kaggle backend through ngrok. Sketch uploads and generation settings are sent to Kaggle, where the selected diffusion pipeline generates the result.';
      } else if (remote) {
        $('backendBannerText').textContent = 'Kaggle connection is unavailable. Check the notebook, ngrok URL and local .env settings. Generation will not silently fall back to a local mock.';
      } else {
        $('backendBannerText').textContent = 'Offline local demo mode is active. Change SKETCH2FUTURE_BACKEND to kaggle to test requests through ngrok.';
      }
    } catch (e) {
      chip.classList.add('offline');
      $('backendStatusText').textContent = 'Local API unavailable';
      chip.title = String(e);
    }
  }

  function switchInputMode(mode) {
    state.sourceMode = mode;
    show($('uploadView'), mode === 'upload');
    show($('drawView'), mode === 'draw');
    for (const [buttonId, ownMode] of [['uploadModeButton', 'upload'], ['drawModeButton', 'draw']]) {
      $(buttonId).classList.toggle('active', mode === ownMode);
      $(buttonId).setAttribute('aria-pressed', String(mode === ownMode));
    }
  }

  function acceptFile(file) {
    if (!file) return;
    const ok = ['image/png', 'image/jpeg', 'image/webp'].includes(file.type) || /\.(png|jpe?g|webp)$/i.test(file.name);
    if (!ok) { showToast('Please choose a PNG, JPG or WEBP image.'); return; }
    if (file.size > MAX_BYTES) { showToast('Image is too large. Maximum size is 10 MB.'); return; }
    state.selectedFile = file;
    if (state.previewObjectUrl) URL.revokeObjectURL(state.previewObjectUrl);
    state.previewObjectUrl = URL.createObjectURL(file);
    $('sketchPreviewImage').src = state.previewObjectUrl;
    $('selectedFileName').textContent = file.name;
    show($('uploadEmpty'), false);
    show($('uploadPreview'), true);
    switchInputMode('upload');
    showToast('Sketch ready for generation.');
  }

  function initUpload() {
    $('uploadModeButton').addEventListener('click', () => switchInputMode('upload'));
    $('drawModeButton').addEventListener('click', () => switchInputMode('draw'));
    $('dropZone').addEventListener('click', () => $('sketchFile').click());
    $('dropZone').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('sketchFile').click(); }
    });
    $('sketchFile').addEventListener('change', (event) => acceptFile(event.target.files?.[0]));
    for (const eventName of ['dragenter', 'dragover']) {
      $('dropZone').addEventListener(eventName, (event) => { event.preventDefault(); $('dropZone').classList.add('drag-over'); });
    }
    for (const eventName of ['dragleave', 'drop']) {
      $('dropZone').addEventListener(eventName, (event) => { event.preventDefault(); $('dropZone').classList.remove('drag-over'); });
    }
    $('dropZone').addEventListener('drop', (event) => acceptFile(event.dataTransfer?.files?.[0]));
    $('sampleButton').addEventListener('click', async () => {
      try {
        const response = await fetch('/static/img/sample_sketch.png');
        if (!response.ok) throw Error('Sample missing');
        acceptFile(new File([await response.blob()], 'sample_cottage_sketch.png', { type: 'image/png' }));
        $('promptInput').value = 'A cozy cottage in the mountains beside a winding path, realistic materials, golden-hour light';
        updatePromptCount();
      } catch (_e) { showToast('Could not load the example sketch.'); }
    });
  }

  function resetDrawing() {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    state.drawingDirty = false;
  }
  function pointFromPointer(event) {
    const rect = canvas.getBoundingClientRect();
    return { x: (event.clientX - rect.left) * canvas.width / rect.width,
      y: (event.clientY - rect.top) * canvas.height / rect.height };
  }
  function drawStart(event) {
    event.preventDefault();
    state.drawing = true;
    state.drawingDirty = true;
    canvas.setPointerCapture(event.pointerId);
    const point = pointFromPointer(event);
    ctx.beginPath(); ctx.moveTo(point.x, point.y);
    ctx.strokeStyle = state.erasing ? '#ffffff' : '#27354e';
    ctx.lineWidth = Number($('brushSize').value) * (state.erasing ? 2.8 : 1);
    ctx.lineTo(point.x + 0.01, point.y + 0.01); ctx.stroke();
  }
  function drawMove(event) {
    if (!state.drawing) return;
    event.preventDefault();
    const point = pointFromPointer(event);
    ctx.lineTo(point.x, point.y); ctx.stroke();
  }
  function drawEnd() { state.drawing = false; }
  function initCanvas() {
    resetDrawing();
    canvas.addEventListener('pointerdown', drawStart);
    canvas.addEventListener('pointermove', drawMove);
    canvas.addEventListener('pointerup', drawEnd);
    canvas.addEventListener('pointercancel', drawEnd);
    $('brushButton').addEventListener('click', () => setErase(false));
    $('eraserButton').addEventListener('click', () => setErase(true));
    $('clearCanvasButton').addEventListener('click', () => { resetDrawing(); showToast('Canvas cleared.'); });
  }
  function setErase(erase) {
    state.erasing = erase;
    $('brushButton').classList.toggle('active', !erase);
    $('eraserButton').classList.toggle('active', erase);
  }

  function updatePromptCount() { $('promptCount').textContent = `${$('promptInput').value.length} / 700`; }
  function activatePreset(preset) {
    state.preset = preset;
    for (const el of document.querySelectorAll('.preset-card')) {
      const active = el.dataset.preset === preset;
      el.classList.toggle('active', active);
      el.setAttribute('aria-pressed', String(active));
    }
    if (preset === 'standard') { $('approachSelect').value = 'a1'; $('stepsInput').value = '28'; }
    if (preset === 'detailed') { $('approachSelect').value = 'a3'; $('stepsInput').value = '35'; }
    toggleLoRA();
  }
  function toggleLoRA() {
    const isA2 = $('approachSelect').value === 'a2';
    const isA3 = $('approachSelect').value === 'a3';
    show($('loraField'), isA2);
    const lineartOption = $('preprocessingSelect').querySelector('option[value="lineart"]');
    lineartOption.disabled = isA3;
    if (isA3 && $('preprocessingSelect').value === 'lineart') {
      $('preprocessingSelect').value = 'auto';
      showToast('Lineart is available for SD 1.5 only. SDXL uses Auto, Canny or Scribble.');
    }
    $('preprocessingHint').textContent = isA3
      ? 'SDXL uses the verified Canny / Scribble selection from the standalone notebook.'
      : 'Auto chooses Canny, Scribble or Lineart from measured sketch features.';
  }
  function initControls() {
    $('promptInput').addEventListener('input', updatePromptCount);
    document.querySelectorAll('.suggestion-chip').forEach((chip) => chip.addEventListener('click', () => {
      const value = $('promptInput').value.trim();
      $('promptInput').value = value ? `${value.replace(/[,\s]+$/, '')}, ${chip.dataset.style}` : chip.dataset.style;
      updatePromptCount();
    }));
    document.querySelectorAll('.preset-card').forEach((card) => card.addEventListener('click', () => activatePreset(card.dataset.preset)));
    $('approachSelect').addEventListener('change', () => { toggleLoRA(); activatePreset('custom'); });
    $('stepsInput').addEventListener('change', () => activatePreset('custom'));
    $('preprocessingSelect').addEventListener('change', () => activatePreset('custom'));
    for (const [rangeId, outputId, decimals] of [['cfgInput', 'cfgOutput', 1], ['controlInput', 'controlOutput', 2], ['loraInput', 'loraOutput', 2]]) {
      $(rangeId).addEventListener('input', () => { $(outputId).textContent = Number($(rangeId).value).toFixed(decimals); });
    }
    $('dismissBanner').addEventListener('click', () => show(document.querySelector('.demo-banner'), false));
  }

  function switchTab(name) {
    for (const tab of document.querySelectorAll('.result-tab')) {
      const active = tab.dataset.tab === name;
      tab.classList.toggle('active', active);
      tab.setAttribute('aria-selected', String(active));
    }
    for (const id of ['result', 'explain', 'compare', 'lab']) show($(`panel-${id}`), id === name);
  }
  function initTabs() {
    document.querySelectorAll('.result-tab').forEach((tab) => tab.addEventListener('click', () => switchTab(tab.dataset.tab)));
    document.querySelectorAll('.go-create').forEach((button) => button.addEventListener('click', () => { switchTab('result'); $('sketchFile').focus(); }));
    $('viewExplanationButton').addEventListener('click', () => switchTab('explain'));
  }

  async function buildFormData() {
    const data = new FormData();
    if (state.sourceMode === 'draw') {
      if (!state.drawingDirty) throw Error('Draw something on the canvas first.');
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
      if (!blob) throw Error('Could not export the canvas.');
      data.append('sketch', blob, 'canvas_sketch.png');
    } else {
      if (!state.selectedFile) throw Error('Upload a sketch or try the example sketch first.');
      data.append('sketch', state.selectedFile, state.selectedFile.name);
    }
    const fields = {
      prompt: $('promptInput').value,
      negative_prompt: $('negativePrompt').value,
      preset: state.preset,
      approach: $('approachSelect').value,
      preprocessing: $('preprocessingSelect').value,
      seed: $('seedInput').value,
      steps: $('stepsInput').value,
      guidance_scale: $('cfgInput').value,
      controlnet_scale: $('controlInput').value,
      lora_scale: $('loraInput').value,
    };
    for (const [key, value] of Object.entries(fields)) data.append(key, value);
    return data;
  }
  async function apiPost(path, payload) {
    const response = await fetch(path, { method: 'POST', body: payload });
    const json = await response.json().catch(() => ({}));
    if (!response.ok) throw Error(json.error || `Request failed: ${response.status}`);
    return json;
  }
  async function generate(event) {
    if (event) event.preventDefault();
    if (state.busy) return;
    let payload;
    try { payload = await buildFormData(); } catch (error) { showToast(error.message); return; }
    const selectedApproach = $('approachSelect').value;
    const modeText = $('preprocessingSelect').value === 'auto' ? 'Auto preprocessing' : $('preprocessingSelect').value.toUpperCase();
    $('generationLoadingMessage').textContent = ({
      a1: 'Running Stable Diffusion 1.5 + ControlNet on Kaggle.',
      a2: 'Running Stable Diffusion 1.5 + ControlNet + your LoRA on Kaggle.',
      a3: 'Running SDXL + ControlNet on Kaggle. Larger images can take longer.',
    })[selectedApproach] + ` ${modeText}.`;
    setBusy(true);
    show($('emptyState'), false);
    show($('generatedState'), false);
    show($('loadingState'), true);
    switchTab('result');
    try {
      const run = await apiPost('/api/generate', payload);
      displayRun(run);
      showToast(run.mode === 'kaggle_api_demo' ? 'Image received from Kaggle.' : (run.mode === 'mock' ? 'Local mock result ready; no AI model was used.' : 'Image received.'));
    } catch (error) {
      show($('loadingState'), false);
      show($('generatedState'), Boolean(state.activeRun));
      show($('emptyState'), !state.activeRun);
      showToast(error.message || 'Could not complete the generation request.');
    } finally { setBusy(false); }
  }

  function addParameter(parent, label, value) {
    const el = document.createElement('div'); el.className = 'parameter-item';
    const small = document.createElement('small'); small.textContent = label;
    const strong = document.createElement('strong'); strong.textContent = String(value);
    el.append(small, strong); parent.appendChild(el);
  }
  function displayRun(run) {
    state.activeRun = run;
    const kaggleDemo = run.mode === 'kaggle_api_demo';
    const isActualAI = run.mode === 'ai_inference';
    show($('emptyState'), false);
    show($('loadingState'), false);
    show($('generatedState'), true);
    $('generatedImage').src = run.output_url;
    $('downloadImageButton').href = run.output_url;
    $('downloadImageButton').download = `sketch2future_${run.run_id}_${isActualAI ? 'result' : 'demo'}.png`;
    $('generatedTitle').textContent = run.lab_label || MODELS[run.approach]?.label || run.model_label;
    const m = run.metadata || {};
    const elapsed = Number(m.generation_ms ?? m.mock_processing_ms ?? 0);
    const origin = kaggleDemo ? 'Kaggle API' : (isActualAI ? 'GPU inference' : 'Local demo');
    $('generatedSubtitle').textContent = `Saved locally · ${origin} · ${(elapsed/1000).toFixed(1)} s · ${(m.preprocessing_selected || 'none').toUpperCase()} guidance`;
    $('resultModePill').textContent = isActualAI ? 'AI GENERATED' : (kaggleDemo ? 'KAGGLE RESULT' : 'MOCK RESULT');
    $('resultModePill').classList.add('ready');
    $('mainResultWatermark').textContent = isActualAI ? 'AI GENERATED' : (kaggleDemo ? 'KAGGLE RESULT' : 'LOCAL MOCK');
    $('resultInfoText').textContent = isActualAI
      ? 'This image was generated by the selected approach. Review preprocessing and parameters in Explain.'
      : (kaggleDemo
         ? 'This result came from Kaggle through ngrok.'
         : 'This is an illustrative local demo. No generative AI model was executed.');

    show($('explainEmpty'), false);
    show($('explainContent'), true);
    $('explainSourceImage').src = run.source_url;
    $('explainConditionImage').src = run.conditioning_url;
    $('explainOutputImage').src = run.output_url;
    $('explainSummary').textContent = run.explanation.summary;
    $('explainPreprocessing').textContent = run.explanation.preprocessing_reason;
    $('explainPrompt').textContent = run.explanation.prompt_role;
    $('explainModel').textContent = run.explanation.model_role;
    $('explainLimitations').textContent = `Important limitation: ${run.explanation.limitation}`;
    $('exportMetadata').href = `/api/runs/${encodeURIComponent(run.run_id)}/metadata`;
    const grid = $('parameterGrid'); grid.replaceChildren();
    addParameter(grid, 'Model selected', run.model_label);
    addParameter(grid, 'Model executed', m.model_executed || 'Not reported');
    addParameter(grid, 'Structural guidance', m.preprocessing_preview || 'Not reported');
    addParameter(grid, 'Preprocessing requested', m.preprocessing_requested ?? 'Not reported');
    addParameter(grid, 'Preprocessing actually used', m.preprocessing_selected ?? 'Not reported');
    addParameter(grid, 'ControlNet checkpoint', m.controlnet_model || 'None (text-only)');
    addParameter(grid, 'Auto-selection fallback', String(Boolean(m.preprocessing_fallback)));
    if (m.preprocessor_model) addParameter(grid, 'Preprocessor checkpoint', m.preprocessor_model);
    if (m.scheduler) addParameter(grid, 'Scheduler', m.scheduler);
    if (m.lora_checkpoint_name) addParameter(grid, 'LoRA checkpoint', m.lora_checkpoint_name);
    addParameter(grid, 'LoRA actually loaded', String(Boolean(m.lora_loaded)));
    addParameter(grid, 'Input resolution', `${m.input_width ?? '?'} × ${m.input_height ?? '?'}`);
    addParameter(grid, 'Seed', m.settings?.seed ?? 'Not reported');
    addParameter(grid, 'Denoising steps (requested)', m.settings?.steps ?? 'Not reported');
    addParameter(grid, 'CFG (requested)', m.settings?.guidance_scale ?? 'Not reported');
    addParameter(grid, 'ControlNet strength (requested)', m.settings?.controlnet_scale ?? 'Not reported');
    if (run.approach === 'a2') addParameter(grid, 'LoRA scale effective', m.lora_scale_effective ?? 'Not reported');
    addParameter(grid, 'ControlNet strength effective', m.effective_controlnet_scale ?? 'None');
    addParameter(grid, 'Negative prompt effective', m.settings?.negative_prompt ?? 'Not reported');
    addParameter(grid, 'Output resolution', `${m.output_width ?? '?'} × ${m.output_height ?? '?'}`);
    addParameter(grid, isActualAI ? 'Generation time' : 'Preview processing time', `${elapsed} ms`);
    addParameter(grid, 'Execution mode', m.execution_mode || run.mode);
    addParameter(grid, 'Effective prompt', m.settings?.prompt || '(default)');
    const metrics = $('xaiMetrics'); metrics.replaceChildren();
    const entries = Object.entries(m.preprocessing_auto_metrics || {});
    if (!entries.length) {
      const p = document.createElement('p'); p.textContent = 'No measured preprocessing metrics were reported.'; metrics.appendChild(p);
    } else {
      for (const [key, val] of entries) {
        const box = document.createElement('div'); box.className = 'xai-metric';
        const label = document.createElement('small'); label.textContent = key.replaceAll('_', ' ');
        const value = document.createElement('strong'); value.textContent = Array.isArray(val) ? val.join(' × ') : String(val);
        box.append(label, value); metrics.appendChild(box);
      }
    }
  }

  function makeComparisonCard(run) {
    const div = document.createElement('div'); div.className = 'comparison-card';
    const img = document.createElement('img'); img.src = run.output_url; img.alt = `Preview for ${run.model_label}`; img.loading = 'lazy';
    const heading = document.createElement('h4'); heading.textContent = MODELS[run.approach]?.short || run.approach;
    const sub = document.createElement('p'); sub.textContent = MODELS[run.approach]?.label || run.model_label;
    const button = document.createElement('button'); button.type = 'button'; button.textContent = 'View result ↗';
    button.addEventListener('click', () => { displayRun(run); switchTab('result'); });
    const extra = document.createElement('p');
    extra.textContent = `Guidance: ${run.metadata?.preprocessing_selected || 'not recorded'}${run.metadata?.preprocessing_fallback ? ' (fallback)' : ''}`;
    div.append(img, heading, sub, extra, button);
    return div;
  }
  async function compare() {
    if (state.busy) return;
    let payload;
    try { payload = await buildFormData(); } catch (error) { showToast(error.message); return; }
    setBusy(true);
    show($('compareEmpty'), false); show($('compareResults'), false); show($('compareLoading'), true);
    try {
      const json = await apiPost('/api/compare', payload);
      state.comparisons = json.runs;
      $('comparisonGrid').replaceChildren(...json.runs.map(makeComparisonCard));
      $('comparisonNotice').textContent = json.notice || 'Review the conditioning method recorded for each approach.';
      show($('compareLoading'), false); show($('compareResults'), true);
      showToast(json.mode === 'kaggle_api_demo' ? 'Three Kaggle generation results received.' : 'Three generation results received.');
    } catch (error) {
      show($('compareLoading'), false);
      show($('compareResults'), state.comparisons.length > 0);
      show($('compareEmpty'), state.comparisons.length === 0);
      showToast(error.message || 'Could not complete comparison.');
    } finally { setBusy(false); }
  }

  function makeLabCard(run) {
    const item = document.createElement('div'); item.className = 'lab-card';
    const img = document.createElement('img'); img.src = run.output_url; img.loading = 'lazy'; img.alt = run.lab_label;
    const heading = document.createElement('h4'); heading.textContent = run.lab_label;
    const sub = document.createElement('p');
    const m = run.metadata || {};
    sub.textContent = `LoRA: ${m.lora_loaded ? 'ON' : 'OFF'} · ControlNet: ${m.controlnet_model ? m.preprocessing_selected : 'OFF'} · ${m.generation_ms ?? 0} ms`;
    const button = document.createElement('button'); button.type = 'button'; button.textContent = 'Inspect result and XAI ↗';
    button.addEventListener('click', () => { displayRun(run); switchTab('explain'); });
    item.append(img, heading, sub, button);
    return item;
  }
  async function runLab() {
    if (state.busy) return;
    let payload;
    try { payload = await buildFormData(); } catch (error) { showToast(error.message); return; }
    setBusy(true);
    show($('labEmpty'), false); show($('labResults'), false); show($('labLoading'), true);
    try {
      const response = await apiPost('/api/ablation', payload);
      if (!Array.isArray(response.runs) || response.runs.length !== 4) throw Error('Incomplete LoRA experiment result.');
      state.labRuns = response.runs;
      $('labGrid').replaceChildren(...response.runs.map(makeLabCard));
      const metrics = $('labDifferences'); metrics.replaceChildren();
      const titles = {A_vs_B:'A vs B · LoRA without ControlNet', C_vs_D:'C vs D · LoRA with ControlNet', B_vs_D:'B vs D · ControlNet added'};
      for (const [key, label] of Object.entries(titles)) {
        const panel = document.createElement('div'); panel.className = 'lab-difference';
        const small = document.createElement('small'); small.textContent = label;
        const number = document.createElement('strong');
        number.textContent = response.pixel_differences?.[key] == null ? 'n/a' : Number(response.pixel_differences[key]).toFixed(3);
        panel.append(small, number); metrics.appendChild(panel);
      }
      $('labNotice').textContent = (response.notice || '') + ' Mean absolute pixel differences are in 0–255 RGB units; they do not measure quality.';
      show($('labLoading'), false); show($('labResults'), true);
      showToast('Four real LoRA Lab results received from Kaggle.');
    } catch (error) {
      show($('labLoading'), false);
      show($('labResults'), state.labRuns.length > 0);
      show($('labEmpty'), state.labRuns.length === 0);
      showToast(error.message || 'LoRA Lab request failed.');
    } finally { setBusy(false); }
  }

  function closeDrawer() {
    $('historyDrawer').classList.remove('open');
    $('historyDrawer').setAttribute('aria-hidden', 'true');
    show($('drawerBackdrop'), false);
  }
  async function openDrawer() {
    show($('drawerBackdrop'), true);
    $('historyDrawer').classList.add('open');
    $('historyDrawer').setAttribute('aria-hidden', 'false');
    await refreshHistory();
  }
  async function refreshHistory() {
    const list = $('historyList'); list.replaceChildren();
    try {
      const response = await fetch('/api/history');
      if (!response.ok) throw Error('Could not load history.');
      const data = await response.json();
      if (!data.runs.length) {
        const empty = document.createElement('div'); empty.className = 'history-placeholder';
        empty.textContent = 'Your local history is empty.\nCreate a result to see it here.';
        list.append(empty); return;
      }
      for (const run of data.runs) {
        const button = document.createElement('button'); button.type = 'button'; button.className = 'history-item';
        const image = document.createElement('img'); image.src = run.output_url; image.alt = 'Saved result'; image.loading = 'lazy';
        const details = document.createElement('div');
        const heading = document.createElement('strong'); heading.textContent = MODELS[run.approach]?.label || 'Generated image';
        const timestamp = document.createElement('span'); timestamp.textContent = new Date(run.created_at).toLocaleString();
        details.append(heading, timestamp); button.append(image, details);
        button.addEventListener('click', () => { displayRun(run); switchTab('result'); closeDrawer(); });
        list.append(button);
      }
    } catch (error) {
      const message = document.createElement('div'); message.className = 'history-placeholder'; message.textContent = error.message;
      list.append(message);
    }
  }
  function initHistory() {
    $('historyButton').addEventListener('click', openDrawer);
    $('closeHistory').addEventListener('click', closeDrawer);
    $('drawerBackdrop').addEventListener('click', closeDrawer);
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeDrawer(); });
    $('clearHistory').addEventListener('click', async () => {
      if (!window.confirm('Remove all saved local images and run records?')) return;
      try {
        const response = await fetch('/api/history', { method: 'DELETE' });
        if (!response.ok) throw Error('Could not clear history');
        state.activeRun = null; state.comparisons = []; state.labRuns = [];
        show($('emptyState'), true); show($('generatedState'), false); show($('loadingState'), false);
        show($('explainEmpty'), true); show($('explainContent'), false);
        show($('compareEmpty'), true); show($('compareResults'), false);
        show($('labEmpty'), true); show($('labResults'), false);
        $('resultModePill').classList.remove('ready'); $('resultModePill').textContent = 'NO IMAGE YET';
        await refreshHistory(); showToast('Local history cleared.');
      } catch (error) { showToast(error.message); }
    });
  }

  function init() {
    initUpload(); initCanvas(); initControls(); initTabs(); initHistory(); checkHealth();
    $('generateForm').addEventListener('submit', generate);
    $('regenerateButton').addEventListener('click', generate);
    $('compareButton').addEventListener('click', compare);
    $('rerunCompare').addEventListener('click', compare);
    $('labButton').addEventListener('click', runLab);
    $('labRerun').addEventListener('click', runLab);
    toggleLoRA(); updatePromptCount();
  }
  document.addEventListener('DOMContentLoaded', init);
})();
