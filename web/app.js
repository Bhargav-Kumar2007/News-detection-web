/**
 * NewsLens — Neural Networks & Deep Learning Demonstration
 * Minimalist, distraction-free live research interface.
 * Handles dual-branch execution (RapidOCR + BiLSTM and ResNet-18 CNN)
 * and transparent OCR concatenation verification.
 */

const $ = (id) => document.getElementById(id);

let selectedFile = null;
let currentPreviewUrl = null;
let toastTimer = null;

// ============================================================================
// UTILITIES
// ============================================================================

function showToast(message) {
  const el = $("toast");
  if (!el) return;
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 3500);
}

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

function formatPercent(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// ============================================================================
// SYSTEM READINESS CHECK
// ============================================================================

async function checkSystemStatus() {
  const dot = $("globalStatusDot");
  const txt = $("globalStatusText");
  try {
    const res = await fetch("/api/status");
    if (!res.ok) throw new Error("Status check failed");
    const data = await res.json();
    const models = data.models || {};
    const textOk = models.text?.ready;
    const imgOk = models.image?.ready;
    const ocrOk = models.ocr?.ready;

    if (textOk && imgOk && ocrOk) {
      if (dot) {
        dot.className = "status-indicator-dot ready";
      }
      if (txt) {
        txt.textContent = "All Neural Models Online";
      }
    } else {
      if (dot) {
        dot.className = "status-indicator-dot missing";
      }
      if (txt) {
        const missing = [];
        if (!textOk) missing.push("BiLSTM");
        if (!imgOk) missing.push("CNN");
        if (!ocrOk) missing.push("OCR");
        txt.textContent = `Partial: ${missing.join(", ")} Offline`;
      }
    }
  } catch (err) {
    if (dot) dot.className = "status-indicator-dot missing";
    if (txt) txt.textContent = "Backend Offline";
  }
}

// ============================================================================
// IMAGE UPLOAD & PREVIEW
// ============================================================================

function handleFileSelection(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showToast("Please select a valid image file (PNG, JPG, WEBP).");
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    showToast("File exceeds maximum allowed size of 12 MB.");
    return;
  }

  selectedFile = file;
  if (currentPreviewUrl) {
    URL.revokeObjectURL(currentPreviewUrl);
  }
  currentPreviewUrl = URL.createObjectURL(file);

  const previewImg = $("imagePreviewImg");
  const previewName = $("previewFileName");
  const previewSize = $("previewFileSize");
  const dropPlaceholder = $("dropPlaceholder");
  const dropPreview = $("dropPreview");

  if (previewImg) previewImg.src = currentPreviewUrl;
  if (previewName) previewName.textContent = file.name;
  if (previewSize) previewSize.textContent = formatFileSize(file.size);

  if (dropPlaceholder) dropPlaceholder.classList.add("hidden");
  if (dropPreview) dropPreview.classList.remove("hidden");

  // Also update Inspection Panel preview thumbnail immediately
  updateInspectionThumbnail(currentPreviewUrl, file.name);
}

function clearImageSelection() {
  selectedFile = null;
  if (currentPreviewUrl) {
    URL.revokeObjectURL(currentPreviewUrl);
    currentPreviewUrl = null;
  }
  const fileInput = $("imageFileInput");
  if (fileInput) fileInput.value = "";

  const dropPlaceholder = $("dropPlaceholder");
  const dropPreview = $("dropPreview");
  if (dropPlaceholder) dropPlaceholder.classList.remove("hidden");
  if (dropPreview) dropPreview.classList.add("hidden");

  updateInspectionThumbnail(null, null);
}

function updateInspectionThumbnail(url, name) {
  const container = $("inspectImageContainer");
  if (!container) return;
  if (url) {
    container.innerHTML = `<img src="${url}" alt="${escapeHTML(name || 'Uploaded preview')}" />`;
  } else {
    container.innerHTML = `<span class="placeholder-text">No image uploaded</span>`;
  }
}

// Setup drag and drop
function setupDropZone() {
  const zone = $("dropZone");
  const fileInput = $("imageFileInput");
  const btnRemove = $("btnRemoveImage");

  if (zone && fileInput) {
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    });
    zone.addEventListener("dragleave", () => {
      zone.classList.remove("dragover");
    });
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
      }
    });
  }

  if (btnRemove) {
    btnRemove.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      clearImageSelection();
    });
  }
}

// Setup character count
function setupHeadlineInput() {
  const input = $("headlineInput");
  const count = $("charCount");
  if (!input || !count) return;

  input.addEventListener("input", () => {
    const len = input.value.length;
    count.textContent = `${len}/500 characters`;
    if (len > 500) {
      count.style.color = "var(--red-fake)";
    } else {
      count.style.color = "var(--text-muted)";
    }
  });
}

// ============================================================================
// QUICK SAMPLES GENERATOR
// Renders synthetic high-contrast news imagery on canvas so OCR is 100% testable
// ============================================================================

function generateSampleCanvas(type) {
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 360;
  const ctx = canvas.getContext("2d");

  if (type === "flood") {
    // Gradient sky and deep river water
    const grad = ctx.createLinearGradient(0, 0, 0, 360);
    grad.addColorStop(0, "#1e293b");
    grad.addColorStop(0.5, "#475569");
    grad.addColorStop(0.51, "#0284c7");
    grad.addColorStop(1, "#075985");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 640, 360);

    // Realistic water reflection streaks
    ctx.strokeStyle = "rgba(255,255,255,0.3)";
    ctx.lineWidth = 2;
    for (let y = 200; y < 350; y += 22) {
      ctx.beginPath();
      ctx.moveTo(30, y);
      ctx.bezierCurveTo(200, y - 8, 420, y + 8, 610, y);
      ctx.stroke();
    }

    // High-contrast breaking news banner for DBNet OCR detection
    ctx.fillStyle = "rgba(15, 23, 42, 0.92)";
    ctx.fillRect(20, 235, 600, 100);
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(20, 235, 600, 100);

    ctx.fillStyle = "#ef4444";
    ctx.font = "bold 16px Arial, sans-serif";
    ctx.fillText("AGRA FLOOD ALERT: YAMUNA OVERFLOW", 40, 268);

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 21px Arial, sans-serif";
    ctx.fillText("HISTORIC FLOODS SUBMERGE TAJ MAHAL GARDENS", 40, 305);
  } else if (type === "clickbait") {
    // Dark classified dossier style
    ctx.fillStyle = "#090d16";
    ctx.fillRect(0, 0, 640, 360);

    ctx.strokeStyle = "#dc2626";
    ctx.lineWidth = 3;
    ctx.strokeRect(18, 18, 604, 324);

    ctx.fillStyle = "#ef4444";
    ctx.font = "bold 24px Arial, sans-serif";
    ctx.fillText("TOP SECRET // CLASSIFIED LEAK", 40, 65);

    ctx.fillStyle = "#94a3b8";
    ctx.font = "14px monospace";
    ctx.fillText("SECURITY CLEARANCE LEVEL 5 - ARCHIVE 1971", 40, 100);

    ctx.fillStyle = "rgba(30, 41, 59, 0.95)";
    ctx.fillRect(35, 140, 570, 155);

    ctx.fillStyle = "#facc15";
    ctx.font = "bold 22px Arial, sans-serif";
    ctx.fillText("GOVERNMENT ADMITS SECRET 1971 TREATY", 55, 195);

    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 19px Arial, sans-serif";
    ctx.fillText("EXTRATERRESTRIAL CONTACT DOCUMENTS DECLASSIFIED", 55, 240);
  } else if (type === "transit") {
    // Clean public transit notice layout
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(0, 0, 640, 360);

    ctx.fillStyle = "#0284c7";
    ctx.fillRect(0, 0, 640, 50);

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 18px Arial, sans-serif";
    ctx.fillText("CITY METROPOLITAN TRANSIT AUTHORITY", 30, 32);

    ctx.fillStyle = "#0f172a";
    ctx.fillRect(30, 80, 580, 120);

    ctx.fillStyle = "#38bdf8";
    ctx.fillRect(55, 105, 75, 45);
    ctx.fillRect(155, 105, 75, 45);
    ctx.fillRect(255, 105, 75, 45);

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 20px Arial, sans-serif";
    ctx.fillText("METRO EXPANSION PROJECT: OPENING OCTOBER 2026", 55, 175);

    ctx.fillStyle = "#334155";
    ctx.font = "bold 18px Arial, sans-serif";
    ctx.fillText("OFFICIAL NOTICE: PASSENGER REVENUE SERVICE CONFIRMED", 30, 245);
    ctx.font = "15px Arial, sans-serif";
    ctx.fillText("Fully automated high-speed transit line connecting terminal stations.", 30, 280);
  }

  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      const file = new File([blob], `${type}_sample.jpg`, { type: "image/jpeg" });
      resolve(file);
    }, "image/jpeg", 0.92);
  });
}

function setupQuickSamples() {
  const sampleMap = {
    sampleFlood: {
      headline: "Severe monsoon floods submerge Taj Mahal gardens as Yamuna river breaches historic danger mark",
      type: "flood",
    },
    sampleClickbait: {
      headline: "SHOCKING: Government admits secret alien treaty signed in 1971",
      type: "clickbait",
    },
    sampleTransit: {
      headline: "City transit authority announces new underground metro line opening this October",
      type: "transit",
    },
  };

  Object.entries(sampleMap).forEach(([id, sample]) => {
    const btn = $(id);
    if (!btn) return;
    btn.addEventListener("click", async () => {
      const headlineInput = $("headlineInput");
      if (headlineInput) {
        headlineInput.value = sample.headline;
        const count = $("charCount");
        if (count) count.textContent = `${sample.headline.length}/500 characters`;
      }
      const file = await generateSampleCanvas(sample.type);
      handleFileSelection(file);
      showToast(`Sample "${sample.type.toUpperCase()}" loaded. Click Run Multimodal Analysis.`);
    });
  });
}

// ============================================================================
// LIVE PIPELINE VISUALIZER (Requirement #2)
// ============================================================================

const PIPELINE_NODES = [
  "stepInput",
  "stepOcr",
  "stepConcat",
  "stepTokenize",
  "stepBiLstm",
  "stepImageUpload",
  "stepImagePreprocess",
  "stepCnn",
  "stepFusion",
];

function resetPipelineVisualizer() {
  PIPELINE_NODES.forEach((id) => {
    const node = $(id);
    if (!node) return;
    node.classList.remove("active", "done");
  });

  const statuses = {
    stepInputStatus: "Awaiting",
    stepOcrStatus: "Awaiting",
    stepConcatStatus: "Awaiting",
    stepTokenizeStatus: "Awaiting",
    stepBiLstmStatus: "Awaiting",
    stepImageUploadStatus: "Awaiting",
    stepImagePreprocessStatus: "Awaiting",
    stepCnnStatus: "Awaiting",
    stepFusionStatus: "Awaiting dual outputs",
  };

  Object.entries(statuses).forEach(([id, text]) => {
    const el = $(id);
    if (el) el.textContent = text;
  });
}

function setStepState(nodeId, statusId, state, text) {
  const node = $(nodeId);
  const status = $(statusId);
  if (node) {
    node.classList.remove("active", "done");
    if (state) node.classList.add(state);
  }
  if (status && text) {
    status.textContent = text;
  }
}

function startPipelineVisualizer(hasHeadline, hasImage) {
  resetPipelineVisualizer();

  // 1. User Input Received
  setStepState("stepInput", "stepInputStatus", "done", "Received");

  // 2. Image Stream Steps
  if (hasImage) {
    setStepState("stepImageUpload", "stepImageUploadStatus", "done", "Uploaded");
    setStepState("stepOcr", "stepOcrStatus", "active", "Running DBNet+CRNN...");
    setStepState("stepImagePreprocess", "stepImagePreprocessStatus", "active", "Resizing to 224×224...");
    setStepState("stepCnn", "stepCnnStatus", "active", "ResNet-18 Forward...");
  } else {
    setStepState("stepImageUpload", "stepImageUploadStatus", "", "Skipped (No image)");
    setStepState("stepOcr", "stepOcrStatus", "", "Skipped");
    setStepState("stepImagePreprocess", "stepImagePreprocessStatus", "", "Skipped");
    setStepState("stepCnn", "stepCnnStatus", "", "Skipped");
  }

  // 3. Text Stream Steps
  if (hasHeadline || hasImage) {
    setStepState("stepConcat", "stepConcatStatus", "active", "Concatenating...");
    setStepState("stepTokenize", "stepTokenizeStatus", "active", "Vocabulary lookup...");
    setStepState("stepBiLstm", "stepBiLstmStatus", "active", "BiLSTM Forward (256d)...");
  }

  // 4. Fusion
  setStepState("stepFusion", "stepFusionStatus", "active", "Awaiting outputs...");
}

function completePipelineVisualizer(data) {
  const timing = data.timing || {};
  const inspect = data.inspection || {};
  const models = data.models || {};
  const fusion = data.fusion || {};

  // Text stream updates
  if (models.text?.ready) {
    if (inspect.ocr_lines_count > 0) {
      setStepState("stepOcr", "stepOcrStatus", "done", `${timing.ocr_ms} ms (${inspect.ocr_lines_count} lines)`);
    } else if (data.ocr?.ready) {
      setStepState("stepOcr", "stepOcrStatus", "done", `${timing.ocr_ms} ms (0 lines detected)`);
    }

    setStepState("stepConcat", "stepConcatStatus", "done", "Combined headline + OCR");
    setStepState("stepTokenize", "stepTokenizeStatus", "done", `${inspect.total_tokens_count} tokens mapped`);
    setStepState("stepBiLstm", "stepBiLstmStatus", "done", `${timing.text_ms} ms`);
  }

  // Image stream updates
  if (models.image?.ready) {
    setStepState("stepImageUpload", "stepImageUploadStatus", "done", "Loaded");
    setStepState("stepImagePreprocess", "stepImagePreprocessStatus", "done", "224×224 RGB Normalized");
    setStepState("stepCnn", "stepCnnStatus", "done", `${timing.image_ms} ms`);
  }

  // Fusion update
  if (fusion.ready) {
    setStepState("stepFusion", "stepFusionStatus", "done", `Fused (${timing.total_ms} ms total)`);
  } else {
    setStepState("stepFusion", "stepFusionStatus", "", "Single branch only");
  }
}

// ============================================================================
// RENDER DUAL PREDICTIONS & COMBINED SIGNAL (Requirement #4)
// ============================================================================

function renderPredictions(data) {
  const text = data.models?.text;
  const image = data.models?.image;
  const fusion = data.fusion;
  const timing = data.timing || {};

  // 1. Text Model Card (BiLSTM)
  const textTag = $("textVerdictTag");
  const textConf = $("textConfVal");
  const textBar = $("textProgressBar");
  const textFake = $("textFakeProb");
  const textReal = $("textRealProb");
  const textDuration = $("textDuration");

  if (text?.ready) {
    const isReal = text.label.toLowerCase() === "real";
    if (textTag) {
      textTag.textContent = text.label.toUpperCase();
      textTag.className = `verdict-tag ${isReal ? "real" : "fake"}`;
    }
    if (textConf) textConf.textContent = formatPercent(text.confidence);
    if (textBar) {
      textBar.style.width = `${(text.confidence * 100).toFixed(1)}%`;
      textBar.style.background = isReal ? "var(--green-real)" : "var(--red-fake)";
    }
    if (textFake) textFake.textContent = formatPercent(text.fake_probability);
    if (textReal) textReal.textContent = formatPercent(text.real_probability);
    if (textDuration) textDuration.textContent = `${timing.text_ms || 0.0} ms`;
  } else {
    if (textTag) {
      textTag.textContent = "NOT RUN";
      textTag.className = "verdict-tag";
    }
    if (textConf) textConf.textContent = "—";
    if (textBar) textBar.style.width = "0%";
    if (textFake) textFake.textContent = "—";
    if (textReal) textReal.textContent = "—";
    if (textDuration) textDuration.textContent = "0.0 ms";
  }

  // 2. Image Model Card (ResNet-18 CNN)
  const imgTag = $("imageVerdictTag");
  const imgConf = $("imageConfVal");
  const imgBar = $("imageProgressBar");
  const imgFake = $("imageFakeProb");
  const imgReal = $("imageRealProb");
  const imgDuration = $("imageDuration");

  if (image?.ready) {
    const isReal = image.label.toLowerCase() === "real";
    if (imgTag) {
      imgTag.textContent = image.label.toUpperCase();
      imgTag.className = `verdict-tag ${isReal ? "real" : "fake"}`;
    }
    if (imgConf) imgConf.textContent = formatPercent(image.confidence);
    if (imgBar) {
      imgBar.style.width = `${(image.confidence * 100).toFixed(1)}%`;
      imgBar.style.background = isReal ? "var(--green-real)" : "var(--red-fake)";
    }
    if (imgFake) imgFake.textContent = formatPercent(image.fake_probability);
    if (imgReal) imgReal.textContent = formatPercent(image.real_probability);
    if (imgDuration) imgDuration.textContent = `${timing.image_ms || 0.0} ms`;
  } else {
    if (imgTag) {
      imgTag.textContent = "NOT RUN";
      imgTag.className = "verdict-tag";
    }
    if (imgConf) imgConf.textContent = "—";
    if (imgBar) imgBar.style.width = "0%";
    if (imgFake) imgFake.textContent = "—";
    if (imgReal) imgReal.textContent = "—";
    if (imgDuration) imgDuration.textContent = "0.0 ms";
  }

  // 3. Combined Multimodal Signal Box
  const fusionBox = $("combinedSignalBox");
  const signalVerdict = $("signalVerdict");
  const signalAgreement = $("signalAgreement");
  const fusionFake = $("fusionFakeProb");
  const fusionReal = $("fusionRealProb");

  if (fusionBox) {
    fusionBox.classList.remove("real-signal", "fake-signal", "conflict-signal");
  }

  if (fusion?.ready) {
    if (signalVerdict) signalVerdict.textContent = fusion.signal;
    if (signalAgreement) signalAgreement.textContent = fusion.agreement_status;
    if (fusionFake) fusionFake.textContent = formatPercent(fusion.fake_probability);
    if (fusionReal) fusionReal.textContent = formatPercent(fusion.real_probability);

    if (fusionBox) {
      if (fusion.is_agreement) {
        if (fusion.fake_probability >= 0.5) {
          fusionBox.classList.add("fake-signal");
        } else {
          fusionBox.classList.add("real-signal");
        }
      } else {
        fusionBox.classList.add("conflict-signal");
      }
    }
  } else {
    if (signalVerdict) signalVerdict.textContent = "AWAITING DUAL BRANCHES";
    if (signalAgreement) {
      signalAgreement.textContent = fusion?.agreement_status || "Both branches required for multimodal late fusion.";
    }
    if (fusionFake) fusionFake.textContent = "—";
    if (fusionReal) fusionReal.textContent = "—";
  }

  // 4. Latency breakdown in footer
  const totalDuration = $("totalDuration");
  const ocrDuration = $("ocrDuration");
  const textDurationSub = $("textDurationSub");
  const imageDurationSub = $("imageDurationSub");

  if (totalDuration) totalDuration.textContent = `${timing.total_ms || 0.0} ms`;
  if (ocrDuration) ocrDuration.textContent = `${timing.ocr_ms || 0.0} ms`;
  if (textDurationSub) textDurationSub.textContent = `${timing.text_ms || 0.0} ms`;
  if (imageDurationSub) imageDurationSub.textContent = `${timing.image_ms || 0.0} ms`;

  // Toggle results view
  const resultsEmpty = $("resultsEmpty");
  const resultsContent = $("resultsContent");
  const statusPill = $("resultStatusPill");

  if (resultsEmpty) resultsEmpty.classList.add("hidden");
  if (resultsContent) resultsContent.classList.remove("hidden");
  if (statusPill) {
    statusPill.textContent = "COMPLETE";
    statusPill.className = "status-pill complete";
  }
}

// ============================================================================
// RENDER TRANSPARENT OCR INSPECTION PANEL (Requirement #5)
// ============================================================================

function renderInspectionPanel(data) {
  const inspect = data.inspection || {};

  // 1. Box 1: Original Image Thumbnail is already set by handleFileSelection / updateInspectionThumbnail

  // 2. Box 2: RapidOCR Extracted Text
  const ocrTextEl = $("inspectOcrText");
  const ocrCountEl = $("ocrLinesCount");
  if (ocrCountEl) {
    ocrCountEl.textContent = `${inspect.ocr_lines_count || 0} lines detected`;
  }
  if (ocrTextEl) {
    if (inspect.ocr_extracted_text && inspect.ocr_extracted_text.trim()) {
      ocrTextEl.innerHTML = `<span style="color:#0f172a;font-weight:500;">${escapeHTML(inspect.ocr_extracted_text)}</span>`;
    } else if (selectedFile) {
      ocrTextEl.innerHTML = `<em>No readable text detected in image by DBNet + CRNN.</em>`;
    } else {
      ocrTextEl.innerHTML = `<em>No image provided for optical character recognition.</em>`;
    }
  }

  // 3. Box 3: Final Concatenated Text (Headline + " " + OCR)
  const combinedEl = $("inspectCombinedText");
  if (combinedEl) {
    if (inspect.combined_text && inspect.combined_text.trim()) {
      combinedEl.innerHTML = `<strong>${escapeHTML(inspect.combined_text)}</strong>`;
    } else {
      combinedEl.innerHTML = `<em>No text inputs provided.</em>`;
    }
  }

  // 4. Box 4: BiLSTM Preprocessed Tokens
  const tokensBox = $("inspectTokensList");
  const tokensBadge = $("tokensCountBadge");
  const totalCount = inspect.total_tokens_count || 0;

  if (tokensBadge) {
    tokensBadge.textContent = `${totalCount} tokens`;
  }

  if (tokensBox) {
    if (inspect.tokens_list && inspect.tokens_list.length > 0) {
      const chips = inspect.tokens_list.map((tok) => {
        const isOov = tok === "<unk>";
        const chipClass = isOov ? "token-chip" : "token-chip";
        const extraStyle = isOov ? "background:#fee2e2;color:#991b1b;border-color:#fca5a5;" : "";
        return `<span class="${chipClass}" style="${extraStyle}">${escapeHTML(tok)}</span>`;
      });

      if (totalCount > 50) {
        chips.push(`<span class="token-chip" style="background:#fef3c7;color:#92400e;">+${totalCount - 50} more truncated</span>`);
      }
      tokensBox.innerHTML = chips.join("");
    } else {
      tokensBox.innerHTML = `<em>No tokens generated.</em>`;
    }
  }
}

// ============================================================================
// FORM SUBMIT & MULTIMODAL INFERENCE
// ============================================================================

function setupAnalyzeForm() {
  const form = $("analyzeForm");
  const btnAnalyze = $("btnAnalyze");
  const btnText = $("btnAnalyzeText");
  const btnSpinner = $("btnSpinner");
  const btnClear = $("btnClear");

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();

      const headline = ($("headlineInput")?.value || "").trim();
      if (!headline && !selectedFile) {
        showToast("Please enter a headline, upload an image, or click a Quick Sample.");
        return;
      }

      // UI loading state
      if (btnAnalyze) btnAnalyze.disabled = true;
      if (btnText) btnText.textContent = "Executing Inference...";
      if (btnSpinner) btnSpinner.classList.remove("hidden");

      const statusPill = $("resultStatusPill");
      if (statusPill) {
        statusPill.textContent = "INFERENCE ACTIVE";
        statusPill.className = "status-pill running";
      }

      startPipelineVisualizer(Boolean(headline), Boolean(selectedFile));

      try {
        const formData = new FormData();
        formData.append("headline", headline);
        if (selectedFile) {
          formData.append("image_file", selectedFile);
        }

        const res = await fetch("/api/analyze", {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Server returned HTTP ${res.status}`);
        }

        const data = await res.json();

        // Render everything
        completePipelineVisualizer(data);
        renderPredictions(data);
        renderInspectionPanel(data);

        showToast("Analysis complete: Dual predictions & OCR inspection updated.");
      } catch (err) {
        showToast(`Inference Error: ${err.message}`);
        if (statusPill) {
          statusPill.textContent = "ERROR";
          statusPill.className = "status-pill";
        }
      } finally {
        if (btnAnalyze) btnAnalyze.disabled = false;
        if (btnText) btnText.textContent = "Run Multimodal Analysis";
        if (btnSpinner) btnSpinner.classList.add("hidden");
      }
    });
  }

  if (btnClear) {
    btnClear.addEventListener("click", () => {
      const headlineInput = $("headlineInput");
      if (headlineInput) headlineInput.value = "";
      const count = $("charCount");
      if (count) count.textContent = "0/500 characters";

      clearImageSelection();
      resetPipelineVisualizer();

      const resultsEmpty = $("resultsEmpty");
      const resultsContent = $("resultsContent");
      const statusPill = $("resultStatusPill");

      if (resultsEmpty) resultsEmpty.classList.remove("hidden");
      if (resultsContent) resultsContent.classList.add("hidden");
      if (statusPill) {
        statusPill.textContent = "AWAITING INPUT";
        statusPill.className = "status-pill";
      }

      const ocrTextEl = $("inspectOcrText");
      const ocrCountEl = $("ocrLinesCount");
      const combinedEl = $("inspectCombinedText");
      const tokensBox = $("inspectTokensList");
      const tokensBadge = $("tokensCountBadge");

      if (ocrCountEl) ocrCountEl.textContent = "0 lines";
      if (ocrTextEl) ocrTextEl.innerHTML = `<em>RapidOCR recognition output will appear here.</em>`;
      if (combinedEl) combinedEl.innerHTML = `<em>Exact concatenated string will appear here.</em>`;
      if (tokensBadge) tokensBadge.textContent = "0 tokens";
      if (tokensBox) tokensBox.innerHTML = `<em>Token sequence mapped to vocabulary IDs (padded to 50 tokens).</em>`;

      showToast("Inputs and inspection panels cleared.");
    });
  }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

window.addEventListener("DOMContentLoaded", () => {
  checkSystemStatus();
  setupDropZone();
  setupHeadlineInput();
  setupQuickSamples();
  setupAnalyzeForm();
});
