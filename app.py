from __future__ import annotations

import importlib.util
import io
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
MODELS_DIR = ROOT / "models"
TEXT_REPO = Path(os.getenv("TEXT_REPO", str(MODELS_DIR / "NEWS-detection-using-text")))
IMAGE_REPO = Path(os.getenv("IMAGE_REPO", str(MODELS_DIR / "NEWS-detection-using-image")))
TEXT_RUN_DIR_RAW = os.getenv("TEXT_RUN_DIR", "outputs/bilstm_50k_20ep")
IMAGE_CHECKPOINT_RAW = os.getenv("IMAGE_CHECKPOINT", "outputs/subset_resnet18/checkpoints/best.pt")
TEXT_LABEL_0 = os.getenv("TEXT_LABEL_0", "FAKE").strip().lower()
TEXT_LABEL_1 = os.getenv("TEXT_LABEL_1", "REAL").strip().lower()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "12"))

STATE: dict[str, Any] = {
    "text_predictor": None,
    "image_predictor": None,
    "ocr_engine": None,
    "text_error": "Not initialized",
    "image_error": "Not initialized",
    "ocr_error": None,
    "startup_complete": False,
}


def _resolve_repo_path(repo: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (repo / candidate).resolve()


def _load_module(module_name: str, file_path: Path, import_root: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}")
    root_string = str(import_root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _initialize_text_model() -> None:
    repo = TEXT_REPO.expanduser().resolve()
    run_dir = _resolve_repo_path(repo, TEXT_RUN_DIR_RAW)
    checkpoint = run_dir / "checkpoints" / "best_model.pt"
    tokenizer = run_dir / "tokenizer.json"
    if not checkpoint.exists() or not tokenizer.exists():
        raise FileNotFoundError(
            "Text model files are missing. Expected both:\n"
            f"  {checkpoint}\n  {tokenizer}\n"
            "Run setup_models.bat, or set TEXT_REPO / TEXT_RUN_DIR in .env."
        )
    module = _load_module("newslens_text_predict", repo / "predict.py", repo)
    STATE["text_predictor"] = module.FakeNewsPredictor(run_dir=run_dir, device=os.getenv("MODEL_DEVICE", "auto"))
    STATE["text_error"] = None


def _initialize_image_model() -> None:
    repo = IMAGE_REPO.expanduser().resolve()
    checkpoint = _resolve_repo_path(repo, IMAGE_CHECKPOINT_RAW)
    if not checkpoint.exists():
        recovery_checkpoint = repo / "outputs/recovery/subset_resnet18/checkpoints/best.pt"
        if recovery_checkpoint.exists():
            checkpoint = recovery_checkpoint
        else:
            raise FileNotFoundError(
                "No trained CNN image checkpoint was found.\n"
                f"Expected checkpoint: {checkpoint}\n"
                "Copy your checkpoint or run setup_models.bat."
            )
    if checkpoint.stat().st_size < 1024:
        raise RuntimeError(
            f"{checkpoint} is only {checkpoint.stat().st_size} bytes. This may be a Git LFS pointer, "
            "not the actual PyTorch checkpoint. Install Git LFS and run `git lfs pull`."
        )
    module = _load_module("newslens_image_predict", repo / "predict.py", repo)
    STATE["image_predictor"] = module.Predictor(
        checkpoint=checkpoint, device=os.getenv("MODEL_DEVICE", "auto")
    )
    STATE["image_error"] = None


def _initialize_ocr() -> None:
    try:
        from rapidocr_onnxruntime import RapidOCR
        STATE["ocr_engine"] = RapidOCR()
        STATE["ocr_error"] = None
    except Exception as exc:
        STATE["ocr_error"] = f"OCR could not start: {exc}"


def _status_payload() -> dict[str, Any]:
    text_model = STATE["text_predictor"]
    image_model = STATE["image_predictor"]
    ocr_engine = STATE["ocr_engine"]
    text_ready = text_model is not None
    image_ready = image_model is not None
    return {
        "app": "NewsLens",
        "ready": text_ready and image_ready and ocr_engine is not None,
        "models": {
            "text": {
                "ready": text_ready,
                "name": "BiLSTM Text Classifier",
                "run_dir": str(getattr(text_model, "run_dir", "")) if text_model else None,
                "device": str(getattr(text_model, "device", "")) if text_model else None,
                "class_names": list(getattr(text_model, "class_names", [])) if text_model else [],
                "error": STATE["text_error"],
            },
            "image": {
                "ready": image_ready,
                "name": f"{getattr(image_model, 'arch', 'CNN')} Image Classifier" if image_model else "ResNet18 Image Classifier",
                "checkpoint": str(getattr(image_model, "checkpoint", "")) if image_model else str(_resolve_repo_path(IMAGE_REPO, IMAGE_CHECKPOINT_RAW)),
                "device": str(getattr(image_model, "device", "")) if image_model else None,
                "class_names": list(getattr(image_model, "class_names", [])) if image_model else [],
                "error": STATE["image_error"],
            },
            "ocr": {
                "ready": ocr_engine is not None,
                "name": "RapidOCR (ONNX Runtime)",
                "error": STATE["ocr_error"],
            },
        },
        "label_mapping": {
            "label_0": "FAKE",
            "label_1": "REAL",
            "verified": True,
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _initialize_text_model()
    except Exception as exc:
        STATE["text_error"] = str(exc)
    try:
        _initialize_image_model()
    except Exception as exc:
        STATE["image_error"] = str(exc)
    _initialize_ocr()
    STATE["startup_complete"] = True
    yield


app = FastAPI(title="NewsLens — Multimodal Fake News Detection", version="3.0.0", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")


@app.get("/")
def home():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/status")
def api_status():
    return _status_payload()


def _run_ocr(image: Image.Image) -> dict[str, Any]:
    engine = STATE["ocr_engine"]
    if engine is None:
        return {"ready": False, "text": "", "lines": [], "duration_ms": 0.0, "error": STATE["ocr_error"] or "OCR is unavailable."}
    start = time.perf_counter()
    try:
        raw = engine(np.array(image.convert("RGB")))
        items = raw[0] if isinstance(raw, tuple) else raw
        lines: list[dict[str, Any]] = []
        if items:
            for item in items:
                if not item or len(item) < 2:
                    continue
                recognized = str(item[1]).strip()
                if not recognized:
                    continue
                try:
                    score = float(item[2]) if len(item) > 2 else None
                except (TypeError, ValueError):
                    score = None
                lines.append({"text": recognized, "confidence": round(score, 4) if score is not None else None})
        text = " ".join(row["text"] for row in lines).strip()
        duration = round((time.perf_counter() - start) * 1000, 1)
        return {
            "ready": True,
            "text": text,
            "lines": lines,
            "duration_ms": duration,
            "error": None,
        }
    except Exception as exc:
        return {"ready": False, "text": "", "lines": [], "duration_ms": 0.0, "error": f"OCR failed: {exc}"}


def _text_fake_probability(prediction: dict[str, Any]) -> float | None:
    label_id = int(prediction.get("label", -1))
    p_label_1 = float(prediction.get("prob_label_1", 0.0))
    # Verified semantics: Label 0 = Fake, Label 1 = Real
    # Therefore p_label_1 is P(Real)
    # P(Fake) = 1.0 - p_label_1
    return 1.0 - p_label_1


@app.post("/api/analyze")
async def analyze(
    headline: str = Form(""),
    image_file: UploadFile | None = File(None),
):
    """
    CRITICAL MULTIMODAL INFERENCE PIPELINE:
    Step 1: Receive user headline and image.
    Step 2: Run RapidOCR to extract visible text from the image.
    Step 3: Concatenate: combined_text = headline + " " + ocr_text
    Step 4: Tokenize combined_text using the exact trained BiLSTM tokenizer (50 max tokens).
    Step 5: Run BiLSTM inference on the tokenized representation.
    Step 6: Independently preprocess image (224x224 RGB normalized) and run CNN ResNet-18 inference.
    Step 7: Compute multimodal late fusion and agreement signal.
    """
    pipeline_start = time.perf_counter()
    headline = headline.strip()
    if not headline and image_file is None:
        raise HTTPException(status_code=400, detail="Please enter a headline, upload an image, or provide both.")

    # 1. Load image
    pil_image: Image.Image | None = None
    original_name = None
    if image_file is not None:
        original_name = image_file.filename or "uploaded_image"
        content = await image_file.read()
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded image file is empty.")
        if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Image exceeds max size of {MAX_UPLOAD_MB} MB.")
        try:
            pil_image = Image.open(io.BytesIO(content)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail="The file could not be decoded as a valid image.")

    # 2. Run OCR on uploaded image
    if pil_image is not None:
        ocr_result = _run_ocr(pil_image)
    else:
        ocr_result = {
            "ready": False,
            "text": "",
            "lines": [],
            "duration_ms": 0.0,
            "error": "No image provided; OCR skipped.",
        }

    ocr_extracted_text = (ocr_result.get("text") or "").strip()

    # 3. Concatenate headline and OCR text
    # Exact specification: combined_text = headline + " " + ocr_text
    if headline and ocr_extracted_text:
        combined_text = f"{headline} {ocr_extracted_text}".strip()
    elif headline:
        combined_text = headline
    elif ocr_extracted_text:
        combined_text = ocr_extracted_text
    else:
        combined_text = ""

    # 4. Extract token breakdown from the BiLSTM tokenizer for transparent inspection
    text_predictor = STATE["text_predictor"]
    tokenizer_tokens = []
    if text_predictor is not None and hasattr(text_predictor, "tokenizer") and combined_text:
        tok = text_predictor.tokenizer
        tokenizer_tokens = tok.tokenize(combined_text)

    # 5. BiLSTM Text Model Inference
    text_result: dict[str, Any]
    text_start = time.perf_counter()
    if text_predictor is not None and combined_text:
        try:
            # Passes combined_text to FakeNewsPredictor.predict()
            # which runs clean_text, tokenizer.encode_batch(), and forward_features()
            pred = text_predictor.predict(combined_text)[0]
            p_label_1 = float(pred.get("prob_label_1", 0.0))
            p_real = round(p_label_1, 4)
            p_fake = round(1.0 - p_label_1, 4)
            label = "Real" if p_real >= 0.5 else "Fake"
            confidence = round(max(p_real, p_fake), 4)
            text_duration = round((time.perf_counter() - text_start) * 1000, 1)

            text_result = {
                "ready": True,
                "label": label,
                "confidence": confidence,
                "fake_probability": p_fake,
                "real_probability": p_real,
                "prob_label_1": p_real,
                "input_used": combined_text,
                "headline_used": headline,
                "ocr_text_used": ocr_extracted_text,
                "token_count": len(tokenizer_tokens),
                "tokens": tokenizer_tokens[:50],
                "tokens_truncated": len(tokenizer_tokens) > 50,
                "duration_ms": text_duration,
                "error": None,
            }
        except Exception as exc:
            text_result = {"ready": False, "error": f"BiLSTM inference failed: {exc}", "duration_ms": 0.0}
    else:
        text_result = {
            "ready": False,
            "error": STATE["text_error"] or "No text available for text branch.",
            "duration_ms": 0.0,
        }

    # 6. Image Model (CNN ResNet-18) Inference
    image_predictor = STATE["image_predictor"]
    image_result: dict[str, Any]
    image_start = time.perf_counter()
    if image_predictor is not None and pil_image is not None:
        try:
            pred = image_predictor.predict(pil_image)
            probs = {str(k).lower(): float(v) for k, v in pred.get("probabilities", {}).items()}
            p_fake = round(probs.get("fake", 0.0), 4)
            p_real = round(probs.get("real", 0.0), 4)
            label = "Fake" if p_fake >= 0.5 else "Real"
            confidence = round(max(p_fake, p_real), 4)
            image_duration = round((time.perf_counter() - image_start) * 1000, 1)

            image_result = {
                "ready": True,
                "label": label,
                "confidence": confidence,
                "fake_probability": p_fake,
                "real_probability": p_real,
                "architecture": str(getattr(image_predictor, "arch", "ResNet-18")),
                "image_resolution": f"{pil_image.width}×{pil_image.height}",
                "duration_ms": image_duration,
                "error": None,
            }
        except Exception as exc:
            image_result = {"ready": False, "error": f"CNN inference failed: {exc}", "duration_ms": 0.0}
    else:
        image_result = {
            "ready": False,
            "error": STATE["image_error"] or "No image provided for image branch.",
            "duration_ms": 0.0,
        }

    # 7. Multimodal Late Fusion & Agreement Analysis
    text_fake = text_result.get("fake_probability")
    img_fake = image_result.get("fake_probability")
    has_both = isinstance(text_fake, (int, float)) and isinstance(img_fake, (int, float))

    fusion: dict[str, Any]
    if has_both:
        mean_fake = round((float(text_fake) + float(img_fake)) / 2.0, 4)
        mean_real = round(1.0 - mean_fake, 4)
        signal = "FAKE-LIKE SIGNAL" if mean_fake >= 0.5 else "REAL-LIKE SIGNAL"

        text_pred_label = text_result.get("label")
        img_pred_label = image_result.get("label")
        is_agreement = text_pred_label == img_pred_label
        divergence = round(abs(float(text_fake) - float(img_fake)) * 100, 1)

        if is_agreement:
            agreement_status = f"Models Agree (Both Predict {text_pred_label})"
        else:
            agreement_status = f"Models Disagree (Text: {text_pred_label}, Image: {img_pred_label} — {divergence}% Divergence)"

        fusion = {
            "ready": True,
            "signal": signal,
            "fake_probability": mean_fake,
            "real_probability": mean_real,
            "confidence": round(max(mean_fake, mean_real), 4),
            "is_agreement": is_agreement,
            "agreement_status": agreement_status,
            "divergence_percent": divergence,
            "aggregation_method": "Unweighted late-fusion mean: P_fused = (P_text + P_image) / 2",
            "academic_notice": "Learned statistical signal from two independent neural architectures. Not proof of real-world factual truth.",
        }
    else:
        fusion = {
            "ready": False,
            "signal": "Awaiting Dual Branches",
            "fake_probability": None,
            "real_probability": None,
            "confidence": None,
            "is_agreement": False,
            "agreement_status": "Provide both headline and image to run the complete multimodal pipeline.",
            "divergence_percent": 0.0,
            "aggregation_method": None,
            "academic_notice": "Single-branch inference active.",
        }

    total_duration = round((time.perf_counter() - pipeline_start) * 1000, 1)

    return {
        "timing": {
            "ocr_ms": ocr_result.get("duration_ms", 0.0),
            "text_ms": text_result.get("duration_ms", 0.0),
            "image_ms": image_result.get("duration_ms", 0.0),
            "total_ms": total_duration,
        },
        "inspection": {
            "original_image_name": original_name,
            "headline": headline,
            "ocr_extracted_text": ocr_extracted_text,
            "ocr_lines_count": len(ocr_result.get("lines", [])),
            "combined_text": combined_text,
            "tokens_list": tokenizer_tokens[:50],
            "total_tokens_count": len(tokenizer_tokens),
            "concatenation_formula": 'combined_text = headline + " " + OCR(image)',
        },
        "ocr": ocr_result,
        "models": {
            "text": text_result,
            "image": image_result,
        },
        "fusion": fusion,
    }


@app.get("/api/model-info")
def api_model_info():
    """Returns verified model architecture, dataset, training metrics, and limitations."""
    return {
        "text_model": {
            "name": "BiLSTM Text Classifier",
            "task": "Classify combined headline + OCR text as Fake or Real",
            "architecture": "Learned Word Embedding (dim 200) → Bidirectional LSTM (128 units/dir = 256 total) → Dropout (0.4) → Dense Layer (128) → Sigmoid Output",
            "parameters": "3,693,513 trainable parameters",
            "dataset": "Fakeddit Reddit 2-way title dataset (clean_title, 2_way_label)",
            "split_sizes": "50,000 train / 5,000 validation / 5,000 test",
            "metrics": {
                "accuracy": "81.44%",
                "precision": "79.85%",
                "recall": "79.29%",
                "f1_score": "79.57%",
                "roc_auc": "89.63%",
            },
            "limitations": "Vocabulary capped at 16,449 words. Rare named entities (landmarks, local politicians) map to <unk>, causing the network to classify based strictly on surrounding syntactic markers.",
        },
        "image_model": {
            "name": "ResNet-18 Image Classifier",
            "task": "Classify original RGB news images as Fake or Real",
            "architecture": "ImageNet Pretrained ResNet-18 → 4 Residual Stages (BasicBlocks with skip connections) → Adaptive Average Pooling → Fully Connected Layer (2 units) → Softmax Output",
            "parameters": "~11.18 Million parameters",
            "dataset": "ReCOVery News Dataset (reliable vs. unreliable article images)",
            "split_sizes": "1,157 train / 137 validation / 142 test",
            "metrics": {
                "accuracy": "90.14%",
                "fake_precision": "100.0%",
                "fake_recall": "26.32%",
                "fake_f1": "41.67%",
                "roc_auc": "76.17%",
            },
            "limitations": "High class imbalance in training set (~90% real news photos). Natural photographic scenes (lighting, natural textures, landmark architecture) strongly activate 'Real' filters, causing high-confidence false negatives on realistic fabricated photos.",
        },
        "fusion_methodology": {
            "technique": "Unweighted Multimodal Late Fusion",
            "formula": "P_fused(Fake) = 0.5 × P_text(Fake) + 0.5 × P_image(Fake)",
            "rationale": "Displays both branches independently while providing an intuitive unified signal. When models disagree, the discordance highlights cross-modal anomalies.",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
