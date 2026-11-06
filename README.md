# NewsLens — Multimodal News Analysis Demo

A local-first web app for the Neural Networks & Deep Learning project. It combines the two existing model repositories at inference time: [text model repository](https://github.com/Bhargav-Kumar2007/NEWS-detection-using-text) and [image model repository](https://github.com/Bhargav-Kumar2007/NEWS-detection-using-image).

- **Text branch:** the BiLSTM model from `NEWS-detection-using-text`. It receives exactly `headline + " " + OCR(image)`.
- **Image branch:** the CNN model from `NEWS-detection-using-image`. It receives the original uploaded image.
- **OCR:** RapidOCR (pretrained, ONNX Runtime) extracts visible text from the uploaded image.
- **Comparison:** both model outputs are displayed independently. When the text label mapping is explicitly configured, the app also shows an unweighted mean of fake-class probabilities. That is an illustrative comparison, not a trained fusion head or calibrated confidence.

## Important checkpoint note

The public text repository contains a run folder at `outputs/bilstm_50k_20ep/` with a BiLSTM checkpoint and tokenizer.

The public image repository's visible training artifacts are under `outputs/recovery/subset_resnet18/` and relate to the **ReCOVery dataset**. The repository does not currently show a Fakeddit image checkpoint at `outputs/subset_resnet18/checkpoints/best.pt`. Do not substitute a ReCOVery checkpoint for your Fakeddit model. The image branch becomes active once you place your actual trained Fakeddit checkpoint at the expected path or set `IMAGE_CHECKPOINT` in `.env`.

## Windows setup (recommended for tomorrow's live demo)

1. Install **Python 3.10 or 3.11**, **Git for Windows**, and (if needed for model assets) **Git LFS**.
2. Extract this folder.
3. Double-click `setup_models.bat`. It clones both model repositories into `models/`.
4. Copy your trained Fakeddit image checkpoint to:

   `models\NEWS-detection-using-image\outputs\subset_resnet18\checkpoints\best.pt`

   Create the parent folders if they do not exist. If your image checkpoint has another name/location, edit `IMAGE_CHECKPOINT` in `.env`.
5. Run `setup.bat` to create a virtual environment and install dependencies.
6. Open `.env`. Verify the text label mapping before enabling the combined score:

   ```env
   TEXT_LABEL_0=FAKE
   TEXT_LABEL_1=REAL
   ```

   **Do not blindly copy this mapping.** The text repository's README says its checkpoint still uses `label_0` / `label_1` and asks the user to verify their meaning. Confirm the mapping from the label IDs used in your training data/checkpoint first. Leave both blank until verified; text predictions will still work, but the combined score will stay disabled.
7. Double-click `run.bat`, then open **http://127.0.0.1:8000** in Chrome or Edge.

The first OCR/model initialization may download pretrained OCR runtime assets; run once before the presentation and check that `/api/status` shows the expected components ready.

## Using your own local checkpoint paths

Edit `.env`:

```env
TEXT_REPO=models/NEWS-detection-using-text
IMAGE_REPO=models/NEWS-detection-using-image
TEXT_RUN_DIR=outputs/bilstm_50k_20ep
IMAGE_CHECKPOINT=outputs/subset_resnet18/checkpoints/best.pt
MODEL_DEVICE=auto
```

`TEXT_RUN_DIR` is relative to `TEXT_REPO`; `IMAGE_CHECKPOINT` is relative to `IMAGE_REPO` unless you provide an absolute path.

## What happens when Analyze is clicked?

1. The browser sends the headline and image to the local FastAPI backend.
2. RapidOCR recognizes text visible in the image.
3. The backend concatenates the typed headline, one space, and OCR text.
4. The text predictor runs `FakeNewsPredictor.predict()` on that combined string.
5. The image predictor runs `Predictor.predict()` on the original RGB image.
6. The page shows OCR text, the exact string used by the text model, separate model outputs, and a comparison only when both branches return compatible fake/real probabilities.

## Notes for the live demonstration

- To demonstrate the **complete** workflow, provide both a headline and an image that contains visible text.
- Use a fresh image/headline combination for each case. The case counter increments after a successful API response.
- Model scores represent learned dataset patterns. They do not verify a real-world claim, the source, or the event.
- The text and image models were trained independently; the simple mean is not a jointly optimized multimodal model.
- If the image model is unavailable, the text path and OCR can still be demonstrated. The image card will clearly show why it could not run.

## Project layout

```text
NewsLensDemo/
├── app.py                 # FastAPI backend + model adapters + OCR
├── web/
│   ├── index.html         # UI and explainers
│   ├── styles.css         # Responsive visual system
│   └── app.js             # Upload, status, pipeline animation, results
├── models/                # Cloned model repositories go here
├── .env.example
├── requirements.txt
├── setup_models.bat
├── setup.bat
└── run.bat
```

## Troubleshooting

- **Text model missing:** confirm `models/NEWS-detection-using-text/outputs/bilstm_50k_20ep/checkpoints/best_model.pt` and `tokenizer.json` exist.
- **Image model missing:** confirm the actual Fakeddit `.pt` checkpoint exists at `IMAGE_CHECKPOINT`; a tiny ~100-byte file is likely a Git LFS pointer, not model weights.
- **PyTorch install issue:** use the official PyTorch install selector for your Python version and hardware, then rerun `pip install -r requirements.txt`.
- **OCR engine missing:** activate `.venv` and run `python -m pip install rapidocr-onnxruntime`.
- **Port 8000 occupied:** stop the other local server or change the port in `run.bat`.
