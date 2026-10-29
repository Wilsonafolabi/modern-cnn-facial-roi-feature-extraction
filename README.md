<<<<<<< HEAD
# CNN-Based Feature Extraction from Facial Regions of Interest

A computer-vision experiment that extracts and visualises convolutional activation maps from five facial regions using three CNN configurations.

## Objective

Inspect how the convolutional representations produced for a full-face crop and four approximate facial-region crops differ across:

1. **Custom CNN (untrained):** randomly initialised reference model; it is not a learned facial-feature baseline.
2. **ResNet-50 (ImageNet):** ImageNet-pretrained ResNet-50 used as a frozen feature extractor.
3. **ResNet-50 (VGGFace2):** ResNet-50 loaded from the `cydonia999/VGGFace2-pytorch` implementation and its associated pretrained checkpoint.

The experiment does not train a model or measure identity-recognition accuracy. It is a descriptive feature-extraction and visualisation study.

## Regions of interest

- Full Face
- Left Eye
- Right Eye
- Nose
- Mouth

The notebook uses dlib's 68-point facial landmark predictor to define eye, nose, and mouth crops. The full-face crop comes from the detected face rectangle. Landmark-derived crops are still approximate regions, not pixel-perfect anatomical masks; localisation accuracy requires separately annotated ground-truth boxes.

## Pipeline

1. Load an image and detect a face.
2. Extract the five facial crops.
3. Resize each crop to 224 × 224 and apply ImageNet mean/std normalisation, as in the notebook.
4. Capture the custom CNN output or the ResNet-50 `layer4` activation map before global average pooling.
5. Save activation-map shape and descriptive statistics (min, max, mean, standard deviation, zero fraction, finite fraction, L2 norm, and forward-pass time).
6. Visualise selected channels and intermediate stages.
7. Compute cosine similarity between flattened activation maps for the five regions within each model.

## Repository layout

```text
notebooks/cnn_face_2_0.ipynb       Original Colab notebook
src/facial_roi/models.py           Model definitions/loading and stage extraction
src/facial_roi/roi_extraction.py   Face and landmark-based ROI crops
src/facial_roi/feature_extraction.py Preprocessing and activation statistics
src/facial_roi/visualization.py    ROI and feature-map figures
src/facial_roi/similarity.py       Cosine-similarity utilities
scripts/run_experiment.py          Local command-line entry point
results/                           Destination for generated outputs
```

## Setup

Python 3.10–3.11 is recommended for compatibility with the older dlib and VGGFace2 checkpoint dependencies. Install a PyTorch/torchvision pair appropriate for your system first if the default package resolver does not match your CUDA setup.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dlib 68-landmark predictor is downloaded on first run into the configured model path. The VGGFace2 checkpoint is also downloaded if no local checkpoint is present. Network access is therefore needed on the first run. Checkpoint licensing and source terms should be reviewed before redistribution; this repository intentionally does not include model weights or face images.

## Run

From the repository root in PowerShell:

```powershell
python scripts/run_experiment.py --image path\to\face.jpg --out results\example_run
```

Use a clear, front-facing image. The script writes a CSV of per-model/per-ROI statistics, ROI coordinates, and figures under the chosen output directory. Do not commit private face images or generated crops without the subject's permission.

The original notebook is designed for Google Colab and includes interactive upload/download cells. It is retained for provenance and convenient notebook-based exploration; the `src/` modules are the reusable local implementation.

## Interpreting the results

- Feature-map dimensions describe the representation's channel and spatial sizes, not its accuracy.
- Channel images are individually normalised for display, so their colours should not be compared as absolute activation magnitudes across channels or models.
- A higher activation standard deviation, larger L2 norm, or higher zero fraction does **not** prove better face recognition.
- Flattened-map cosine similarity measures geometric similarity between those activation arrays. High similarity across crops does not by itself prove that a model has learned identity-specific context or distinguishes regions better.
- The custom CNN is untrained; its activations are random-initialisation responses.
- The experiment does not establish which model is best at face recognition. That claim requires an identity-labelled benchmark and a suitable protocol (for example, verification ROC-AUC/EER or TAR at a specified FAR).
- Timing is hardware- and runtime-dependent; a single pass is not a robust performance benchmark.

## Limitations

- A single input image is not sufficient to generalise across subjects, poses, lighting, or demographic groups.
- Landmark-derived crops depend on face detection and landmark quality.
- The ImageNet normalisation is applied to all three configurations to preserve the notebook's comparison; the VGGFace2 implementation/checkpoint may have its own recommended preprocessing, which should be verified for a strict model comparison.
- No training, identity classification, or recognition-accuracy evaluation is performed.

## Reproducibility

- Custom CNN seed: `42`.
- Input resolution: `224 × 224`.
- ResNet feature-map extraction point: `layer4`, before global average pooling.
- The notebook and modules save descriptive outputs; generated results depend on the input image, package versions, hardware, and checkpoint availability.

## License

No license is declared yet. Add a license only after deciding how you want the original code to be reused and checking the terms for any third-party code/checkpoints. The VGGFace2 checkpoint and dlib landmark predictor are not bundled here.
=======
# modern-cnn-facial-roi-feature-extraction
Comparative study of CNN feature extraction from facial regions of interest (ROIs) using a custom CNN, ImageNet-pretrained ResNet-50, and VGGFace2 fine-tuned ResNet-50 across full-face, eye, nose, and mouth regions.
>>>>>>> 6681e5ff759af5edfca155af9d5537ad5e74cfc5
