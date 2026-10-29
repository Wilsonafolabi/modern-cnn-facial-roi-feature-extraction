"""Run the three-model, five-ROI feature extraction experiment locally."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from facial_roi.models import load_models, forward_feature_map, extract_progressive_stages
from facial_roi.roi_extraction import load_image, extract_rois
from facial_roi.feature_extraction import extract_feature_map
from facial_roi.visualization import plot_rois, plot_feature_maps, plot_progressive_stages
from facial_roi.similarity import cosine_similarity_matrix, plot_similarity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to one face image")
    parser.add_argument("--out", default="results/run", help="Output directory")
    parser.add_argument("--predictor", default="models/shape_predictor_68_face_landmarks.dat")
    args = parser.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = load_image(args.image)
    regions, boxes = extract_rois(image, args.predictor)
    plot_rois(regions, out / "figures" / "roi_crops.png")
    models = load_models(device=device)

    all_maps, rows, vectors = {}, [], {}
    for model_name, model in models.items():
        for region_name, crop in regions.items():
            fmap, stats = extract_feature_map(model_name, model, crop, device, forward_feature_map)
            all_maps[(model_name, region_name)] = fmap
            vectors.setdefault(model_name, []).append(fmap.reshape(-1).numpy())
            rows.append({"model": model_name, "region": region_name, **stats})
        stages = extract_progressive_stages(model_name, model, __import__("facial_roi.feature_extraction", fromlist=["prepare_input"]).prepare_input(regions["Full Face"], device))
        plot_progressive_stages(stages, f"Progressive feature maps — {model_name}", output_path=out / "figures" / (model_name.lower().replace(" ", "_").replace("-", "") + "_stages.png"))
        plot_feature_maps(all_maps, [model_name], "Full Face", output_path=out / "figures" / (model_name.lower().replace(" ", "_").replace("-", "") + "_full_face_maps.png"))

    pd.DataFrame(rows).to_csv(out / "feature_extraction_results.csv", index=False)
    (out / "roi_boxes.json").write_text(json.dumps({k: list(v) for k, v in boxes.items()}, indent=2))
    # Per-model ROI similarities: flattened map dimensions are consistent within each model.
    for model_name in models:
        ordered = [all_maps[(model_name, r)] for r in regions]
        matrix = cosine_similarity_matrix(ordered)
        plot_similarity(matrix, list(regions), f"ROI cosine similarity — {model_name}", output_path=out / "figures" / (model_name.lower().replace(" ", "_").replace("-", "") + "_roi_similarity.png"))
    print(f"Saved outputs to: {out.resolve()}")
    print("Note: cosine similarity of flattened activation maps is descriptive, not a face-recognition accuracy evaluation.")

if __name__ == "__main__":
    main()
