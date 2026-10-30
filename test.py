"""Evaluate a BiCueCC Gehler-Shi checkpoint."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bicuecc import BiCueCC
from bicuecc.data import EvaluationDataset
from bicuecc.metrics import angular_errors_deg, summarize_angular_errors


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-csv", default="results/predictions.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--npy-channel-order", choices=["bgr", "rgb"], default="bgr")
    args = parser.parse_args()
    model = BiCueCC(visual_embed=160, hist_channels=56, hist_embed=128,
                   local_embed=128, local_channels=64).to(args.device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    dataset = EvaluationDataset(args.csv, args.data_root, npy_channel_order=args.npy_channel_order)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "pred_r", "pred_g", "pred_b", "gt_r", "gt_g", "gt_b", "angular_error"])
        for images, targets, paths in tqdm(loader, desc="Testing"):
            pred = model(images.to(args.device))
            batch_errors = angular_errors_deg(pred, targets.to(args.device)).cpu()
            errors.extend(batch_errors.tolist())
            for path, p, t, error in zip(paths, pred.cpu().tolist(), targets.tolist(), batch_errors.tolist()):
                writer.writerow([path, *p, *t, error])
    print(json.dumps(summarize_angular_errors(errors), indent=2))


if __name__ == "__main__":
    main()
