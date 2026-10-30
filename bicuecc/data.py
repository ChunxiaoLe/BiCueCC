"""Deterministic Gehler-Shi evaluation preprocessing."""
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class EvaluationDataset(Dataset):
    def __init__(self, csv_path, root, image_size=512, npy_channel_order="bgr"):
        self.root = Path(root)
        self.image_size = image_size
        self.channel_order = npy_channel_order
        with open(csv_path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"image_path", "illum_r", "illum_g", "illum_b"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"CSV must contain {sorted(required)}")
            self.rows = list(reader)
        if not self.rows:
            raise ValueError("Evaluation CSV is empty")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root / row["image_path"]
        if path.suffix.lower() == ".npy":
            image = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
            if self.channel_order == "bgr":
                image = image[:, :, ::-1]
        else:
            with Image.open(path) as source:
                image = np.asarray(source.convert("RGB"), dtype=np.float32)
        if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
            raise ValueError(f"Expected finite HWC three-channel image: {path}")
        maximum = float(np.max(image))
        if maximum > 0:
            image = image / maximum
        image = np.clip(image, 0, 1)
        if row.get("mask_path", "").strip():
            mask = np.asarray(np.load(self.root / row["mask_path"], allow_pickle=False), dtype=np.float32)
            if mask.ndim == 2:
                mask = mask[:, :, None]
            image = image * mask
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        tensor = torch.pow(tensor.clamp_min(0), 1 / 2.2)
        tensor = TF.resize(tensor, [self.image_size, self.image_size])
        tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        target = torch.tensor([float(row[f"illum_{c}"]) for c in "rgb"], dtype=torch.float32)
        if not torch.isfinite(target).all() or target.norm() <= 0:
            raise ValueError(f"Invalid illuminant for {path}")
        target = torch.nn.functional.normalize(target, dim=0)
        return tensor, target, row["image_path"]
