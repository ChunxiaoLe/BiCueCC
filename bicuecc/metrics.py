import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def angular_errors_deg(pred, target, eps=1e-7):
    pred = F.normalize(pred, dim=1, eps=eps)
    target = F.normalize(target, dim=1, eps=eps)
    cosine = torch.sum(pred * target, dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def summarize_angular_errors(errors):
    values = np.asarray(errors, dtype=np.float64)
    values = np.sort(values)
    if values.size == 0:
        raise ValueError("Cannot summarize empty error array.")

    q1 = np.percentile(values, 25)
    q2 = np.percentile(values, 50)
    q3 = np.percentile(values, 75)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "trimean": float((q1 + 2 * q2 + q3) / 4),
        "best25": float(np.mean(values[: max(1, values.size // 4)])),
        "worst25": float(np.mean(values[-max(1, values.size // 4) :])),
        "p95": float(np.percentile(values, 95)),
    }

