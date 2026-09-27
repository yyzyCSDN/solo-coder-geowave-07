from __future__ import annotations
import numpy as np
from geowave.blend.schedule import BlendingSchedule
from geowave.blend.operator import BlendOperator

def relative_residual(record: np.ndarray, residual: np.ndarray) -> float:
    norm = np.linalg.norm(np.asarray(record, float))
    if norm == 0.0:
        raise ValueError('record is all zeros')
    return float(np.linalg.norm(np.asarray(residual, float)) / norm)

def source_leakage(estimates: dict[str, np.ndarray], residual: np.ndarray,
                   schedule: BlendingSchedule) -> dict[str, float]:
    op = BlendOperator(schedule, np.asarray(residual).shape)
    counts = {sid: len(ts) for sid, ts in schedule.sweep_times.items()}
    out = {}
    for sid, panel in op.adjoint(residual).items():
        est = np.asarray(estimates[sid], float)
        denom = counts[sid] * np.linalg.norm(est)
        out[sid] = float(np.linalg.norm(panel) / denom) if denom else float('inf')
    return out

def separation_error(estimates: dict[str, np.ndarray],
                     truth: dict[str, np.ndarray]) -> dict[str, float]:
    out = {}
    for sid, ref in truth.items():
        denom = np.linalg.norm(np.asarray(ref, float))
        diff = np.linalg.norm(np.asarray(estimates[sid], float) - ref)
        out[sid] = float(diff / denom) if denom else float('inf')
    return out
