from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pywt
from geowave.blend.schedule import BlendingSchedule
from geowave.blend.operator import BlendOperator

@dataclass(frozen=True)
class SeparationConfig:
    max_iter: int = 150
    tol: float = 1e-2
    wavelet: str = 'db4'
    level: int = 3
    start_percentile: float = 99.5
    threshold_decay: float = 0.93
    threshold_floor: float = 0.002
    step_safety: float = 1.05
    stagnation_window: int = 12

@dataclass
class SeparationResult:
    estimates: dict[str, np.ndarray]
    residual_history: list[float]
    converged: bool
    step_size: float
    stagnated: bool = False

    @property
    def iterations(self) -> int:
        return len(self.residual_history)

    @property
    def final_residual(self) -> float:
        return self.residual_history[-1] if self.residual_history else float('nan')

class _SwtDomain:
    def __init__(self, wavelet: str, level: int, shape: tuple[int, ...]):
        self.wavelet = wavelet
        if len(shape) == 1:
            shape = (1, shape[0])
        max_level = max(1, int(np.log2(min(shape))))
        self.level = max(1, min(level, max_level))
        mult = 2 ** self.level
        self.shape = shape
        self.padded = tuple(int(np.ceil(s / mult)) * mult for s in shape)

    def _pad(self, x: np.ndarray) -> np.ndarray:
        x2 = np.asarray(x, float).reshape(self.shape)
        out = np.zeros(self.padded)
        out[:self.shape[0], :self.shape[1]] = x2
        return out

    def _crop(self, x: np.ndarray, ref: tuple[int, ...]) -> np.ndarray:
        return x[:self.shape[0], :self.shape[1]].reshape(ref)

    def forward(self, x: np.ndarray):
        return pywt.swt2(self._pad(x), self.wavelet, level=self.level)

    def inverse(self, coeffs, ref: tuple[int, ...]) -> np.ndarray:
        return self._crop(pywt.iswt2(coeffs, self.wavelet), ref)

    @staticmethod
    def soft(coeffs, lam: float):
        def shrink(a):
            return np.sign(a) * np.maximum(np.abs(a) - lam, 0.0)
        out = [(coeffs[0][0], tuple(shrink(a) for a in coeffs[0][1]))]
        for approx, details in coeffs[1:]:
            out.append((approx, tuple(shrink(a) for a in details)))
        return out

    @staticmethod
    def flatten(coeffs) -> np.ndarray:
        parts = [np.abs(coeffs[0][0]).ravel()]
        for _, details in coeffs:
            parts.extend(np.abs(a).ravel() for a in details)
        return np.concatenate(parts)

def separate(record: np.ndarray, schedule: BlendingSchedule,
             config: SeparationConfig | None = None) -> SeparationResult:
    cfg = config or SeparationConfig()
    d = np.asarray(record, float)
    record_norm = np.linalg.norm(d)
    if record_norm == 0.0:
        raise ValueError('record is all zeros')
    op = BlendOperator(schedule, d.shape)
    ids = schedule.source_ids
    step = 1.0 / (cfg.step_safety * op.lipschitz())
    domain = _SwtDomain(cfg.wavelet, cfg.level, d.shape)

    def denoise(x: np.ndarray, lam: float) -> np.ndarray:
        return domain.inverse(domain.soft(domain.forward(x), lam), x.shape)

    thresholds = {}
    for sid, panel in op.adjoint(d).items():
        coeffs = domain.flatten(domain.forward(step * panel))
        thresholds[sid] = float(np.percentile(coeffs, cfg.start_percentile))

    estimates = {sid: np.zeros_like(d) for sid in ids}
    previous = {sid: np.zeros_like(d) for sid in ids}
    extrapolated = {sid: np.zeros_like(d) for sid in ids}
    momentum = 1.0
    history: list[float] = []
    converged = False
    stagnated = False
    for k in range(cfg.max_iter):
        lam_scale = max(cfg.threshold_decay ** k, cfg.threshold_floor)
        residual = d - op.apply(extrapolated)
        gradient = op.adjoint(residual)
        for sid in ids:
            previous[sid] = estimates[sid]
            estimates[sid] = denoise(extrapolated[sid] + step * gradient[sid],
                                     thresholds[sid] * lam_scale)
        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum ** 2)) / 2.0
        weight = (momentum - 1.0) / next_momentum
        for sid in ids:
            extrapolated[sid] = estimates[sid] + weight * (estimates[sid] - previous[sid])
        momentum = next_momentum
        rel = float(np.linalg.norm(d - op.apply(estimates)) / record_norm)
        if history and rel > history[-1]:
            momentum = 1.0
            extrapolated = {sid: estimates[sid].copy() for sid in ids}
        history.append(rel)
        if rel < cfg.tol:
            converged = True
            break
        w = cfg.stagnation_window
        if len(history) > w and history[-1 - w] - history[-1] < 1e-3 * history[-1 - w]:
            stagnated = True
            break
    return SeparationResult(estimates, history, converged, step, stagnated)
