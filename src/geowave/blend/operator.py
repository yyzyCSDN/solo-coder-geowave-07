from __future__ import annotations
import numpy as np
from geowave.blend.schedule import BlendingSchedule

def shift_gather(data: np.ndarray, tau: float, dt: float) -> np.ndarray:
    x = np.asarray(data, float)
    n = x.shape[-1]
    pad = int(np.ceil(abs(tau) / dt)) + 2
    zeros = np.zeros(x.shape[:-1] + (pad,))
    xp = np.concatenate([zeros, x, zeros], axis=-1)
    spec = np.fft.rfft(xp, axis=-1)
    freqs = np.fft.rfftfreq(xp.shape[-1], dt)
    spec = spec * np.exp(-2j * np.pi * freqs * tau)
    y = np.fft.irfft(spec, n=xp.shape[-1], axis=-1)
    return y[..., pad:pad + n]

class BlendOperator:
    def __init__(self, schedule: BlendingSchedule, shape: tuple[int, ...]):
        if shape[-1] != schedule.n_samples:
            raise ValueError(f'gather has {shape[-1]} samples, '
                             f'schedule expects {schedule.n_samples}')
        self.schedule = schedule
        self.shape = tuple(shape)
        times = schedule.sweep_times
        max_delay = max(t for ts in times.values() for t in ts)
        self.pad = int(np.ceil(max_delay / schedule.dt)) + 2
        self.n_padded = schedule.n_samples + 2 * self.pad
        freqs = np.fft.rfftfreq(self.n_padded, schedule.dt)
        self._phase = {}
        for sid, ts in times.items():
            phases = np.exp(-2j * np.pi * np.outer(freqs, np.asarray(ts)))
            self._phase[sid] = phases.sum(axis=1)

    def _pad(self, x: np.ndarray) -> np.ndarray:
        zeros = np.zeros(x.shape[:-1] + (self.pad,))
        return np.concatenate([zeros, x, zeros], axis=-1)

    def _crop(self, xp: np.ndarray) -> np.ndarray:
        return xp[..., self.pad:self.pad + self.schedule.n_samples]

    def apply(self, estimates: dict[str, np.ndarray]) -> np.ndarray:
        total = None
        for sid, phase in self._phase.items():
            if sid not in estimates:
                raise KeyError(f'missing estimate for source: {sid}')
            spec = np.fft.rfft(self._pad(np.asarray(estimates[sid], float)), axis=-1)
            total = phase * spec if total is None else total + phase * spec
        if total is None:
            raise ValueError('schedule has no sources')
        return self._crop(np.fft.irfft(total, n=self.n_padded, axis=-1))

    def adjoint(self, record: np.ndarray) -> dict[str, np.ndarray]:
        spec = np.fft.rfft(self._pad(np.asarray(record, float)), axis=-1)
        return {sid: self._crop(np.fft.irfft(np.conj(phase) * spec,
                                             n=self.n_padded, axis=-1))
                for sid, phase in self._phase.items()}

    def lipschitz(self, n_iter: int = 50, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        v = {sid: rng.standard_normal(self.shape) for sid in self._phase}
        norm = 0.0
        for _ in range(n_iter):
            w = self.adjoint(self.apply(v))
            norm = float(np.sqrt(sum(np.sum(a * a) for a in w.values())))
            if norm == 0.0:
                raise ValueError('operator is zero')
            v = {sid: w[sid] / norm for sid in w}
        return float(sum(np.sum(self.adjoint(self.apply(v))[sid] * v[sid])
                         for sid in v))

def blend(shots: dict[str, np.ndarray], schedule: BlendingSchedule) -> np.ndarray:
    first = next(iter(shots.values()))
    op = BlendOperator(schedule, np.asarray(first).shape)
    return op.apply(shots)

def pseudo_deblend(record: np.ndarray, schedule: BlendingSchedule) -> dict[str, np.ndarray]:
    op = BlendOperator(schedule, np.asarray(record).shape)
    counts = {sid: len(ts) for sid, ts in schedule.sweep_times.items()}
    return {sid: panel / counts[sid] for sid, panel in op.adjoint(record).items()}
