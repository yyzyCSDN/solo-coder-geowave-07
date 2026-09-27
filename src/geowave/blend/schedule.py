from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np

@dataclass(frozen=True)
class SweepEvent:
    source_id: str
    sweep_index: int
    fire_time: float
    dither: float

    @property
    def delay(self) -> float:
        return self.fire_time + self.dither

@dataclass(frozen=True)
class BlendingSchedule:
    record_id: str
    dt: float
    n_samples: int
    sweeps: tuple[SweepEvent, ...]
    seed: int

    @classmethod
    def generate(cls, record_id: str, source_ids: list[str], dt: float, n_samples: int,
                 seed: int, n_sweeps: int, slot_interval: float,
                 dither_max: float) -> BlendingSchedule:
        if dt <= 0 or n_samples <= 0:
            raise ValueError('invalid time axis')
        if n_sweeps <= 0 or slot_interval <= 0 or dither_max < 0:
            raise ValueError('invalid sweep count, slot interval or dither')
        rng = np.random.default_rng(seed)
        n_sources = len(source_ids)
        sweeps = []
        for i, sid in enumerate(source_ids):
            for m in range(n_sweeps):
                fire = (m * n_sources + i) * slot_interval
                sweeps.append(SweepEvent(sid, m, fire, float(rng.uniform(0.0, dither_max))))
        return cls(record_id, dt, n_samples, tuple(sweeps), seed)

    @property
    def source_ids(self) -> list[str]:
        seen = []
        for s in self.sweeps:
            if s.source_id not in seen:
                seen.append(s.source_id)
        return seen

    @property
    def sweep_times(self) -> dict[str, tuple[float, ...]]:
        out: dict[str, list[float]] = {}
        for s in self.sweeps:
            out.setdefault(s.source_id, []).append(s.delay)
        return {sid: tuple(times) for sid, times in out.items()}

    def collisions(self, min_gap: float) -> list[tuple[SweepEvent, SweepEvent, float]]:
        ordered = sorted(self.sweeps, key=lambda s: s.delay)
        hits = []
        for a, b in zip(ordered, ordered[1:]):
            gap = b.delay - a.delay
            if gap < min_gap:
                hits.append((a, b, float(gap)))
        return hits

    def to_dict(self) -> dict:
        return {'record_id': self.record_id, 'dt': self.dt, 'n_samples': self.n_samples,
                'seed': self.seed,
                'sweeps': [{'source_id': s.source_id, 'sweep_index': s.sweep_index,
                            'fire_time': s.fire_time, 'dither': s.dither}
                           for s in self.sweeps]}

    @classmethod
    def from_dict(cls, data: dict) -> BlendingSchedule:
        sweeps = tuple(SweepEvent(s['source_id'], int(s['sweep_index']),
                                  float(s['fire_time']), float(s['dither']))
                       for s in data['sweeps'])
        return cls(data['record_id'], float(data['dt']), int(data['n_samples']),
                   sweeps, int(data['seed']))

    def save(self, path: str) -> None:
        with open(path, 'w') as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> BlendingSchedule:
        with open(path) as fh:
            return cls.from_dict(json.load(fh))
