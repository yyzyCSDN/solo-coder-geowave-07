from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from geowave.blend.schedule import BlendingSchedule
from geowave.blend.operator import blend
from geowave.blend.separate import SeparationConfig, separate
from geowave.blend.metrics import source_leakage, separation_error

class JobStatus(str, Enum):
    OK = 'ok'
    SCHEDULE_COLLISION = 'schedule_collision'
    MISSING_SOURCE_RECORD = 'missing_source_record'
    NOT_CONVERGED = 'not_converged'

@dataclass
class SeparationJobResult:
    status: JobStatus
    schedule_id: str
    estimates: dict[str, np.ndarray] | None = None
    residual_history: list[float] = field(default_factory=list)
    crosstalk: dict[str, float] | None = None
    truth_error: dict[str, float] | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is JobStatus.OK

    @property
    def final_residual(self) -> float | None:
        return self.residual_history[-1] if self.residual_history else None

    def summary(self) -> str:
        lines = [f'job {self.schedule_id}: status={self.status.value}']
        if self.residual_history:
            lines.append(f'  iterations={len(self.residual_history)} '
                         f'final_relative_residual={self.residual_history[-1]:.6f}')
        if self.crosstalk:
            worst = max(self.crosstalk.items(), key=lambda kv: kv[1])
            mean = float(np.mean(list(self.crosstalk.values())))
            lines.append(f'  crosstalk(leakage) worst={worst[0]}:{worst[1]:.4f} '
                         f'mean={mean:.4f}')
        for key, value in self.diagnostics.items():
            lines.append(f'  {key}={value}')
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        return {'status': self.status.value, 'schedule_id': self.schedule_id,
                'residual_history': list(self.residual_history),
                'crosstalk': self.crosstalk, 'truth_error': self.truth_error,
                'diagnostics': self.diagnostics}

def run_separation_job(shots: dict[str, np.ndarray], schedule: BlendingSchedule,
                       config: SeparationConfig | None = None,
                       min_delay_gap: float | None = None,
                       truth: dict[str, np.ndarray] | None = None) -> SeparationJobResult:
    cfg = config or SeparationConfig()
    gap = min_delay_gap if min_delay_gap is not None else schedule.dt
    hits = schedule.collisions(gap)
    if hits:
        return SeparationJobResult(
            JobStatus.SCHEDULE_COLLISION, schedule.record_id,
            diagnostics={'min_delay_gap': gap,
                         'collisions': [{'a': a.source_id, 'a_sweep': a.sweep_index,
                                         'b': b.source_id, 'b_sweep': b.sweep_index,
                                         'gap_s': g} for a, b, g in hits]})

    missing = [sid for sid in schedule.source_ids if sid not in shots]
    if missing:
        return SeparationJobResult(
            JobStatus.MISSING_SOURCE_RECORD, schedule.record_id,
            diagnostics={'missing_sources': missing,
                         'required_sources': schedule.source_ids})

    record = blend(shots, schedule)
    result = separate(record, schedule, cfg)
    residual = record - blend(result.estimates, schedule)
    crosstalk = source_leakage(result.estimates, residual, schedule)
    truth_error = separation_error(result.estimates, truth) if truth else None
    status = JobStatus.OK if result.converged else JobStatus.NOT_CONVERGED
    diagnostics = {'step_size': result.step_size,
                   'sweeps_per_source': {sid: len(ts)
                                         for sid, ts in schedule.sweep_times.items()}}
    if not result.converged:
        diagnostics.update({'tol': cfg.tol, 'max_iter': cfg.max_iter,
                            'stagnated': result.stagnated,
                            'final_relative_residual': result.final_residual})
    return SeparationJobResult(status, schedule.record_id, result.estimates,
                               result.residual_history, crosstalk, truth_error, diagnostics)
