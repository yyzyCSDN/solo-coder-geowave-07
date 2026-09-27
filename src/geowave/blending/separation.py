"""Iterative deblending (separation) with sparse-domain crosstalk suppression.

The separation solves the blending inverse problem with a sparse-promoting
iteration in the common-receiver f-k domain. At every receiver, the panel is
laid out as (firing position/source, time): the desired arrivals vary
smoothly (linearly) from source to source and therefore concentrate on a few
f-k ridges, while interference from other sources is dumped at random dither
positions that are incoherent along that axis.

The iteration is a coherent-part peeling (basis-pursuit style) scheme:

    coherent_k = prox_lambda ( residual_k )                  # sparse filter
    x_{k+1}    = x_k + coherent_k
    residual   = residual_k - coherent_k + step A^T(b - A x_{k+1})

with a geometrically decaying threshold. Each iteration records the data
residual and a crosstalk estimate so a run can be judged without ground
truth. Failure modes are returned as a decidable status, never raised:

* SCHEDULE_COLLISION  - the schedule places sources too close together
* MISSING_RECORD      - an activated source record or blended record is absent
* INVALID_INPUT       - shapes/sample counts do not match the schedule
* NOT_CONVERGED       - iterations exhausted or residual progress stalled
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import numpy as np
from geowave.model.traces import Trace, TraceHeader, TraceSet
from .operator import (
    BlendedData, InputIssue, forward_cube, adjoint_cube, ata_crosstalk_cube,
    check_blended_input, MISSING_RECORD, MISSING_SOURCE_RECORD, MISSING_RECEIVER,
)
from .schedule import BlendingSchedule, ScheduleIssue, COLLISION, DUPLICATE_SOURCE

class SeparationStatus(str, Enum):
    OK_CONVERGED = 'converged'
    SCHEDULE_COLLISION = 'schedule-collision'
    MISSING_RECORD = 'missing-record'
    INVALID_INPUT = 'invalid-input'
    NOT_CONVERGED = 'not-converged'

@dataclass(frozen=True)
class SeparationConfig:
    max_iterations: int = 50
    residual_tol: float = 5e-2
    threshold_start_fraction: float = 0.3
    threshold_end_fraction: float = 0.1
    power_iterations: int = 12
    stall_iterations: int = 8
    stall_min_progress: float = 1e-4

    def __post_init__(self):
        if self.max_iterations < 1:
            raise ValueError('max_iterations must be >= 1')
        if not 0 < self.residual_tol < 1:
            raise ValueError('residual_tol must be inside (0, 1)')
        if not 0 < self.threshold_end_fraction <= self.threshold_start_fraction <= 1:
            raise ValueError('need 0 < end_fraction <= start_fraction <= 1')
        if self.power_iterations < 1:
            raise ValueError('power_iterations must be >= 1')
        if self.stall_iterations < 1:
            raise ValueError('stall_iterations must be >= 1')

    def fingerprint(self) -> str:
        raw = json.dumps(self.__dict__, sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(raw).hexdigest()

@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    residual_norm: float
    relative_residual: float
    crosstalk_ratio: float
    threshold: float

@dataclass(frozen=True)
class CrosstalkReport:
    mean_ratio: float                 # per-interferer leak level, mean over sources
    max_ratio: float
    per_source_ratio: dict[str, float]
    truth_relative_error: dict[str, float] = field(default_factory=dict)

@dataclass
class SeparationResult:
    status: SeparationStatus
    reason: str
    config: SeparationConfig
    schedule: BlendingSchedule
    receiver_ids: tuple[str, ...]
    shots: dict[str, np.ndarray]      # source_id -> (n_receivers, record_samples)
    iterations: list[IterationRecord]
    crosstalk: CrosstalkReport | None
    schedule_issues: list[ScheduleIssue]
    input_issues: list[InputIssue]

    @property
    def converged(self) -> bool:
        return self.status is SeparationStatus.OK_CONVERGED

    @property
    def residual_history(self) -> list[float]:
        return [it.relative_residual for it in self.iterations]

    def residual_curve(self) -> np.ndarray:
        return np.asarray(self.residual_history, dtype=float)

    def deblended_traceset(self, name: str = 'deblended') -> TraceSet:
        dt = self.schedule.dt_s
        traces: list[Trace] = []
        for source_id in self.schedule.source_ids:
            panel = self.shots[source_id]
            for irecv, receiver_id in enumerate(self.receiver_ids):
                header = TraceHeader(
                    trace_id=f'{source_id}-{receiver_id}', shot_id=source_id,
                    receiver_id=receiver_id, source_x=0.0, source_y=0.0,
                    receiver_x=0.0, receiver_y=0.0, dt_s=dt, start_s=0.0)
                traces.append(Trace(header, panel[irecv],
                                    tags={'process': 'deblend',
                                          'schedule': self.schedule.schedule_id}))
        return TraceSet(traces, name)

def _cr_fk_soft_threshold(shots: np.ndarray, threshold: float) -> np.ndarray:
    """Soft threshold in the orthonormal common-receiver f-k domain.

    shots is (n_source, n_receiver, n_time). Rearranged to per-receiver
    (source, time) panels, f-k transformed, pointwise soft-thresholded and
    inverted.
    """
    panels = np.swapaxes(shots, 0, 1).copy()  # (n_receiver, n_source, n_time)
    for irecv in range(panels.shape[0]):
        coeffs = np.fft.fft2(panels[irecv], norm='ortho')
        magnitude = np.abs(coeffs)
        shrink = np.maximum(magnitude - threshold, 0.0) / np.where(
            magnitude > 0, magnitude, 1.0)
        panels[irecv] = np.real(np.fft.ifft2(coeffs * shrink, norm='ortho'))
    return np.swapaxes(panels, 0, 1)

def _crosstalk(shots: np.ndarray, schedule: BlendingSchedule,
               source_index: dict[str, int]) -> CrosstalkReport:
    """Estimate crosstalk through the off-diagonal part of A^T A.

    Re-blending the current estimate and pseudo-deblending it back isolates
    the energy that other sources leak into each source panel. The raw level
    is divided by sqrt(m), where m is the number of interfering sources, so
    it reads as a per-neighbour amount (a perfect model blended with m equal
    neighbours otherwise measures ~1.0).
    """
    leak = ata_crosstalk_cube(shots, schedule, source_index)
    per_source: dict[str, float] = {}
    for isrc, source_id in enumerate(schedule.source_ids):
        m = sum(len(rec.activations) - 1
                for rec in schedule.records
                for act in rec.activations if act.source_id == source_id)
        scale = np.sqrt(max(m, 1))
        denom = float(np.linalg.norm(shots[isrc]))
        per_source[source_id] = float(
            np.linalg.norm(leak[isrc]) / (scale * denom + 1e-30))
    ratios = np.asarray(list(per_source.values()), dtype=float)
    return CrosstalkReport(float(ratios.mean()), float(ratios.max()), per_source)

def _truth_error(shots: dict[str, np.ndarray], truth: dict[str, np.ndarray]
                 ) -> dict[str, float]:
    return {source_id: float(np.linalg.norm(shots[source_id] - true_panel)
                             / (np.linalg.norm(true_panel) + 1e-30))
            for source_id, true_panel in truth.items()}

def _operator_step(blended_shape, schedule: BlendingSchedule, n_source: int,
                   source_index: dict[str, int], n_power: int) -> float:
    """1 / largest singular value of A, estimated by power iteration on A^T A."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal((n_source, blended_shape[1], schedule.record_samples))
    v /= np.linalg.norm(v)
    for _ in range(n_power):
        w = adjoint_cube(forward_cube(v, schedule, source_index), schedule, n_source)
        norm = np.linalg.norm(w)
        v = w / norm if norm > 0 else v
    eigenvalue = np.linalg.norm(
        adjoint_cube(forward_cube(v, schedule, source_index), schedule, n_source))
    return 1.0 / max(eigenvalue, 1e-30)

def _failed(status: SeparationStatus, reason: str, config: SeparationConfig,
            schedule: BlendingSchedule, receiver_ids: tuple[str, ...],
            schedule_issues: list[ScheduleIssue],
            input_issues: list[InputIssue]) -> SeparationResult:
    return SeparationResult(
        status, reason, config, schedule, receiver_ids, {}, [], None,
        schedule_issues, input_issues)

def separate(data: BlendedData, config: SeparationConfig | None = None,
             truth: dict[str, np.ndarray] | None = None) -> SeparationResult:
    """Run iterative separation and always return a judgeable result.

    The status is the first thing to inspect: only ``OK_CONVERGED`` carries
    separated shots whose data residual met the tolerance. Non-converged runs
    still return the best estimate plus the full residual/crosstalk history.
    """
    config = config or SeparationConfig()
    schedule = data.schedule
    receiver_ids = tuple(data.receiver_ids)

    schedule_issues = schedule.validate()
    if schedule_issues:
        status = (SeparationStatus.SCHEDULE_COLLISION
                  if any(i.code in {COLLISION, DUPLICATE_SOURCE} for i in schedule_issues)
                  else SeparationStatus.INVALID_INPUT)
        return _failed(status, 'schedule validation failed', config, schedule,
                       receiver_ids, schedule_issues, [])

    input_issues = check_blended_input(data)
    if input_issues:
        missing_codes = {MISSING_RECORD, MISSING_SOURCE_RECORD, MISSING_RECEIVER}
        status = (SeparationStatus.MISSING_RECORD
                  if any(i.code in missing_codes for i in input_issues)
                  else SeparationStatus.INVALID_INPUT)
        return _failed(status, 'blended data do not match schedule', config,
                       schedule, receiver_ids, [], input_issues)

    source_index = {sid: i for i, sid in enumerate(schedule.source_ids)}
    blended_cube = np.stack([data.records[rid] for rid in schedule.record_ids])
    data_norm = float(np.linalg.norm(blended_cube))
    n_source = len(schedule.source_ids)
    step = _operator_step(blended_cube.shape, schedule, n_source, source_index,
                          config.power_iterations)

    pseudo = adjoint_cube(blended_cube, schedule, n_source)
    coeff_peak = max(
        float(np.max(np.abs(np.fft.fft2(panel, norm='ortho'))))
        for panel in np.swapaxes(pseudo, 0, 1))
    threshold0 = max(config.threshold_start_fraction * coeff_peak, 1e-30)

    # Coherent-part peeling state: estimate x plus a model-space residual.
    estimate = np.zeros_like(pseudo)
    model_residual = pseudo.copy()
    history: list[IterationRecord] = []
    best_rel = np.inf
    stall = 0
    status = SeparationStatus.NOT_CONVERGED
    reason = (f'relative residual above tolerance {config.residual_tol:g} '
              f'after {config.max_iterations} iterations')

    for k in range(config.max_iterations):
        threshold = threshold0 * (config.threshold_end_fraction
                                  / config.threshold_start_fraction) ** (
            k / max(config.max_iterations - 1, 1))
        coherent = _cr_fk_soft_threshold(model_residual, threshold)
        estimate = estimate + coherent
        model_residual = model_residual - coherent

        data_residual = blended_cube - forward_cube(estimate, schedule, source_index)
        relative = float(np.linalg.norm(data_residual) / (data_norm + 1e-30))
        # Data-consistency reinjection into the model-space residual.
        model_residual = model_residual + step * adjoint_cube(
            data_residual, schedule, n_source)

        leak = _crosstalk(estimate, schedule, source_index)
        history.append(IterationRecord(k + 1,
                                       float(np.linalg.norm(data_residual)),
                                       relative, leak.mean_ratio, threshold))

        if relative <= config.residual_tol:
            status = SeparationStatus.OK_CONVERGED
            reason = f'relative residual {relative:.3e} <= {config.residual_tol:g}'
            break
        if relative < best_rel - config.stall_min_progress:
            best_rel = relative
            stall = 0
        else:
            stall += 1
            if stall >= config.stall_iterations:
                reason = (f'relative residual stalled at {relative:.3e} for '
                          f'{config.stall_iterations} iterations')
                break

    shots = {sid: estimate[i] for i, sid in enumerate(schedule.source_ids)}
    crosstalk = _crosstalk(estimate, schedule, source_index)
    if truth is not None:
        crosstalk = CrosstalkReport(
            crosstalk.mean_ratio, crosstalk.max_ratio, crosstalk.per_source_ratio,
            _truth_error(shots, truth))
    return SeparationResult(
        status, reason, config, schedule, receiver_ids, shots,
        history, crosstalk, [], [])
