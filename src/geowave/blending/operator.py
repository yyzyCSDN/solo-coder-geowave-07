"""Forward blending and adjoint (pseudo-deblending) operators.

A blended record is the delay-and-sum of several single-shot gathers that
share a receiver layout. All delays are integer samples, so the operators are
pure array copies with no interpolation and are exactly reproducible.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import TraceSet
from .schedule import BlendingSchedule

# Input check issue codes.
MISSING_SOURCE_RECORD = 'missing-source-record'
MISSING_RECEIVER = 'missing-receiver'
MISSING_RECORD = 'missing-record'
BAD_SAMPLE_COUNT = 'bad-sample-count'
BAD_DT = 'bad-dt'
NON_FINITE = 'non-finite'

@dataclass(frozen=True)
class InputIssue:
    code: str
    record_id: str
    detail: str
    source_id: str = ''

@dataclass
class BlendedData:
    schedule: BlendingSchedule
    receiver_ids: tuple[str, ...]
    records: dict[str, np.ndarray]  # record_id -> (n_receivers, record_samples + max_delay)

    def __post_init__(self):
        expected_n = self.schedule.record_samples + self.schedule.max_delay_samples
        for record_id, cube in self.records.items():
            arr = np.asarray(cube, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != expected_n:
                raise ValueError(f'blended record {record_id} must have shape (n_receivers, {expected_n})')
            self.records[record_id] = arr

    @property
    def n_total_samples(self) -> int:
        return self.schedule.record_samples + self.schedule.max_delay_samples

def check_unblended_input(schedule: BlendingSchedule, dataset: TraceSet,
                          dt_tol: float = 1e-12) -> list[InputIssue]:
    """Find missing source records, missing receivers and shape mismatches."""
    issues: list[InputIssue] = []
    by_shot = dataset.by_shot()
    receivers_per_source: dict[str, set[str]] = {}
    for source_id, traces in by_shot.items():
        receivers_per_source[source_id] = set()
        for tr in traces:
            if abs(tr.header.dt_s - schedule.dt_s) > dt_tol:
                issues.append(InputIssue(BAD_DT, '',
                                         f'trace {tr.header.trace_id} dt {tr.header.dt_s} '
                                         f'!= schedule dt {schedule.dt_s}', source_id))
            if not np.all(np.isfinite(tr.samples)):
                issues.append(InputIssue(NON_FINITE, '',
                                         f'trace {tr.header.trace_id} contains non-finite samples',
                                         source_id))
            if tr.samples.size != schedule.record_samples:
                issues.append(InputIssue(BAD_SAMPLE_COUNT, '',
                                         f'trace {tr.header.trace_id} has {tr.samples.size} samples, '
                                         f'expected {schedule.record_samples}', source_id))
            receivers_per_source[source_id].add(tr.header.receiver_id)
    for rec in schedule.records:
        for act in rec.activations:
            if act.source_id not in by_shot:
                issues.append(InputIssue(MISSING_SOURCE_RECORD, rec.record_id,
                                         f'no traces for activated source {act.source_id}',
                                         act.source_id))
    present_layouts = [recv for sid, recv in receivers_per_source.items()
                       if sid in schedule.source_ids]
    if present_layouts:
        reference = set.union(*present_layouts)
        for rec in schedule.records:
            for act in rec.activations:
                layout = receivers_per_source.get(act.source_id)
                if layout is None:
                    continue
                missing = reference - layout
                for receiver_id in sorted(missing):
                    issues.append(InputIssue(MISSING_RECEIVER, rec.record_id,
                                             f'source {act.source_id} missing receiver {receiver_id}',
                                             act.source_id))
    return issues

def check_blended_input(data: BlendedData) -> list[InputIssue]:
    schedule = data.schedule
    issues: list[InputIssue] = []
    expected_n = data.n_total_samples
    expected_shape = (len(data.receiver_ids), expected_n)
    for rec_id in schedule.record_ids:
        cube = data.records.get(rec_id)
        if cube is None:
            issues.append(InputIssue(MISSING_RECORD, rec_id,
                                     f'blended record {rec_id} not provided'))
            continue
        if cube.shape != expected_shape:
            issues.append(InputIssue(BAD_SAMPLE_COUNT, rec_id,
                                     f'record {rec_id} shape {cube.shape} != {expected_shape}'))
        if not np.all(np.isfinite(cube)):
            issues.append(InputIssue(NON_FINITE, rec_id,
                                     f'record {rec_id} contains non-finite samples'))
    return issues

def _shift_add(dst: np.ndarray, src: np.ndarray, start: int) -> None:
    """Add src into dst along the last (time) axis at sample offset ``start``.

    Negative starts and overflow at either end are clipped to the overlap.
    """
    n_dst = dst.shape[1]
    n_src = src.shape[1]
    d0 = max(0, start)
    d1 = min(n_dst, start + n_src)
    if d1 <= d0:
        return
    s0 = d0 - start
    dst[:, d0:d1] += src[:, s0:s0 + (d1 - d0)]

def forward_cube(shot_cube: np.ndarray, schedule: BlendingSchedule,
                 source_index: dict[str, int]) -> np.ndarray:
    """Blend shot cube (n_source, n_receiver, record_samples) into records."""
    n_recv = shot_cube.shape[1]
    n_total = schedule.record_samples + schedule.max_delay_samples
    out = np.zeros((len(schedule.records), n_recv, n_total))
    for irec, rec in enumerate(schedule.records):
        for act in rec.activations:
            isrc = source_index[act.source_id]
            _shift_add(out[irec], shot_cube[isrc], act.delay_samples)
    return out

def adjoint_cube(blended_cube: np.ndarray, schedule: BlendingSchedule,
                 n_source: int) -> np.ndarray:
    """Pseudo-deblend: shift each record back and sum into per-source gathers."""
    n_recv = blended_cube.shape[1]
    out = np.zeros((n_source, n_recv, schedule.record_samples))
    for isrc, source_id in enumerate(schedule.source_ids):
        for irec, rec in enumerate(schedule.records):
            for act in rec.activations:
                if act.source_id == source_id:
                    _shift_add(out[isrc], blended_cube[irec], -act.delay_samples)
    return out

def ata_crosstalk_cube(shot_cube: np.ndarray, schedule: BlendingSchedule,
                       source_index: dict[str, int]) -> np.ndarray:
    """Cross-source part of the normal operator A^T A.

    The diagonal (a source contributing to itself) is removed, so the result
    is exactly the interference that other sources leak into an estimate
    through the blending operator.
    """
    leak = np.zeros_like(shot_cube)
    for rec in schedule.records:
        for target in rec.activations:
            itarget = source_index[target.source_id]
            for other in rec.activations:
                if other.source_id == target.source_id:
                    continue
                iother = source_index[other.source_id]
                _shift_add(leak[itarget], shot_cube[iother],
                           other.delay_samples - target.delay_samples)
    return leak

def shared_receiver_layout(dataset: TraceSet) -> tuple[str, ...]:
    receivers = {tr.header.receiver_id for tr in dataset.traces}
    return tuple(sorted(receivers))

def traceset_to_shot_cube(dataset: TraceSet, schedule: BlendingSchedule,
                          receiver_ids: tuple[str, ...]) -> np.ndarray:
    index = {(tr.header.shot_id, tr.header.receiver_id): tr for tr in dataset.traces}
    cube = np.zeros((len(schedule.source_ids), len(receiver_ids), schedule.record_samples))
    receiver_index = {rid: i for i, rid in enumerate(receiver_ids)}
    for isrc, source_id in enumerate(schedule.source_ids):
        for receiver_id, irecv in receiver_index.items():
            tr = index.get((source_id, receiver_id))
            if tr is not None:
                cube[isrc, irecv] = tr.samples
    return cube

def forward_blend(dataset: TraceSet, schedule: BlendingSchedule,
                  receiver_ids: tuple[str, ...] | None = None) -> BlendedData:
    """Simulate blended records from unblended single-shot data."""
    issues = check_unblended_input(schedule, dataset)
    fatal = [i for i in issues if i.code in {MISSING_SOURCE_RECORD, MISSING_RECEIVER,
                                             BAD_SAMPLE_COUNT, BAD_DT, NON_FINITE}]
    if fatal:
        details = '; '.join(i.detail for i in fatal)
        raise ValueError(f'cannot blend: {details}')
    if receiver_ids is None:
        receiver_ids = shared_receiver_layout(dataset)
    shot_cube = traceset_to_shot_cube(dataset, schedule, receiver_ids)
    source_index = {sid: i for i, sid in enumerate(schedule.source_ids)}
    blended_cube = forward_cube(shot_cube, schedule, source_index)
    records = {rec.record_id: blended_cube[irec]
               for irec, rec in enumerate(schedule.records)}
    return BlendedData(schedule, tuple(receiver_ids), records)

def pseudo_deblend(data: BlendedData) -> dict[str, np.ndarray]:
    """Adjoint stack of the blended records; contains un-suppressed crosstalk."""
    schedule = data.schedule
    blended_cube = np.stack([data.records[rid] for rid in schedule.record_ids])
    cube = adjoint_cube(blended_cube, schedule, len(schedule.source_ids))
    return {sid: cube[i] for i, sid in enumerate(schedule.source_ids)}
