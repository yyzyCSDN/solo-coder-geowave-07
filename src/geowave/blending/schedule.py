"""Reproducible slip-sweep (simultaneous source / blended acquisition) schedules.

A schedule records, for every blended record, which sources fired and at what
integer-sample random delay relative to the record start. Delays are expressed
in samples so that replaying a schedule never touches an interpolation kernel
and two runs with the same seed reproduce bit-identical blended data.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import numpy as np

# Schedule validation issue codes.
COLLISION = 'collision'
DUPLICATE_SOURCE = 'duplicate-source'
MISSING_ACTIVATION = 'missing-activation'
BAD_SOURCE_REF = 'bad-source-ref'
DELAY_RANGE = 'delay-range'
EMPTY_RECORD = 'empty-record'

@dataclass(frozen=True)
class SourceActivation:
    source_id: str
    delay_samples: int

@dataclass(frozen=True)
class BlendedRecordSpec:
    record_id: str
    activations: tuple[SourceActivation, ...]

@dataclass(frozen=True)
class BlendingSchedule:
    schedule_id: str
    dt_s: float
    record_samples: int
    source_ids: tuple[str, ...]
    records: tuple[BlendedRecordSpec, ...]
    seed: int | None = None
    generator: str = ''
    min_delay_separation_samples: int = 0
    max_delay_samples: int = 0
    n_sweeps: int = 1

    def __post_init__(self):
        if self.dt_s <= 0:
            raise ValueError('sample interval must be positive')
        if self.record_samples <= 0:
            raise ValueError('record length must be positive')
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError('duplicate source identifiers')
        if len(self.records) != len({r.record_id for r in self.records}):
            raise ValueError('duplicate record identifiers')
        if not self.records:
            raise ValueError('schedule has no records')

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(r.record_id for r in self.records)

    def activation_index(self) -> dict[str, list[SourceActivation]]:
        return {r.record_id: list(r.activations) for r in self.records}

    def sources_per_record(self) -> dict[str, int]:
        return {r.record_id: len(r.activations) for r in self.records}

    def validate(self) -> list['ScheduleIssue']:
        """Return every schedule defect; empty list means the schedule is safe."""
        issues: list[ScheduleIssue] = []
        known_sources = set(self.source_ids)
        activated: set[str] = set()
        for rec in self.records:
            seen_in_record: set[str] = set()
            if not rec.activations:
                issues.append(ScheduleIssue(EMPTY_RECORD, rec.record_id, 'record has no source activation'))
            delays: list[tuple[str, int]] = []
            for act in rec.activations:
                if act.source_id not in known_sources:
                    issues.append(ScheduleIssue(BAD_SOURCE_REF, rec.record_id,
                                                f'unknown source {act.source_id}', act.source_id))
                if act.source_id in seen_in_record:
                    issues.append(ScheduleIssue(DUPLICATE_SOURCE, rec.record_id,
                                                f'source {act.source_id} fires twice in record', act.source_id))
                seen_in_record.add(act.source_id)
                activated.add(act.source_id)
                if not 0 <= act.delay_samples <= self.max_delay_samples:
                    issues.append(ScheduleIssue(DELAY_RANGE, rec.record_id,
                                                f'delay {act.delay_samples} outside [0,{self.max_delay_samples}]',
                                                act.source_id))
                delays.append((act.source_id, act.delay_samples))
            if self.min_delay_separation_samples > 0:
                delays.sort(key=lambda item: item[1])
                for (s0, d0), (s1, d1) in zip(delays, delays[1:]):
                    if d1 - d0 < self.min_delay_separation_samples:
                        issues.append(ScheduleIssue(COLLISION, rec.record_id,
                                                    f'sources {s0}@{d0} and {s1}@{d1} are closer than '
                                                    f'{self.min_delay_separation_samples} samples', f'{s0}|{s1}'))
        for source_id in self.source_ids:
            if source_id not in activated:
                issues.append(ScheduleIssue(MISSING_ACTIVATION, '',
                                            f'source {source_id} never activated', source_id))
        return issues

    def to_dict(self) -> dict:
        return {
            'schedule_id': self.schedule_id,
            'dt_s': self.dt_s,
            'record_samples': self.record_samples,
            'source_ids': list(self.source_ids),
            'seed': self.seed,
            'generator': self.generator,
            'min_delay_separation_samples': self.min_delay_separation_samples,
            'max_delay_samples': self.max_delay_samples,
            'n_sweeps': self.n_sweeps,
            'records': [
                {'record_id': r.record_id,
                 'activations': [{'source_id': a.source_id, 'delay_samples': a.delay_samples}
                                 for a in r.activations]}
                for r in self.records
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'BlendingSchedule':
        records = tuple(
            BlendedRecordSpec(r['record_id'],
                              tuple(SourceActivation(a['source_id'], int(a['delay_samples']))
                                    for a in r['activations']))
            for r in data['records'])
        return cls(data['schedule_id'], float(data['dt_s']), int(data['record_samples']),
                   tuple(data['source_ids']), records,
                   seed=data.get('seed'), generator=data.get('generator', ''),
                   min_delay_separation_samples=int(data.get('min_delay_separation_samples', 0)),
                   max_delay_samples=int(data.get('max_delay_samples', 0)),
                   n_sweeps=int(data.get('n_sweeps', 1)))

    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(raw).hexdigest()

@dataclass(frozen=True)
class ScheduleIssue:
    code: str
    record_id: str
    detail: str
    source_id: str = ''

def random_blending_schedule(
        schedule_id: str,
        source_ids: tuple[str, ...] | list[str],
        dt_s: float,
        record_samples: int,
        sources_per_record: int,
        seed: int,
        max_delay_samples: int,
        min_delay_separation_samples: int = 0,
        n_sweeps: int = 1,
        record_ids: tuple[str, ...] | list[str] | None = None) -> BlendingSchedule:
    """Build a reproducible random slip-sweep schedule.

    Sources are shuffled deterministically from ``seed`` and laid out over
    consecutive records, ``sources_per_record`` at a time; one source is
    allowed to appear only once per record. Random integer firing delays are
    drawn in ``[0, max_delay_samples]`` and rejected/resampled until they
    honour ``min_delay_separation_samples``.

    ``n_sweeps`` replays the same source groups in independent sweeps, each
    with freshly drawn (but seed-determined) delays. Repeated dithering is
    what makes slip-sweep crosstalk incoherent and therefore separable; the
    total record count is therefore
    ``ceil(len(sources)/sources_per_record) * n_sweeps``.
    """
    source_ids = tuple(source_ids)
    if not source_ids:
        raise ValueError('need at least one source')
    if sources_per_record <= 0:
        raise ValueError('sources_per_record must be positive')
    if sources_per_record > len(source_ids):
        raise ValueError('sources_per_record exceeds number of sources')
    if n_sweeps < 1:
        raise ValueError('n_sweeps must be >= 1')
    if max_delay_samples < 0:
        raise ValueError('max_delay_samples must be non-negative')
    if min_delay_separation_samples * (sources_per_record - 1) > max_delay_samples:
        raise ValueError('delay window too narrow to honour minimum separation')
    n_groups = -(-len(source_ids) // sources_per_record)
    n_records = n_groups * n_sweeps
    if record_ids is None:
        record_ids = tuple(f'b{i:03d}' for i in range(n_records))
    record_ids = tuple(record_ids)
    if len(record_ids) != n_records:
        raise ValueError('record id count does not match required record count')

    rng = np.random.default_rng(seed)
    order = list(source_ids)
    rng.shuffle(order)
    groups = [order[g * sources_per_record:(g + 1) * sources_per_record]
              for g in range(n_groups)]
    records: list[BlendedRecordSpec] = []
    irec = 0
    for sweep in range(n_sweeps):
        for group in groups:
            delays: list[int] = []
            for _source in group:
                for _attempt in range(10_000):
                    candidate = int(rng.integers(0, max_delay_samples + 1))
                    if all(abs(candidate - d) >= min_delay_separation_samples
                           for d in delays):
                        delays.append(candidate)
                        break
                else:  # pragma: no cover - guarded by the feasibility check above
                    raise RuntimeError('could not draw separated delays')
            records.append(BlendedRecordSpec(
                record_ids[irec],
                tuple(SourceActivation(s, d) for s, d in zip(group, delays))))
            irec += 1
    return BlendingSchedule(
        schedule_id, dt_s, record_samples, source_ids, tuple(records),
        seed=int(seed), generator='numpy.random.default_rng',
        min_delay_separation_samples=min_delay_separation_samples,
        max_delay_samples=max_delay_samples, n_sweeps=n_sweeps)
