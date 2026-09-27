import numpy as np
import pytest
from geowave.model.traces import Trace, TraceHeader, TraceSet
from geowave.blending import (
    BlendingSchedule, BlendedRecordSpec, SourceActivation,
    random_blending_schedule,
    BlendedData,
    forward_blend, forward_cube, adjoint_cube, pseudo_deblend,
    check_unblended_input,
    SeparationStatus, SeparationConfig, separate,
    COLLISION, MISSING_SOURCE_RECORD, MISSING_RECORD, BAD_SAMPLE_COUNT,
)

DT = 0.004
N = 256
N_SRC = 48
N_RECV = 6
BLEND_FACTOR = 3
N_SWEEPS = 2
MAX_DELAY = 64
SOURCES = tuple(f's{i}' for i in range(N_SRC))
RECEIVERS = tuple(f'r{i}' for i in range(N_RECV))


def ricker(t, f):
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def make_shots():
    """Two coherent dipping Ricker events per shot, shared receiver layout."""
    t = np.arange(N) * DT
    traces = []
    panels = {}
    for s in range(N_SRC):
        sid = f's{s}'
        panel = np.zeros((N_RECV, N))
        sx = s * 25.0
        for j in range(N_RECV):
            off = j * 25.0 - sx
            sig = np.zeros(N)
            for vel, t0, amp, f0 in ((2200.0, 0.75, 0.90, 22.0),
                                     (1500.0, 0.85, 0.50, 28.0)):
                arr = off / vel + t0
                if 0 < arr < N * DT:
                    sig += amp * ricker(t - arr, f0)
            panel[j] = sig
            traces.append(Trace(
                TraceHeader(f'{sid}-r{j}', sid, f'r{j}', sx, 0.0,
                            j * 25.0, 0.0, DT), sig))
        panels[sid] = panel
    return TraceSet(traces, 'shots'), panels


@pytest.fixture
def shots():
    return make_shots()[0]


@pytest.fixture
def schedule():
    return random_blending_schedule(
        'sched-1', SOURCES, DT, N, BLEND_FACTOR, seed=20260926,
        max_delay_samples=MAX_DELAY, min_delay_separation_samples=8,
        n_sweeps=N_SWEEPS)


def test_schedule_is_reproducible(schedule):
    again = random_blending_schedule(
        'sched-1', SOURCES, DT, N, BLEND_FACTOR, seed=20260926,
        max_delay_samples=MAX_DELAY, min_delay_separation_samples=8,
        n_sweeps=N_SWEEPS)
    assert again.to_dict() == schedule.to_dict()
    assert again.fingerprint() == schedule.fingerprint()
    assert schedule.validate() == []
    # 48 sources / 3 per record, replayed over 2 sweeps -> 32 records.
    assert len(schedule.records) == 32
    assert schedule.n_sweeps == 2
    fired = {sid: 0 for sid in SOURCES}
    for rec in schedule.records:
        assert len(rec.activations) == BLEND_FACTOR
        for act in rec.activations:
            fired[act.source_id] += 1
            assert 0 <= act.delay_samples <= MAX_DELAY
    assert all(count == N_SWEEPS for count in fired.values())

    other = random_blending_schedule(
        'sched-1', SOURCES, DT, N, BLEND_FACTOR, seed=7,
        max_delay_samples=MAX_DELAY, min_delay_separation_samples=8,
        n_sweeps=N_SWEEPS)
    assert other.fingerprint() != schedule.fingerprint()


def test_schedule_roundtrip(schedule):
    restored = BlendingSchedule.from_dict(schedule.to_dict())
    assert restored.to_dict() == schedule.to_dict()
    assert restored.fingerprint() == schedule.fingerprint()
    assert restored.n_sweeps == N_SWEEPS


def test_schedule_collision_is_flagged():
    collision = BlendingSchedule(
        'bad', DT, N, ('s0', 's1', 's2'), (
            BlendedRecordSpec('b0', (SourceActivation('s0', 10),
                                     SourceActivation('s1', 12))),
            BlendedRecordSpec('b1', (SourceActivation('s2', 4),)),
        ),
        min_delay_separation_samples=8, max_delay_samples=40)
    issues = collision.validate()
    assert COLLISION in [i.code for i in issues]
    assert issues[0].record_id == 'b0'


def test_schedule_duplicate_and_missing_activation():
    dup = BlendingSchedule(
        'dup', DT, N, ('s0', 's1'), (
            BlendedRecordSpec('b0', (SourceActivation('s0', 0),
                                     SourceActivation('s0', 5))),
        ), max_delay_samples=10)
    codes = {i.code for i in dup.validate()}
    assert 'duplicate-source' in codes
    assert 'missing-activation' in codes


def test_forward_and_adjoint_are_adjoints(shots, schedule):
    blended = forward_blend(shots, schedule)
    n_total = N + MAX_DELAY
    assert all(cube.shape == (N_RECV, n_total)
               for cube in blended.records.values())

    stacked = np.stack([blended.records[rid] for rid in schedule.record_ids])
    assert adjoint_cube(stacked, schedule, N_SRC).shape == (N_SRC, N_RECV, N)

    rng = np.random.default_rng(1)
    x = rng.standard_normal((N_SRC, N_RECV, N))
    y = rng.standard_normal((len(schedule.records), N_RECV, n_total))
    source_index = {sid: i for i, sid in enumerate(SOURCES)}
    lhs = float(np.sum(forward_cube(x, schedule, source_index) * y))
    rhs = float(np.sum(x * adjoint_cube(y, schedule, N_SRC)))
    assert abs(lhs - rhs) < 1e-8 * max(abs(lhs), 1.0)

    pseudo_map = pseudo_deblend(blended)
    assert set(pseudo_map) == set(SOURCES)


def test_separation_converges_and_reproduces(shots, schedule):
    _, panels = make_shots()
    blended = forward_blend(shots, schedule)
    cfg = SeparationConfig(max_iterations=80, residual_tol=0.2,
                           threshold_end_fraction=0.05)
    result = separate(blended, cfg, truth=panels)

    assert result.status is SeparationStatus.OK_CONVERGED, result.reason
    assert result.converged
    residual = result.residual_history
    assert residual[-1] <= cfg.residual_tol
    assert residual[-1] < residual[0]

    # Quality: separated shots are far closer to truth than pseudo-deblend.
    pseudo_map = pseudo_deblend(blended)
    pseudo_err = np.mean([
        np.linalg.norm(pseudo_map[sid] - panels[sid])
        / np.linalg.norm(panels[sid]) for sid in SOURCES])
    sep_err = np.mean(list(result.crosstalk.truth_relative_error.values()))
    assert sep_err < 0.5 * pseudo_err
    assert sep_err < 0.4
    for sid in (SOURCES[0], SOURCES[N_SRC // 2], SOURCES[-1]):
        recovered = result.shots[sid].ravel()
        true = panels[sid].ravel()
        corr = np.dot(recovered, true) / (
            np.linalg.norm(recovered) * np.linalg.norm(true))
        assert corr > 0.9

    # Crosstalk report is populated per source and bounded.
    assert set(result.crosstalk.per_source_ratio) == set(SOURCES)
    assert 0.0 <= result.crosstalk.mean_ratio
    assert len(result.iterations) == len(residual)

    # Same schedule + data + config reproduce the run bit-for-bit.
    again = separate(blended, cfg, truth=panels)
    assert again.residual_history == result.residual_history
    for sid in SOURCES:
        assert np.array_equal(again.shots[sid], result.shots[sid])

    traceset = result.deblended_traceset()
    assert len(traceset.traces) == N_SRC * N_RECV
    assert set(traceset.by_shot()) == set(SOURCES)


def test_missing_source_record_detected(shots, schedule):
    assert check_unblended_input(schedule, shots) == []

    reduced = TraceSet([tr for tr in shots.traces if tr.header.shot_id != 's0'],
                       'shots')
    issues = check_unblended_input(schedule, reduced)
    assert any(i.code == MISSING_SOURCE_RECORD and i.source_id == 's0'
               for i in issues)

    partial = TraceSet(
        [tr for tr in shots.traces
         if not (tr.header.shot_id == 's1' and tr.header.receiver_id == 'r0')],
        'shots')
    issues = check_unblended_input(schedule, partial)
    assert any(i.source_id == 's1' and 'r0' in i.detail for i in issues)


def test_missing_blended_record_gives_failure_status(schedule, shots):
    blended = forward_blend(shots, schedule)
    del blended.records[schedule.records[0].record_id]
    result = separate(blended)
    assert result.status is SeparationStatus.MISSING_RECORD
    assert result.iterations == []
    assert result.shots == {}
    assert any(i.code == MISSING_RECORD for i in result.input_issues)


def test_collision_schedule_short_circuits(shots, schedule):
    n_groups = -(-N_SRC // BLEND_FACTOR)
    records = []
    for sweep in range(N_SWEEPS):
        for g in range(n_groups):
            chunk = SOURCES[g * BLEND_FACTOR:(g + 1) * BLEND_FACTOR]
            records.append(BlendedRecordSpec(
                f'b{sweep}_{g}',
                tuple(SourceActivation(sid, 3 * i)
                      for i, sid in enumerate(chunk))))
    bad = BlendingSchedule(
        'bad', DT, N, SOURCES, tuple(records),
        min_delay_separation_samples=16, max_delay_samples=MAX_DELAY)
    good = forward_blend(shots, schedule)
    data = BlendedData(bad, RECEIVERS, dict(good.records))
    result = separate(data)
    assert result.status is SeparationStatus.SCHEDULE_COLLISION
    assert any(i.code == COLLISION for i in result.schedule_issues)
    assert result.crosstalk is None
    assert result.iterations == []


def test_non_convergence_reports_residual_and_crosstalk(shots, schedule):
    blended = forward_blend(shots, schedule)
    cfg = SeparationConfig(max_iterations=2, residual_tol=1e-9,
                           stall_iterations=100)
    result = separate(blended, cfg)
    assert result.status is SeparationStatus.NOT_CONVERGED
    assert len(result.iterations) == 2
    assert result.crosstalk is not None
    assert set(result.crosstalk.per_source_ratio) == set(SOURCES)
    assert result.residual_history[-1] > cfg.residual_tol
    assert '1e-09' in result.reason or 'tolerance' in result.reason


def test_stall_is_reported(shots, schedule):
    # Constant high threshold: the residual quickly reaches its floor and
    # makes no further progress, so the stall detector must fire.
    blended = forward_blend(shots, schedule)
    cfg = SeparationConfig(max_iterations=30, residual_tol=1e-9,
                           threshold_start_fraction=0.9,
                           threshold_end_fraction=0.9,
                           stall_iterations=4, stall_min_progress=0.055)
    result = separate(blended, cfg)
    assert result.status is SeparationStatus.NOT_CONVERGED
    assert 'stalled' in result.reason
    assert len(result.iterations) <= 30


def test_bad_sample_count_input(shots, schedule):
    blended = forward_blend(shots, schedule)
    rid = schedule.record_ids[0]
    blended.records[rid] = blended.records[rid][:, :-3]
    result = separate(blended)
    assert result.status is SeparationStatus.INVALID_INPUT
    assert any(i.code == BAD_SAMPLE_COUNT for i in result.input_issues)


def test_non_finite_input_is_rejected(shots, schedule):
    blended = forward_blend(shots, schedule)
    rid = schedule.record_ids[0]
    bad_cube = blended.records[rid].copy()
    bad_cube[0, 10] = np.nan
    blended.records[rid] = bad_cube
    result = separate(blended)
    assert result.status is SeparationStatus.INVALID_INPUT
    assert any(i.code == 'non-finite' for i in result.input_issues)
