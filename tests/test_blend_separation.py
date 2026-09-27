import numpy as np
import pytest
from geowave.blend.schedule import SweepEvent, BlendingSchedule
from geowave.blend.operator import BlendOperator, shift_gather, blend, pseudo_deblend
from geowave.blend.separate import SeparationConfig, separate
from geowave.blend.job import JobStatus, run_separation_job

DT = 0.004
N_SAMPLES = 2048
N_TRACES = 24

def ricker(freq=25.0, n=41, dt=DT):
    t = (np.arange(n) - (n - 1) / 2) * dt
    a = (np.pi * freq * t) ** 2
    return (1 - 2 * a) * np.exp(-a)

def make_shot(sx, n_reflectors=4, dt=DT, n_samples=N_SAMPLES, n_traces=N_TRACES):
    shot = np.zeros((n_traces, n_samples))
    wavelet = ricker()
    half = len(wavelet) // 2
    for k in range(n_reflectors):
        t0 = 0.3 + 0.25 * k
        vel = 1800.0 + 150.0 * k
        amp = 1.0 - 0.15 * k
        for itr in range(n_traces):
            offset = abs(itr * 25.0 - sx)
            idx = int(round(np.sqrt(t0 ** 2 + (offset / vel) ** 2) / dt))
            if half <= idx < n_samples - half:
                shot[itr, idx - half:idx + half + 1] += amp * wavelet
    return shot

@pytest.fixture(scope='module')
def schedule():
    return BlendingSchedule.generate('rec-1', ['s1', 's2', 's3'], DT, N_SAMPLES,
                                     seed=42, n_sweeps=6, slot_interval=0.3,
                                     dither_max=0.2)

@pytest.fixture(scope='module')
def shots():
    return {'s1': make_shot(100.0), 's2': make_shot(300.0), 's3': make_shot(500.0)}

def test_schedule_reproducible():
    kw = dict(dt=DT, n_samples=512, n_sweeps=4, slot_interval=0.3, dither_max=0.2)
    a = BlendingSchedule.generate('r', ['s1', 's2'], seed=7, **kw)
    b = BlendingSchedule.generate('r', ['s1', 's2'], seed=7, **kw)
    c = BlendingSchedule.generate('r', ['s1', 's2'], seed=8, **kw)
    assert a.sweep_times == b.sweep_times
    assert a.sweep_times != c.sweep_times
    assert len(a.sweeps) == 8
    assert a.source_ids == ['s1', 's2']

def test_schedule_json_roundtrip(tmp_path, schedule):
    clone = BlendingSchedule.from_dict(schedule.to_dict())
    assert clone == schedule
    path = tmp_path / 'sched.json'
    schedule.save(str(path))
    assert BlendingSchedule.load(str(path)) == schedule

def test_collision_detection():
    sweeps = (SweepEvent('s1', 0, 0.0, 0.10), SweepEvent('s2', 0, 0.5, 0.0),
              SweepEvent('s3', 0, 0.5, 0.03))
    sched = BlendingSchedule('r', DT, 512, sweeps, seed=0)
    hits = sched.collisions(min_gap=0.05)
    assert len(hits) == 1
    a, b, gap = hits[0]
    assert {a.source_id, b.source_id} == {'s2', 's3'}
    assert gap == pytest.approx(0.03)
    assert sched.collisions(min_gap=0.001) == []

def test_shift_roundtrip():
    x = make_shot(200.0)
    y = shift_gather(shift_gather(x, 0.137, DT), -0.137, DT)
    assert np.linalg.norm(y - x) / np.linalg.norm(x) < 1e-10

def test_operator_adjoint(schedule, shots):
    op = BlendOperator(schedule, shots['s1'].shape)
    rng = np.random.default_rng(0)
    est = {sid: rng.standard_normal(shots['s1'].shape) for sid in schedule.source_ids}
    rec = rng.standard_normal(shots['s1'].shape)
    lhs = float(np.sum(op.apply(est) * rec))
    rhs = float(sum(np.sum(est[s] * op.adjoint(rec)[s]) for s in est))
    assert lhs == pytest.approx(rhs, rel=1e-10)

def test_operator_matches_naive_shifts(schedule, shots):
    op = BlendOperator(schedule, shots['s1'].shape)
    naive = sum(shift_gather(shots[sid], t, DT)
                for sid, ts in schedule.sweep_times.items() for t in ts)
    assert np.allclose(op.apply(shots), naive)

def test_lipschitz_bound(schedule, shots):
    op = BlendOperator(schedule, shots['s1'].shape)
    lam = op.lipschitz()
    rng = np.random.default_rng(1)
    v = {sid: rng.standard_normal(shots['s1'].shape) for sid in schedule.source_ids}
    num = float(sum(np.sum(op.adjoint(op.apply(v))[s] * v[s]) for s in v))
    den = float(sum(np.sum(a * a) for a in v.values()))
    assert num / den <= lam * 1.05

def test_job_collision_status(shots):
    sweeps = (SweepEvent('s1', 0, 0.0, 0.0), SweepEvent('s2', 0, 0.0, 0.001))
    sched = BlendingSchedule('rec-c', DT, N_SAMPLES, sweeps, seed=1)
    result = run_separation_job({'s1': shots['s1'], 's2': shots['s2']}, sched)
    assert result.status is JobStatus.SCHEDULE_COLLISION
    assert result.estimates is None
    assert result.diagnostics['collisions'][0]['gap_s'] == pytest.approx(0.001)

def test_job_missing_source_status(schedule, shots):
    result = run_separation_job({'s1': shots['s1'], 's3': shots['s3']}, schedule)
    assert result.status is JobStatus.MISSING_SOURCE_RECORD
    assert result.diagnostics['missing_sources'] == ['s2']
    assert result.estimates is None

@pytest.fixture(scope='module')
def job_result(schedule, shots):
    return run_separation_job(shots, schedule, truth=shots)

def test_separation_converged(job_result):
    assert job_result.status is JobStatus.OK
    assert job_result.final_residual < 1e-2
    assert job_result.residual_history[-1] < job_result.residual_history[0]

def test_separation_quality(job_result):
    for sid, err in job_result.truth_error.items():
        assert err < 0.1, f'{sid} error {err}'
    for sid, leak in job_result.crosstalk.items():
        assert leak < 0.1, f'{sid} leakage {leak}'

def test_not_converged_status(schedule, shots):
    cfg = SeparationConfig(max_iter=2, tol=1e-9)
    result = run_separation_job(shots, schedule, config=cfg, truth=shots)
    assert result.status is JobStatus.NOT_CONVERGED
    assert result.estimates is not None
    assert len(result.residual_history) == 2
    assert result.diagnostics['final_relative_residual'] > cfg.tol
    assert set(result.crosstalk) == {'s1', 's2', 's3'}
    assert all(np.isfinite(v) for v in result.crosstalk.values())

def test_stagnation_flag(schedule, shots):
    cfg = SeparationConfig(max_iter=40, tol=1e-12, threshold_floor=0.5,
                           stagnation_window=8)
    result = run_separation_job(shots, schedule, config=cfg)
    assert result.status is JobStatus.NOT_CONVERGED
    assert result.diagnostics['stagnated'] is True

def test_job_deterministic(schedule, shots):
    cfg = SeparationConfig(max_iter=5)
    r1 = run_separation_job(shots, schedule, config=cfg)
    r2 = run_separation_job(shots, schedule, config=cfg)
    assert r1.status == r2.status
    assert r1.residual_history == r2.residual_history
    for sid in r1.estimates:
        assert np.array_equal(r1.estimates[sid], r2.estimates[sid])
