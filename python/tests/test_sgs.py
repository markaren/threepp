"""SuccessGuidedSampler: the ring buffer, the kernel and the baselines, on synthetic outcomes."""
import pytest

torch = pytest.importorskip("torch")

from threepp.rl.sgs import SuccessGuidedSampler  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _fill(s, rates, n=200, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    rates = torch.tensor(rates, device=DEV)
    for _ in range(n):
        cells = torch.arange(len(rates), device=DEV)
        s.record(cells, torch.rand(len(rates), device=DEV, generator=g) < rates)


def test_window_tracks_recent_outcomes():
    s = SuccessGuidedSampler(3, DEV, window=10)
    cells = torch.tensor([0, 0, 0, 1], device=DEV)
    s.record(cells, torch.tensor([1, 0, 1, 1], device=DEV, dtype=torch.bool))
    p = s.success_rate()
    assert p[0].item() == pytest.approx(2 / 3)
    assert p[1].item() == 1.0
    assert p[2].item() == s.target                       # unseen: optimistic
    # 20 failures push the window past its 10 slots: only the last 10 count
    s.record(torch.zeros(20, dtype=torch.long, device=DEV), torch.zeros(20, dtype=torch.bool, device=DEV))
    assert s.success_rate()[0].item() == 0.0
    assert s.count[0].item() == 23 and s.head[0].item() == 23 % 10


def test_duplicates_in_one_call_each_get_a_slot():
    s = SuccessGuidedSampler(2, DEV, window=8)
    cells = torch.tensor([1, 0, 1, 1, 0, 1], device=DEV)
    succ = torch.tensor([1, 0, 1, 0, 1, 1], device=DEV, dtype=torch.bool)
    s.record(cells, succ)
    assert s.success_rate()[1].item() == pytest.approx(3 / 4)
    assert s.success_rate()[0].item() == pytest.approx(1 / 2)
    assert s.head.tolist() == [2, 4]


def test_kernel_peaks_at_target_and_floors_the_extremes():
    s = SuccessGuidedSampler(5, DEV, target=0.5, kappa=1.0)
    _fill(s, [0.0, 0.25, 0.5, 0.75, 1.0])
    pr = s.probs()
    assert pr.argmax().item() == 2
    assert pr[0] < pr[1] < pr[2] and pr[4] < pr[3] < pr[2]
    assert (pr > 0).all()                               # the floor keeps every cell reachable
    assert pr.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_target_shifts_the_peak():
    s = SuccessGuidedSampler(11, DEV, target=0.66, kappa=5.0)
    _fill(s, [i / 10 for i in range(11)])
    assert s.probs().argmax().item() in (6, 7)


def test_baselines():
    u = SuccessGuidedSampler(4, DEV, mode="uniform")
    _fill(u, [0.0, 0.3, 0.6, 1.0])
    assert torch.allclose(u.probs(), torch.full((4,), 0.25, device=DEV))
    f = SuccessGuidedSampler(4, DEV, mode="sfl")
    _fill(f, [0.0, 0.3, 0.6, 1.0])
    pr = f.probs()
    assert pr.argmax().item() == 2 and pr[0].item() == pytest.approx(pr[3].item())


def test_sample_follows_probs():
    s = SuccessGuidedSampler(3, DEV, target=0.5, kappa=1.0,
                             generator=torch.Generator(device=DEV).manual_seed(1))
    _fill(s, [0.0, 0.5, 1.0])
    draws = torch.bincount(s.sample(20000), minlength=3).float() / 20000
    assert torch.allclose(draws, s.probs(), atol=0.02)
