"""Success-guided sampling of task configurations (Zhang et al., "A Balanced Data Diet",
CoRL 2026, arXiv:2610.12465).

A task owns a discrete pool of N configurations (start states, goals, terrain cells, ...).
The sampler keeps, per configuration, a window of the last H episode outcomes, and every reset
draws the next configuration from a distribution that favours the ones the policy currently
succeeds at about `target` of the time: a Beta-shaped kernel over the success estimate p_i,

    w_i = (p_i + eps)^(kappa t) * (1 - p_i + eps)^(kappa (1 - t)),
    P(i) = softmax(log(w_i + eps) / T).

Configurations the policy always fails or always solves get the floor weight, never zero, so
their estimates keep updating. Those the paper calls the "balanced data diet": neither too easy
nor too hard. The sampler is the whole method; PPO and the task are untouched, which is why it
fits VecTask's on_reset / on_done hooks without touching the base.

    sampler = SuccessGuidedSampler(n_cells, device, target=0.5, kappa=1.0)
    # on_reset(idx):  cell = sampler.sample(idx.numel());  place env idx in cell
    # on_done(idx):   sampler.record(cell[idx], success[idx])

`mode` also offers the baselines the paper compares against: "uniform" (ignore the estimates)
and "sfl" (Sampling for Learnability, w_i = p_i (1 - p_i)).
"""
import torch


class SuccessGuidedSampler:
    MODES = ("sgs", "uniform", "sfl")

    def __init__(self, n_cells, device, *, mode="sgs", target=0.5, kappa=1.0, temperature=2.0,
                 window=100, eps=1e-4, optimistic=True, generator=None):
        """
        n_cells      size of the configuration pool
        mode         "sgs" (the method), "uniform" or "sfl" (baselines)
        target       the success rate the kernel peaks at (paper: 0.66 locomotion, 0.5 manipulation)
        kappa        kernel concentration (paper: 5 locomotion, 1 manipulation); 0 = uniform
        temperature  softmax temperature over the log-weights (paper: 2)
        window       outcomes kept per configuration (paper: 100)
        eps          floor that keeps every configuration reachable
        optimistic   a configuration with no outcome yet counts as p = target (peak weight), so
                     unseen cells are tried before the estimates can starve them. The paper does
                     not say what it does here; with p = 0 an unseen cell would sit at the floor.
        generator    torch.Generator on `device` for the categorical draw
        """
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.n, self.device, self.mode = int(n_cells), torch.device(device), mode
        self.target, self.kappa, self.temperature = float(target), float(kappa), float(temperature)
        self.window, self.eps, self.optimistic = int(window), float(eps), bool(optimistic)
        self.g = generator
        self.outcomes = torch.zeros(self.n, self.window, device=self.device)
        self.count = torch.zeros(self.n, dtype=torch.long, device=self.device)   # total recorded
        self.head = torch.zeros(self.n, dtype=torch.long, device=self.device)    # next ring slot
        self._slots = torch.arange(self.window, device=self.device)

    # ---- estimates ----------------------------------------------------------------
    def success_rate(self):
        """Per-configuration mean of the recorded window, [N]; `target` (optimistic) or 0 for
        configurations without any outcome yet."""
        filled = (self._slots[None, :] < self.count[:, None]).float()
        n = filled.sum(1)
        p = (self.outcomes * filled).sum(1) / n.clamp_min(1.0)
        if self.optimistic:
            p = torch.where(n > 0, p, torch.full_like(p, self.target))
        return p

    def weights(self):
        p = self.success_rate()
        if self.mode == "uniform":
            return torch.ones_like(p)
        if self.mode == "sfl":
            return p * (1.0 - p)
        t, k, e = self.target, self.kappa, self.eps
        return (p + e) ** (k * t) * (1.0 - p + e) ** (k * (1.0 - t))

    def probs(self):
        """The sampling distribution over configurations, [N], sums to 1."""
        w = self.weights()
        if self.mode == "sfl":
            w = w.clamp_min(self.eps)
            return w / w.sum()
        if self.mode == "uniform":
            return w / w.sum()
        logits = torch.log(w + self.eps) / max(self.temperature, 1.0)
        return torch.softmax(logits, 0)

    # ---- the two hooks -----------------------------------------------------------
    def sample(self, n):
        """Draw n configuration indices i.i.d. from probs(), [n] long."""
        return torch.multinomial(self.probs(), n, replacement=True, generator=self.g)

    def record(self, cells, success):
        """Append one Boolean outcome per entry to the cells' windows. Duplicate cells within one
        call each get their own slot, in the order given."""
        cells = cells.to(self.device).long().reshape(-1)
        success = success.to(self.device).float().reshape(-1)
        if cells.numel() == 0:
            return
        # stable sort groups duplicates; rank within the group spreads them over consecutive slots
        order = torch.argsort(cells, stable=True)
        sc, ss = cells[order], success[order]
        first = torch.searchsorted(sc, sc, right=False)
        rank = torch.arange(sc.numel(), device=self.device) - first
        slot = (self.head[sc] + rank) % self.window
        self.outcomes[sc, slot] = ss
        per_cell = torch.bincount(cells, minlength=self.n)
        self.count += per_cell
        self.head = (self.head + per_cell) % self.window

    def state(self):
        return {"outcomes": self.outcomes.clone(), "count": self.count.clone(), "head": self.head.clone()}

    def load(self, st):
        self.outcomes.copy_(st["outcomes"]); self.count.copy_(st["count"]); self.head.copy_(st["head"])
