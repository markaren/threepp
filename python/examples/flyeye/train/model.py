"""TrainableLobe: optic_lobe.OpticLobe with the flyvis free parameters as nn.Parameters.

Same dynamics, same connectome, same input convention. Per node i, explicit forward Euler,
every term at time t (flyvis PPNeuronIGRSynapses, reproduced in optic_lobe.py):

    tau_i  = max(time_const[type_i], float32(dt))
    v_i   <- v_i + dt / tau_i * (-v_i + bias[type_i] + sum_j W_ij relu(v_j) + x_i)

What is free (flyvis `flyvis/config/network/{node_config,edge_config}`):

| group           | shape | flyvis config                                      | here                |
|---|---|---|---|
| `time_const`    | 65    | `TimeConstant`, per type, requires_grad, no clamp  | learned as log tau  |
| `bias`          | 65    | `RestingPotential`, per type, requires_grad        | learned directly    |
| `syn_strength`  | 604   | `SynapseCountScaling`, per edge type, `non_negative` | raw, clamped at 0 |

What is fixed: `edge_type_sign` (604), `edge_count_log` (2355, log mean synapse count per
(type pair, du, dv)), the index tensors, the node `cell_type`, `receptor_node_index`.

`time_const` is learned as log tau so it cannot go negative; flyvis learns tau directly and
relies on the `max(tau, dt)` clamp, which we keep in the forward. `syn_strength` is the raw
parameter clamped with `clamp_min(0)` in the forward, which is what flyvis's
`Network.clamp()` does to the stored value after every optimizer step (`network.py:480-500`);
`train.py` also clamps the stored value so a checkpoint never holds a negative strength.
31 of the 604 strengths are already exactly 0 in member 000.

Per-edge weight, rebuilt once per forward pass over a window (the parameters only change
between optimizer steps):

    w_e = sign[ti_e] * exp(count_log[ci_e]) * clamp_min(strength, 0)[ti_e]

`sign[ti] * exp(count_log[ci])` is constant, so it is precomputed per edge; only the
multiply by `strength[ti]` is on the graph. The (post, pre) pairs of the shipped connectome
are unique (checked), so a permutation into row-major order gives a coalesced COO matrix
straight away, and a second permutation gives the transpose.

`torch.sparse.mm` does backpropagate into a sparse tensor's values, but measured on this
matrix (1.51 M non-zeros, batch 8, RTX 4070) it costs 25 ms against 0.35 ms for the forward,
which is 70x the whole rest of a step. `SpMM` below is the same product with the two
gradients written out, and costs about 1.5 ms:

    out    = A @ r                      (A row = post, col = pre)
    d r    = A^T @ d out
    d w_e  = sum_b d out[b, post_e] * r[b, pre_e]

The matrix values themselves are detached inside the product; the gradient reaches
`syn_strength` through `d w_e`, which is what `w_e = signed_count_e * strength[ti_e]` needs.

Gathered rates: `run_window` returns relu(v) at a fixed node index (8 x 721 for the T4/T5
readout, 34 x 721 for the flyvis output cell types), so no (T, B, 45669) tensor is ever kept.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .. import MODEL_NPZ
from ..readouts import MOTION_TYPES

# flyvis connectome `output_units` (fib25-fib19_v2.2.json), the 34 types its flow decoder reads.
OUTPUT_TYPES = ("T1", "T2", "T2a", "T3", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
                "Tm1", "Tm2", "Tm3", "Tm4", "Tm5Y", "Tm5a", "Tm5b", "Tm5c", "Tm9", "Tm16", "Tm20",
                "Tm28", "Tm30", "TmY3", "TmY4", "TmY5a", "TmY9", "TmY10", "TmY13", "TmY14",
                "TmY15", "TmY18")


class SpMM(torch.autograd.Function):
    """(B, N) x sparse (N, N)^T with the two gradients written out; see the module docstring.

    `values` carries the gradient for the edge weights. `A` and `At` are built from the same
    values detached, so the forward number is identical to `torch.sparse.mm(A, r.T).T`.
    """

    @staticmethod
    def forward(ctx, values, r, A, At, post, pre):
        ctx.save_for_backward(r)
        ctx.A, ctx.At, ctx.post, ctx.pre = A, At, post, pre
        return torch.sparse.mm(A, r.T).T

    @staticmethod
    def backward(ctx, g):
        (r,) = ctx.saved_tensors
        g = g.contiguous()
        gr = torch.sparse.mm(ctx.At, g.T).T if ctx.needs_input_grad[1] else None
        gv = (g[:, ctx.post] * r[:, ctx.pre]).sum(0) if ctx.needs_input_grad[0] else None
        return gv, gr, None, None, None, None


class TrainableLobe(nn.Module):
    def __init__(self, npz_path: str | Path = MODEL_NPZ, device="cuda", dtype=torch.float32,
                 train_tau=True, train_bias=True, train_strength=True):
        super().__init__()
        m = np.load(npz_path)
        self.src_npz = str(npz_path)
        self.type_names = [str(s) for s in m["type_names"]]
        self.n_types = len(self.type_names)
        self.n_nodes = len(m["cell_type"])
        cell_type = m["cell_type"].astype(np.int64)

        # per-node tau/bias -> per-type, asserting they are constant within a type
        tau_t = np.zeros(self.n_types, np.float64)
        bias_t = np.zeros(self.n_types, np.float64)
        for k in range(self.n_types):
            s = cell_type == k
            if not s.any():
                raise ValueError(f"cell type {k} ({self.type_names[k]}) has no nodes")
            for name, src, dst in (("time_const", m["time_const"], tau_t), ("bias", m["bias"], bias_t)):
                v = src[s]
                if v.max() != v.min():
                    raise ValueError(f"{name} is not constant within type {self.type_names[k]}: "
                                     f"{v.min()} .. {v.max()}")
                dst[k] = float(v[0])
        self.log_tau = nn.Parameter(torch.tensor(np.log(tau_t), dtype=dtype), requires_grad=train_tau)
        self.bias = nn.Parameter(torch.tensor(bias_t, dtype=dtype), requires_grad=train_bias)
        self.strength = nn.Parameter(torch.tensor(m["edge_type_syn_strength"], dtype=dtype),
                                     requires_grad=train_strength)

        ti = m["edge_type_index"].astype(np.int64)
        ci = m["edge_count_index"].astype(np.int64)
        post, pre = m["post"].astype(np.int64), m["pre"].astype(np.int64)
        signed_count = m["edge_type_sign"].astype(np.float64)[ti] * np.exp(m["edge_count_log"].astype(np.float64))[ci]
        order = np.lexsort((pre, post))  # row-major, unique (post, pre) -> already coalesced
        key = post * self.n_nodes + pre
        if len(np.unique(key)) != len(key):
            raise ValueError("duplicate (post, pre) edges: fold them before building the matrix")
        post_s, pre_s = post[order], pre[order]
        t_order = np.lexsort((post_s, pre_s))  # the same edges in the transpose's row-major order
        self.register_buffer("edge_type_index", torch.from_numpy(ti[order]))
        self.register_buffer("signed_count", torch.tensor(signed_count[order], dtype=dtype))
        self.register_buffer("mat_index", torch.from_numpy(np.stack([post_s, pre_s])))
        self.register_buffer("mat_index_t", torch.from_numpy(np.stack([pre_s[t_order], post_s[t_order]])))
        self.register_buffer("t_order", torch.from_numpy(t_order))
        self.register_buffer("edge_post", torch.from_numpy(post_s))
        self.register_buffer("edge_pre", torch.from_numpy(pre_s))
        self.register_buffer("node_type", torch.from_numpy(cell_type))
        self.register_buffer("receptor_index", torch.from_numpy(m["receptor_node_index"].astype(np.int64)))
        self.register_buffer("u", torch.from_numpy(m["u"].astype(np.int64)))
        self.register_buffer("v_coord", torch.from_numpy(m["v"].astype(np.int64)))
        self._dtype = dtype
        self.to(device)
        self.device = torch.device(device)
        # verification copies of the starting parameters, for the drift log
        self.register_buffer("tau0", torch.exp(self.log_tau.detach()).clone())
        self.register_buffer("bias0", self.bias.detach().clone())
        self.register_buffer("strength0", self.strength.detach().clone())

    # -- parameters ------------------------------------------------------------------------
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    def edge_weights(self) -> torch.Tensor:
        """(E,) per-edge signed weight in the matrix's row-major order."""
        return self.signed_count * self.strength.clamp_min(0)[self.edge_type_index]

    def matrix(self, values: torch.Tensor | None = None) -> torch.Tensor:
        """Sparse (N, N) COO weight matrix, row = post, col = pre."""
        v = self.edge_weights() if values is None else values
        return torch.sparse_coo_tensor(self.mat_index, v, (self.n_nodes,) * 2, is_coalesced=True)

    def matrices(self):
        """(values (E,), A, A^T) for one forward pass; A and A^T hold the values detached."""
        values = self.edge_weights()
        d = values.detach()
        A = torch.sparse_coo_tensor(self.mat_index, d, (self.n_nodes,) * 2, is_coalesced=True)
        At = torch.sparse_coo_tensor(self.mat_index_t, d[self.t_order], (self.n_nodes,) * 2, is_coalesced=True)
        return values, A, At

    def spmm(self, r: torch.Tensor, mats) -> torch.Tensor:
        values, A, At = mats
        return SpMM.apply(values, r, A, At, self.edge_post, self.edge_pre)

    def node_params(self, dt: float):
        """(inv_tau (N,), bias (N,)) with flyvis's float32(dt) clamp."""
        dt32 = torch.tensor(float(np.float32(dt)), dtype=self._dtype, device=self.device)
        inv_tau = 1 / torch.maximum(self.tau(), dt32)
        return inv_tau[self.node_type], self.bias[self.node_type]

    def type_index(self, name: str) -> torch.Tensor:
        return torch.nonzero(self.node_type == self.type_names.index(name))[:, 0]

    def gather_index(self, names=MOTION_TYPES) -> torch.Tensor:
        """(len(names), 721) node indices in lattice (box_eye) order."""
        return torch.stack([self.type_index(n) for n in names])

    # -- dynamics --------------------------------------------------------------------------
    def reset(self, batch: int | None = None) -> torch.Tensor:
        b = self.bias[self.node_type]
        return b.clone() if batch is None else b.expand(batch, -1).clone()

    def step(self, v, receptors, dt, mats, inv_tau, bias_n):
        """One Euler step. v (B, N), receptors (B, 721). Pure, returns the new v.

        `mats` is either the (values, A, A^T) triple from `matrices()` (the differentiable
        path) or a plain sparse matrix from `matrix()` (the no-grad path)."""
        x = torch.zeros_like(v)
        x[:, self.receptor_index] = receptors[:, None, :].expand(-1, self.receptor_index.shape[0], -1)
        r = torch.relu(v)
        syn = self.spmm(r, mats) if isinstance(mats, tuple) else torch.sparse.mm(mats, r.T).T
        return v + inv_tau * (-v + bias_n + syn + x) * dt

    @torch.no_grad()
    def fade_in(self, receptors0: torch.Tensor, dt: float, t_fade: float = 1.0) -> torch.Tensor:
        """flyvis fade_in_state on (B, 721): int(t_fade/dt) steps ramping the first frame's
        contrast up from grey 0.5, starting from v = bias. Never on the graph."""
        A, (inv_tau, bias_n) = self.matrix(), self.node_params(dt)
        inv_tau, bias_n = inv_tau.detach(), bias_n.detach()
        v = self.reset(len(receptors0)).detach()
        n = int(t_fade / dt)
        ramp = torch.linspace(0, 1, n, dtype=v.dtype, device=self.device)
        r0 = receptors0.to(device=self.device, dtype=v.dtype) - 0.5
        for k in range(n):
            v = self.step(v, ramp[k] * r0 + 0.5, dt, A, inv_tau, bias_n)
        return v

    def central_index(self) -> torch.Tensor:
        """(65,) the u = v = 0 node of every cell type, flyvis's `central_cells_index`."""
        out = []
        for k in range(self.n_types):
            idx = torch.nonzero(self.node_type == k)[:, 0]
            out.append(idx[(self.u[idx] == 0) & (self.v_coord[idx] == 0)][0])
        return torch.stack(out)

    def run_window(self, receptors: torch.Tensor, v0: torch.Tensor, dt: float,
                   index: torch.Tensor, mats=None, extra_index: torch.Tensor | None = None):
        """receptors (T, B, 721), v0 (B, N) -> (rates (T, B, K, 721), extra, v_T).

        rates = relu(v) gathered at `index` (K, 721) after each step, i.e. frame-aligned with
        the recordings (frame k is the state after stepping on frame k's receptors). `extra`
        is v itself gathered at `extra_index` (1-D), for the activity penalty, or None.
        """
        if mats is None:
            mats = self.matrices() if torch.is_grad_enabled() else self.matrix()
        inv_tau, bias_n = self.node_params(dt)
        v = v0
        out, ex = [], []
        for k in range(len(receptors)):
            v = self.step(v, receptors[k], dt, mats, inv_tau, bias_n)
            out.append(torch.relu(v)[:, index])
            if extra_index is not None:
                ex.append(v[:, extra_index])
        return torch.stack(out), (torch.stack(ex) if extra_index is not None else None), v

    @torch.no_grad()
    def rest_rates(self, index: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """(K, 721) relu(v) at uniform grey 0.5 with the current parameters: MotionField.rest."""
        grey = torch.full((1, index.shape[1]), 0.5, dtype=self._dtype, device=self.device)
        v = self.fade_in(grey, dt)
        return torch.relu(v)[0, index].clone()

    # -- export ----------------------------------------------------------------------------
    @torch.no_grad()
    def export_npz(self, path: str | Path) -> Path:
        """Write a checkpoint in data/flyeye_model.npz's format (per-node tau/bias, per-edge
        weight, plus the shared tables), so OpticLobe / members.py / score_members.py load it."""
        src = np.load(self.src_npz)
        out = {k: src[k] for k in src.files}
        tau_t = self.tau().detach().cpu().numpy().astype(np.float32)
        bias_t = self.bias.detach().cpu().numpy().astype(np.float32)
        strength = self.strength.detach().clamp_min(0).cpu().numpy().astype(np.float32)
        ct = src["cell_type"].astype(np.int64)
        out["time_const"] = tau_t[ct]
        out["bias"] = bias_t[ct]
        out["edge_type_syn_strength"] = strength
        ti = src["edge_type_index"].astype(np.int64)
        ci = src["edge_count_index"].astype(np.int64)
        w = (src["edge_type_sign"].astype(np.float64)[ti] * np.exp(src["edge_count_log"].astype(np.float64))[ci]
             * strength.astype(np.float64)[ti])
        out["weight"] = w.astype(np.float32)
        out["retrained_from"] = np.array(str(self.src_npz))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)
        return path

    # -- drift -----------------------------------------------------------------------------
    @torch.no_grad()
    def drift(self) -> dict:
        tau, bias, st = self.tau(), self.bias, self.strength.clamp_min(0)
        rel_tau = (tau - self.tau0) / self.tau0
        dst = st - self.strength0.clamp_min(0)
        return dict(
            tau_l2=float(rel_tau.norm()), tau_max_rel=float(rel_tau.abs().max()),
            bias_l2=float((bias - self.bias0).norm()), bias_max=float((bias - self.bias0).abs().max()),
            strength_l2=float(dst.norm()), strength_max=float(dst.abs().max()),
            strength_rel_l2=float(dst.norm() / max(float(self.strength0.norm()), 1e-12)),
            strength_zero=int((st == 0).sum()), strength_zero_start=int((self.strength0 <= 0).sum()),
        )

    @torch.no_grad()
    def drift_table(self, top=8) -> dict:
        """Which types' tau and bias moved most, which edge types' strengths moved most."""
        tau, bias, st = self.tau(), self.bias, self.strength.clamp_min(0)
        rel_tau = ((tau - self.tau0) / self.tau0).cpu().numpy()
        dbias = (bias - self.bias0).cpu().numpy()
        dst = (st - self.strength0.clamp_min(0)).cpu().numpy()
        st0 = self.strength0.clamp_min(0).cpu().numpy()
        ok = lambda a, n: [[self.type_names[i], round(float(a[i]), 5)] for i in np.argsort(-np.abs(a))[:n]]
        return dict(tau_rel=ok(rel_tau, top), bias=ok(dbias, top),
                    strength_abs=[[int(i), round(float(st0[i]), 5), round(float(st[i].item()), 5)]
                                  for i in np.argsort(-np.abs(dst))[:top]],
                    strength_newly_zero=int(((st0 > 0) & (st.cpu().numpy() == 0)).sum()))


def unit_field(rates: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
    """MotionField with unit gains from gathered T4/T5 rates.

    rates (..., 8, 721) relu(v) in MOTION_TYPES order, rest (8, 721) the grey-0.5 rate ->
    (..., 2, 721) with x = (T4b - T4a) + (T5b - T5a), y = (T4c - T4d) + (T5c - T5d).
    """
    r = rates - rest
    x = r[..., 1, :] - r[..., 0, :] + r[..., 5, :] - r[..., 4, :]
    y = r[..., 2, :] - r[..., 3, :] + r[..., 6, :] - r[..., 7, :]
    return torch.stack([x, y], dim=-2)
