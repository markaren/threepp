"""Own runtime for the pretrained flyvis optic-lobe network (no flyvis import).

Per cell i, explicit forward Euler, every term at time t (flyvis PPNeuronIGRSynapses):

    tau_i = max(time_const_i, float32(dt))
    v_i  <- v_i + dt / tau_i * (-v_i + bias_i + sum_j W_ij relu(v_j) + x_i)

W is a sparse CSR matrix (row = post, col = pre, 1.5 M signed weights). x is non-zero
only on R1..R8, which all receive the same BoxEye value for their column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import MODEL_NPZ


class OpticLobe:
    def __init__(self, npz_path: str | Path = MODEL_NPZ, device="cpu", dtype=torch.float32):
        m = np.load(npz_path)
        self.device, self.dtype = torch.device(device), dtype
        self.type_names = [str(s) for s in m["type_names"]]
        self.n_nodes = len(m["cell_type"])
        if dtype == torch.float64:
            # rebuild from the shared tables so float64 is exact, not float32 widened
            ti, ci = m["edge_type_index"].astype(np.int64), m["edge_count_index"].astype(np.int64)
            w = m["edge_type_sign"].astype(np.float64)[ti] * np.exp(m["edge_count_log"].astype(np.float64))[ci]
            w = w * m["edge_type_syn_strength"].astype(np.float64)[ti]
        else:
            w = m["weight"]
        post, pre = m["post"].astype(np.int64), m["pre"].astype(np.int64)
        W = torch.sparse_coo_tensor(np.stack([post, pre]), torch.from_numpy(w), (self.n_nodes,) * 2)
        self.W = W.coalesce().to_sparse_csr().to(device=self.device, dtype=dtype)
        t = lambda a: torch.from_numpy(np.asarray(a)).to(self.device)
        self.time_const = t(m["time_const"]).to(dtype)  # raw; the clamp depends on dt
        self.bias = t(m["bias"]).to(dtype)
        self.receptor_index = t(m["receptor_node_index"].astype(np.int64))  # (8, 721)
        self.cell_type = t(m["cell_type"].astype(np.int64))
        self.u, self.v_coord = t(m["u"].astype(np.int64)), t(m["v"].astype(np.int64))
        self._x = torch.zeros(self.n_nodes, dtype=dtype, device=self.device)
        self._tau_dt, self._inv_tau = None, None
        self.v = None
        self.reset()

    # -- state -----------------------------------------------------------------------------
    def reset(self) -> torch.Tensor:
        """flyvis initial state when none is given: v = bias."""
        self.v = self.bias.clone()
        return self.v

    def _inv_tau_for(self, dt: float) -> torch.Tensor:
        if self._tau_dt != dt:
            dt32 = torch.tensor(dt, dtype=torch.float32).to(self.dtype)  # flyvis clamps with float32(dt)
            self._inv_tau = 1 / torch.maximum(self.time_const, dt32.to(self.device))
            self._tau_dt = dt
        return self._inv_tau

    def step(self, receptors: torch.Tensor, dt: float) -> torch.Tensor:
        """One Euler step with receptor input (721,) held for dt. Returns v (45669,)."""
        x = self._x
        x.zero_()
        x[self.receptor_index] = receptors.to(device=self.device, dtype=self.dtype)  # same value on R1..R8
        syn = self.W @ torch.relu(self.v)
        self.v = self.v + self._inv_tau_for(dt) * (-self.v + self.bias + syn + x) * dt
        return self.v

    def fade_in(self, receptors0: torch.Tensor, dt: float, t_fade: float = 1.0) -> torch.Tensor:
        """flyvis fade_in_state: from v = bias, int(t_fade/dt) steps ramping the contrast of
        the first frame up from grey 0.5: x_k = linspace(0, 1, n)[k] * (frame0 - 0.5) + 0.5."""
        self.reset()
        n = int(t_fade / dt)
        ramp = torch.linspace(0, 1, n, dtype=self.dtype, device=self.device)
        r0 = receptors0.to(device=self.device, dtype=self.dtype) - 0.5
        for k in range(n):
            self.step(ramp[k] * r0 + 0.5, dt)
        return self.v

    # -- readout ---------------------------------------------------------------------------
    def type_index(self, name: str) -> torch.Tensor:
        return torch.nonzero(self.cell_type == self.type_names.index(name))[:, 0]

    def by_type(self, name: str, v: torch.Tensor | None = None) -> torch.Tensor:
        """Columnar type -> (..., 721) in lattice (box_eye) order. v may be (N,) or (T, N)."""
        v = self.v if v is None else v
        return v[..., self.type_index(name)]

    def central(self, name: str, v: torch.Tensor | None = None) -> torch.Tensor:
        """Response of the central column (u = v = 0) of a type; scalar or (T,)."""
        v = self.v if v is None else v
        idx = self.type_index(name)
        c = idx[(self.u[idx] == 0) & (self.v_coord[idx] == 0)]
        return v[..., c[0]]
