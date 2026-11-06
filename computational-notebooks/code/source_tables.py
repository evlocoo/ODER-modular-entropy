"""Subset-information calculations for the finite Hayden–Preskill source."""
from __future__ import annotations
import numpy as np
from hp_source import finite_hp_state, pure_subsystem_entropy

def mask_axes(mask: int, axes: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(axis for index, axis in enumerate(axes) if mask & (1 << index))


def pure_entropy_fast(state: np.ndarray, keep_axes: tuple[int, ...]) -> float:
    """Use the smaller side of a pure-state bipartition."""
    keep = tuple(sorted(keep_axes))
    if len(keep) > state.ndim / 2:
        keep = tuple(index for index in range(state.ndim) if index not in keep)
    return pure_subsystem_entropy(state, keep)


def hp_joint_information_tensor(
    seed: int,
    *,
    return_entropies: bool = False,
) -> tuple[np.ndarray, dict[str, float], np.ndarray | None, np.ndarray | None]:
    """J[D_mask,E_mask]=I(Q:D_mask E_mask)/(2S(Q))."""
    state, diagnostics = finite_hp_state(seed, black_hole_qubits=5, emitted_qubits=5)
    q_axis = 0
    d_axes = tuple(range(2, 7))
    e_axes = tuple(range(7, 12))
    s_q = pure_entropy_fast(state, (q_axis,))
    information = np.zeros((32, 32), dtype=float)
    sx = np.zeros_like(information) if return_entropies else None
    sqx = np.zeros_like(information) if return_entropies else None
    for d_mask in range(32):
        d_keep = mask_axes(d_mask, d_axes)
        for e_mask in range(32):
            x_keep = d_keep + mask_axes(e_mask, e_axes)
            s_x = pure_entropy_fast(state, x_keep) if x_keep else 0.0
            s_qx = pure_entropy_fast(state, (q_axis,) + x_keep)
            value = (s_q + s_x - s_qx) / (2 * s_q)
            information[d_mask, e_mask] = float(np.clip(value, 0.0, 1.0))
            if return_entropies:
                sx[d_mask, e_mask] = s_x
                sqx[d_mask, e_mask] = s_qx
    diagnostics = {
        **{key: float(value) for key, value in diagnostics.items() if isinstance(value, (int, float))},
        "S_Q_bits": float(s_q),
        "minimum_information": float(np.min(information)),
        "maximum_information": float(np.max(information)),
    }
    return information, diagnostics, sx, sqx



def generate_source_tables(first_seed=2026082100, count=32):
    tables=[]; diagnostics=[]
    for i in range(count):
        table, diagnostic, sx, sqx = hp_joint_information_tensor(first_seed+i, return_entropies=i==0)
        tables.append(table); diagnostics.append(diagnostic)
        if i==0: first_sx, first_sqx = sx, sqx
    return np.stack(tables), first_sx, first_sqx, diagnostics

