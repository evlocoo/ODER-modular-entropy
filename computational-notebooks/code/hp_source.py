"""Finite Hayden–Preskill source using seeded complex-Gaussian QR and explicit register ordering."""
from __future__ import annotations
import numpy as np


def pure_subsystem_entropy(state: np.ndarray, keep_axes) -> float:
    """Base-2 von Neumann entropy of a subsystem of a pure qubit tensor."""
    keep=tuple(sorted(keep_axes))
    rest=tuple(i for i in range(state.ndim) if i not in keep)
    if len(keep)>len(rest): keep,rest=rest,keep
    matrix=np.transpose(state,keep+rest).reshape(2**len(keep),-1)
    eigenvalues=np.linalg.eigvalsh(matrix@matrix.conj().T)
    eigenvalues=eigenvalues[eigenvalues>1e-15]
    return float(-np.sum(eigenvalues*np.log2(eigenvalues)))


def finite_hp_state(seed: int, black_hole_qubits: int=5, emitted_qubits: int=5):
    """Return pure tensor with axes Q, remaining hole, emitted D, early E.

    Q and diary A initially form a Bell pair; n hole qubits B and n early
    radiation qubits E form a maximally entangled pair. A seeded Haar unitary
    acts on A B, whose output is partitioned into remaining C and emitted D.
    Every register is returned as individual qubit axes.
    """
    n=int(black_hole_qubits)
    m=int(emitted_qubits)
    if n<0 or not 0<=m<=n+1:raise ValueError('Invalid qubit counts')
    dim=2**(n+1)
    rng=np.random.default_rng(int(seed))
    matrix=rng.normal(size=(dim,dim))+1j*rng.normal(size=(dim,dim))
    unitary,triangular=np.linalg.qr(matrix)
    diagonal=np.diag(triangular)
    unitary=unitary*(diagonal/np.abs(diagonal))
    remaining=n+1-m
    state=unitary.reshape(2**remaining,2**m,2,2**n).transpose(2,0,1,3)
    state=state.reshape((2,)*(2*n+2))/np.sqrt(dim)
    diagnostics={
        'black_hole_qubits':n,
        'emitted_qubits':m,
        'remaining_qubits':remaining,
        'unitary_residual':float(np.linalg.norm(unitary.conj().T@unitary-np.eye(dim),ord=2)),
        'normalization_error':float(abs(np.vdot(state.ravel(),state.ravel()).real-1)),
    }
    return state,diagnostics

