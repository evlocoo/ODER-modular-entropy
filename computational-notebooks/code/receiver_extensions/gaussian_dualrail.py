"""Exact dual-rail Pauli-plus-erasure reduction of identical Gaussian rail maps.

Convention: [q,p]=i, vacuum covariance I/2, output mean=X input mean,
output covariance=X V X.T+Y. Channel must have zero displacement.
Each rail has its own independent identical environment. Accessible output is
one oscillator per rail; total photon number !=1 is mapped to an erasure flag.
No Fock cutoff enters the formulas.
"""
from __future__ import annotations
import numpy as np


def gaussian_dualrail(X, Y, *, tolerance=1e-9):
    X=np.asarray(X,dtype=float); Y=np.asarray(Y,dtype=float)
    if X.shape != (2,2) or Y.shape != (2,2):
        raise ValueError('X,Y must each be 2 by 2')
    if not np.allclose(Y,Y.T,atol=tolerance,rtol=0):
        raise ValueError('Y must be symmetric')
    J=np.array([[0.,1.],[-1.,0.]])
    cp=Y+0.5j*(J-X@J@X.T)
    mineig=float(np.linalg.eigvalsh(cp).min())
    if mineig < -tolerance:
        raise ValueError(f'Gaussian channel violates complete positivity: {mineig}')
    alpha=0.5*(X[0,0]+X[1,1]+1j*(X[1,0]-X[0,1]))
    beta=0.5*(X[0,0]-X[1,1]+1j*(X[1,0]+X[0,1]))
    V=0.5*X@X.T+Y
    # Vacuum-output characteristic function exp[-x.T H x], x=(Re xi,Im xi).
    H=np.array([[V[1,1],-V[0,1]],[-V[0,1],V[0,0]]])
    G=H+0.5*np.eye(2)
    pref=1/np.sqrt(np.linalg.det(G))
    Sigma=0.5*np.linalg.inv(G)
    w=np.array([alpha.conjugate()-beta,1j*(alpha.conjugate()+beta)])
    Z=np.real(np.outer(w.conjugate(),w))
    R=float(np.trace(Sigma)); Q=float(np.trace(Z@Sigma))
    QR=Q*R+2*float(np.trace(Z@Sigma@Sigma))
    a=float(pref); b=float(pref*(1-R)); c=float(pref*(1-Q))
    d=float(pref*(1-Q-R+QR))
    u=complex(pref*(w.conjugate()@Sigma@np.array([1.,1j])))
    v=complex(-pref*(w.conjugate()@Sigma@np.array([1.,-1j])))
    ad=a*d; bc=b*c
    weights=np.array([(ad+abs(u)**2)/2,(bc+abs(v)**2)/2,
                      (bc-abs(v)**2)/2,(ad-abs(u)**2)/2])
    survival=float(ad+bc); erasure=1-survival
    if min(np.min(weights),erasure,a,b,c,d) < -100*tolerance:
        raise RuntimeError('Unphysical Fock block; check channel or numerical conditioning')
    # Keep raw values so tiny numerical residuals are visible, not hidden.
    return dict(X=X.tolist(),Y=Y.tolist(),vacuum_output_covariance=V.tolist(),
                alpha=[alpha.real,alpha.imag],beta=[beta.real,beta.imag],
                a=a,b=b,c=c,d=d,u=[u.real,u.imag],v=[v.real,v.imag],
                pauli_weights=dict(zip(('I','X','Y','Z'),map(float,weights))),
                survival=survival,erasure=float(erasure),
                unrecorded_pauli_error=float(weights[1:].sum()),
                deterministic_bell_fidelity=float(weights[0]+erasure/4),
                cp_min_eigenvalue=mineig)


def from_bogoliubov(alpha,beta,vacuum_output_covariance,*,tolerance=1e-9):
    """Alternative interface when a_out=alpha*a+beta*a^dagger+environment."""
    alpha=complex(alpha); beta=complex(beta)
    X=np.array([[(alpha+beta).real,-(alpha-beta).imag],
                [(alpha+beta).imag,(alpha-beta).real]])
    V=np.asarray(vacuum_output_covariance,float)
    return gaussian_dualrail(X,V-0.5*X@X.T,tolerance=tolerance)


def product_error_bound(channels):
    """Half trace-distance/decoder score perturbation under independent rail pairs."""
    errors=np.array([c['unrecorded_pauli_error'] for c in channels])
    if np.any(errors < -1e-8) or np.any(errors>1+1e-8):
        raise ValueError('Invalid error weights')
    errors=np.clip(errors,0.,1.)
    return float(-np.expm1(np.log1p(-errors).sum()))
