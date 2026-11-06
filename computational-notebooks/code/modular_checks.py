#!/usr/bin/env python3
"""Finite detector and conformal-clock checks; not a black-hole model.

Reusable calculations; no files are written on import.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
from scipy.special import gamma


def entropy(rho):
    eig = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    eig = eig[eig > 1e-13]
    return float(-np.sum(eig * np.log(eig)))


def partial_trace(rho, dims, keep):
    out = rho.reshape(tuple(dims) * 2)
    dims = list(dims)
    for axis in reversed(range(len(dims))):
        if axis not in keep:
            out = np.trace(out, axis1=axis, axis2=axis + len(dims))
            dims.pop(axis)
    return out.reshape(int(np.prod(dims)), int(np.prod(dims)))


def binary_entropy(x):
    return float(-sum(y * np.log(y) for y in (x, 1-x) if y > 1e-14))


def finite_detector():
    # The faithful state defines the REFERENCE modular Hamiltonian.
    # It is not the actual Bell-input marginal, which is I/2.
    reference_p, coupling = 0.8, 0.37
    energies = -np.log([reference_p, 1-reference_p])
    k = np.diag(energies)
    eye = np.eye(2)
    lowering = np.array([[0., 1.], [0., 0.]])
    exchange = np.kron(lowering.T, lowering) + np.kron(lowering, lowering.T)
    h_ab = np.kron(k, eye) + np.kron(eye, k) + coupling * exchange
    h_qab = np.kron(eye, h_ab)
    psi0 = np.zeros(8, dtype=complex)
    psi0[0] = psi0[6] = 1 / np.sqrt(2)
    rho0 = np.outer(psi0, psi0.conj())
    maximum = dict(density_error=0., information_error=0., probability_error=0.)
    samples = []
    for theta in np.linspace(0., 2*np.pi, 81):
        tau = theta / coupling
        unitary = expm(-1j * tau * h_qab)
        rho = unitary @ rho0 @ unitary.conj().T
        analytic = np.zeros(8, dtype=complex)
        analytic[0] = np.exp(-2j * energies[0] * tau) / np.sqrt(2)
        phase = np.exp(-1j * sum(energies) * tau) / np.sqrt(2)
        analytic[6] = phase * np.cos(theta)
        analytic[5] = -1j * phase * np.sin(theta)
        qb = partial_trace(rho, [2,2,2], [0,2])
        q = partial_trace(rho, [2,2,2], [0])
        b = partial_trace(rho, [2,2,2], [2])
        retrieval = (entropy(q) + entropy(b) - entropy(qb)) / (2*np.log(2))
        probability = float(np.sin(theta)**2)
        analytic_retrieval = (np.log(2) + binary_entropy(probability/2)
                              - binary_entropy((1-probability)/2)) / (2*np.log(2))
        maximum['density_error'] = max(maximum['density_error'],
            float(np.linalg.norm(rho - np.outer(analytic, analytic.conj()))))
        maximum['information_error'] = max(maximum['information_error'],
            abs(retrieval - analytic_retrieval))
        maximum['probability_error'] = max(maximum['probability_error'],
            abs(2*float(b[1,1].real) - probability))
        if any(abs(theta-target) < 1e-10 for target in
               [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 2*np.pi]):
            samples.append(dict(g_tau=float(theta), tau=float(tau),
                                conditional_excitation_transfer=probability,
                                normalized_reference_information=retrieval))
    assert max(maximum.values()) < 1e-11, maximum
    probe_p = 0.1
    return dict(reference_state=[reference_p, 1-reference_p],
                modular_energies=energies.tolist(), coupling=coupling,
                sample_count=81, errors=maximum, samples=samples,
                nonlinear_example=dict(transfer_probability=probe_p,
                    reference_information=(np.log(2)+binary_entropy(probe_p/2)
                    -binary_entropy((1-probe_p)/2))/(2*np.log(2))),
                scope='Closed resonant exchange detector, not flagged erasure; '
                      'modular reference state differs from Bell-input marginal.')


def diamond_checks():
    # Given central conformal diamond orbit t(s)=T*tanh(pi*s).
    # Check exact geometry and the dimension-one weighted scalar correlator.
    radius = 2.0
    ts = lambda s: radius*np.tanh(np.pi*s)
    derivative = lambda s: np.pi*radius/np.cosh(np.pi*s)**2
    real_values = np.linspace(-3., 3., 121)
    imaginary_values = np.linspace(-.249, .249, 101)
    strip_values = np.tanh(np.pi*(real_values[:,None]+1j*imaginary_values[None,:]))
    max_interior = float(np.max(abs(strip_values)))
    boundary_error = float(np.max(abs(abs(np.tanh(np.pi*(real_values+0.25j)))-1)))
    delta = .35
    weighted, unweighted = [], []
    for mean in [-.5, -.25, 0., .25, .5]:
        s1, s2 = mean+delta/2, mean-delta/2
        w = -1/(4*np.pi**2*(ts(s1)-ts(s2))**2)
        unweighted.append(float(w))
        weighted.append(float(derivative(s1)*derivative(s2)*w))
    stationary_value = float(-1/(4*np.sinh(np.pi*delta)**2))
    weighted_error = float(max(abs(x-stationary_value) for x in weighted))
    assert max_interior < 1
    assert boundary_error < 1e-12
    assert weighted_error < 1e-12
    return dict(radius=radius, assumed_orbit='t(s)=T*tanh(pi*s)',
                strip_halfwidth=.25, sampled_max_abs_interior=max_interior,
                boundary_unit_modulus_error=boundary_error,
                proper_time_profile='A(t)=t/T on -T<t<T; not tanh in t',
                clock_samples=[dict(s=s, t=float(ts(s)),
                    dt_ds=float(derivative(s)), ds_dt=float(1/derivative(s)))
                    for s in [0., .25, .5, 1.]],
                delta_s=delta, unweighted_correlator=unweighted,
                weighted_correlator=weighted,
                weighted_stationary_value=stationary_value,
                weighted_stationarity_error=weighted_error,
                scope='Noncoincident real-time massless dimension-one scalar '
                      'correlator. These algebraic checks do not prove a QFT '
                      'modular-flow theorem, full KMS condition, or physical '
                      'implementation of conformal detector weighting.')


def dilation_survival():
    """Positive endpoint construction on L2(R+,dp/p), not a local receiver."""
    rows = []
    errors = dict(norm=0., overlap=0., hazard=0., reference_information=0.,
                  coherent_information=0., isometry=0., minimum_choi_eigenvalue=0.)
    for h in [.5, 1., 1.5]:
        for ell in [.7, 1.3]:
            normalization = (2*ell)**h / np.sqrt(gamma(2*h))
            for u in [0., .5, 1., 2., 4.]:
                norm, norm_quad_err = quad(
                    lambda momentum: normalization**2 * np.exp(-2*h*u)
                    * momentum**(2*h-1)
                    * np.exp(-2*ell*np.exp(-u)*momentum), 0., np.inf,
                    epsabs=2e-12, epsrel=2e-12)
                overlap, overlap_quad_err = quad(
                    lambda momentum: normalization**2 * np.exp(-h*u)
                    * momentum**(2*h-1)
                    * np.exp(-ell*(1+np.exp(-u))*momentum), 0., np.inf,
                    epsabs=2e-12, epsrel=2e-12)
                analytic_overlap = np.cosh(u/2)**(-2*h)
                survival = analytic_overlap**2
                retrieval = 1-survival
                step = 1e-5
                log_survival = lambda x: -4*h*np.log(np.cosh(x/2))
                numerical_hazard = -(log_survival(u+step)-log_survival(u-step))/(2*step)
                hazard = 2*h*np.tanh(u/2)
                # V: logical input C^2 -> residual C^3 tensor memory C^3.
                # Index 2 is the orthogonal erasure/vacuum flag in each output.
                v = np.zeros((9,2), dtype=complex)
                for j in [0,1]:
                    v[3*j+2,j] = np.sqrt(survival)
                    v[6+j,j] = np.sqrt(retrieval)
                bell = np.array([1,0,0,1], dtype=complex)/np.sqrt(2)
                choi_purification = np.kron(np.eye(2),v) @ bell
                global_rho = np.outer(choi_purification,choi_purification.conj())
                qm = partial_trace(global_rho,[2,3,3],[0,2])
                q = partial_trace(global_rho,[2,3,3],[0])
                memory = partial_trace(global_rho,[2,3,3],[2])
                r_from_entropy = (entropy(q)+entropy(memory)-entropy(qm))/(2*np.log(2))
                coherent_information = entropy(memory)-entropy(qm)
                errors['norm'] = max(errors['norm'],abs(norm-1))
                errors['overlap'] = max(errors['overlap'],abs(overlap-analytic_overlap))
                errors['hazard'] = max(errors['hazard'],abs(numerical_hazard-hazard))
                errors['reference_information'] = max(errors['reference_information'],
                                                       abs(r_from_entropy-retrieval))
                errors['coherent_information'] = max(errors['coherent_information'],
                    abs(coherent_information-(2*retrieval-1)*np.log(2)))
                errors['isometry'] = max(errors['isometry'],float(np.linalg.norm(v.conj().T@v-np.eye(2))))
                errors['minimum_choi_eigenvalue'] = min(errors['minimum_choi_eigenvalue'],
                                                       float(np.min(np.linalg.eigvalsh(qm))))
                rows.append(dict(h=h, ell=ell, u=u, norm_integral=norm,
                    overlap_integral=overlap, overlap_exact=float(analytic_overlap),
                    quadrature_reported_error=max(norm_quad_err,overlap_quad_err),
                    survival=float(survival), retrieval=float(retrieval),
                    hazard=float(hazard), r_from_entropy=r_from_entropy,
                    coherent_information_nats=coherent_information))
    assert errors['norm'] < 1e-10
    assert errors['overlap'] < 1e-10
    assert errors['hazard'] < 1e-9
    assert errors['reference_information'] < 1e-12
    assert errors['coherent_information'] < 1e-12
    assert errors['isometry'] < 1e-12
    assert errors['minimum_choi_eigenvalue'] > -1e-12
    return dict(hilbert_space='L2(R+,dp/p)',
                packet='psi(p)=(2*ell)^h/sqrt(Gamma(2*h))*p^h*exp(-ell*p)',
                evolution='psi_u(p)=psi(exp(-u)*p)',
                amplitude='sech(u/2)^(2*h)', survival='sech(u/2)^(4*h)',
                halfline_hazard='2*h*tanh(u/2) for u>=0',
                errors=errors, samples=rows,
                scope='Endpoint family with a fixed initial-mode projection and '
                      'coherent spectator-label routing. No proof of a single '
                      'continuous collector, local causal receiver, or a '
                      'calibrated proper-time clock. Modular interpretation '
                      'requires a specified algebra and state, beyond dilation '
                      'unitarity alone.')


