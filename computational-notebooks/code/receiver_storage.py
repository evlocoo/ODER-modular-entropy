"""Finite switched exchange, flagged loss, and five-share storage comparison.

Equations: app:worked-noisy-receiver and app:noise-ordering-proof in the
accompanying manuscript. Time is measured in units of 1/Omega.
"""
import itertools
import math
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm

TOL = 1e-10


def envelope(t):
    if t <= .8:
        return 1.
    if t >= 1.:
        return 0.
    q = (t-.8)/.2
    return 1-3*q*q+2*q*q*q


def exchange(sites=2, acquisition_loss=0., rtol=1e-12, atol=1e-14,
             max_step=1e-3):
    """No-jump amplitudes; uniform loss commutes with coherent exchange."""
    if sites not in (2, 3) or acquisition_loss < 0:
        raise ValueError('Use two or three sites and a nonnegative loss rate.')
    def rhs(t, u):
        links = [5*np.pi/9*envelope(t)] if sites == 2 else [np.sqrt(2)*envelope(t), np.sqrt(6)]
        h = np.diag(links, 1)+np.diag(links, -1)
        return -1j*h@u-acquisition_loss*u/2
    initial = np.zeros(sites, complex)
    initial[0] = 1
    # Split at the switch onset so an adaptive step cannot miss the taper.
    a = solve_ivp(rhs, (0., .8), initial, method='DOP853', rtol=rtol,
                  atol=atol, max_step=max_step, dense_output=True)
    b = solve_ivp(rhs, (.8, 1.), a.y[:, -1], method='DOP853', rtol=rtol,
                  atol=atol, max_step=max_step, dense_output=True)
    if not (a.success and b.success):
        raise RuntimeError((a.message, b.message))
    times = np.linspace(0, 1, 501)
    states = np.column_stack([a.sol(t) if t <= .8 else b.sol(t) for t in times])
    return times, states


def independent_three_site(steps):
    """Time-ordered midpoint matrix exponentials, independently of the ODE."""
    h = np.array([[0, np.sqrt(2), 0], [np.sqrt(2), 0, np.sqrt(6)], [0, np.sqrt(6), 0]])
    u = expm(-.8j*h) @ np.array([1, 0, 0], complex)
    dt = .2/steps
    for j in range(steps):
        q = (j+.5)/steps
        link = np.sqrt(2)*(1-3*q*q+2*q*q*q)
        h = np.array([[0, link, 0], [link, 0, np.sqrt(6)], [0, np.sqrt(6), 0]])
        u = expm(-1j*dt*h)@u
    return u


def survival(acquisition_loss=.1, storage_loss=.05, hold=.4):
    if min(acquisition_loss, storage_loss, hold) < 0:
        raise ValueError('Loss rates and hold must be nonnegative.')
    return math.exp(-acquisition_loss-storage_loss*hold)


def terminal_bounds(p, full_floor, no_early_ceiling,
                    verification_floor=None, verification_ceiling=None):
    """All outcomes included. Early-only recovery has Bell overlap 1/4."""
    result = dict(full_early_lower=.25+p*(full_floor-.25),
                  no_early_upper=.25+p*(no_early_ceiling-.25),
                  full_early_universal_upper=.25+.75*p)
    if verification_floor is not None:
        result['verification_full_lower'] = .5+p*(verification_floor-.5)
    if verification_ceiling is not None:
        result['verification_no_early_upper'] = .5+p*(verification_ceiling-.5)
    return result


def erasure_kraus(p, dimension=2):
    if not 0 <= p <= 1:
        raise ValueError('Invalid probability.')
    k = np.zeros((dimension+1, dimension), complex)
    k[:dimension] = np.sqrt(p)*np.eye(dimension)
    kraus = [k]
    for j in range(dimension):
        k = np.zeros_like(k)
        k[dimension, j] = np.sqrt(1-p)
        kraus.append(k)
    return kraus


def partial_trace(rho, dims, keep):
    dims = list(dims)
    tensor = rho.reshape(dims+dims)
    for i in reversed(range(len(dims))):
        if i not in keep:
            tensor = np.trace(tensor, axis1=i, axis2=i+len(dims))
            dims.pop(i)
    return tensor.reshape(math.prod(dims), math.prod(dims))


def entropy(rho):
    values = np.linalg.eigvalsh(rho)
    if values.min() < -TOL or not np.isfinite(values).all():
        raise AssertionError('Invalid density eigenvalues.')
    positive = values[values > 0]
    return float(-positive@np.log2(positive))


def information(rho, dims, a, b):
    return entropy(partial_trace(rho, dims, a))+entropy(partial_trace(rho, dims, b))-entropy(partial_trace(rho, dims, sorted(set(a+b))))


def channel_checks():
    """Check the channel on entangled coherent inputs, not populations only."""
    v = np.array([1, 1j, 2, -1, .5j, 2j, 1, 3], complex)
    v /= np.linalg.norm(v)
    rho = np.outer(v, v.conj())  # Q,E,X all have dimension two.
    i_qe = information(rho, [2, 2, 2], [0], [1])
    i_qex = information(rho, [2, 2, 2], [0], [1, 2])
    rows = []
    for p in [0., .2, survival(), 1.]:
        kraus = erasure_kraus(p)
        trace_residual = np.linalg.norm(sum(k.conj().T@k for k in kraus)-np.eye(2))
        out = np.zeros((12, 12), complex)
        for k in kraus:
            lift = np.kron(np.eye(4), k)
            out += lift@rho@lift.conj().T
        predicted = i_qe+p*(i_qex-i_qe)
        residual = abs(information(out, [2, 2, 3], [0], [1, 2])-predicted)
        bell = np.array([1, 0, 0, 1], complex)/np.sqrt(2)
        choi = sum(np.outer(np.kron(np.eye(2), k)@bell,
                            (np.kron(np.eye(2), k)@bell).conj()) for k in kraus)
        minimum = float(np.linalg.eigvalsh(choi).min())
        assert max(trace_residual, residual, abs(np.trace(out)-1)) < TOL
        assert minimum >= -TOL
        rows.append(dict(p=p, trace_preservation_residual=float(trace_residual),
                         conditional_information_residual=float(residual),
                         choi_minimum_eigenvalue=minimum))
    return rows


def noisy_protocol(s, protocol):
    """Explicit activation-mask and conditional survivor-mask probabilities."""
    if not 0 <= s <= 1 or protocol not in ('A', 'B'):
        raise ValueError('Invalid survival or protocol.')
    activation = {0: .4, 31: .6} if protocol == 'A' else {
        sum(1 << j for j in mask): .1 for mask in itertools.combinations(range(5), 3)}
    terminal = np.zeros(32)
    for active, weight in activation.items():
        for retained in range(32):
            if retained & ~active:
                continue
            k = retained.bit_count()
            terminal[retained] += weight*s**k*(1-s)**(active.bit_count()-k)
    marginal = np.array([sum(w for m, w in enumerate(terminal) if m & (1 << j)) for j in range(5)])
    q = float(sum(w for m, w in enumerate(terminal) if m.bit_count() >= 3))
    formula = .6*(10*s**3-15*s**4+6*s**5) if protocol == 'A' else s**3
    assert abs(terminal.sum()-1) < TOL
    assert np.min(terminal) >= 0
    assert abs(q-formula) < TOL and np.max(abs(marginal-.6*s)) < TOL
    expected_active=sum(w*m.bit_count() for m,w in activation.items())
    return dict(protocol=protocol, survival=s, authorized_probability=q,
                bell_fidelity=.25+.75*q, individual_terminal_efficiency=float(marginal[0]),
                activation_entropy_bits=float(-sum(w*np.log2(w) for w in activation.values())),
                peak_active_links=5 if protocol == 'A' else 3,
                expected_active_links=expected_active,
                expected_integrated_g2_over_Omega=expected_active*(5*np.pi/9)**2*(.8+.2*13/35)), terminal


def validate():
    area = quad(envelope, 0, .8)[0]+quad(envelope, .8, 1)[0]
    assert abs(5*np.pi/9*area-np.pi/2) < 1e-14
    records = []
    for sites in [2, 3]:
        times, u = exchange(sites)
        _, loss = exchange(sites, acquisition_loss=.1)
        eta = float(np.sum(abs(u[1:, -1])**2))
        p = eta*survival()
        norm_error = float(np.max(abs(np.sum(abs(u)**2, axis=0)-1)))
        loss_error = float(np.max(abs(loss-np.exp(-.05*times)*u)))
        assert max(norm_error, loss_error) < TOL
        if sites == 2:
            assert abs(u[1, -1]+1j) < TOL
        else:
            independent = [independent_three_site(n) for n in [500, 1000, 2000]]
            raw = [float(np.linalg.norm(x-u[:, -1])) for x in independent]
            extrapolated = (4*independent[-1]-independent[-2])/3
            extra = float(np.linalg.norm(extrapolated-u[:, -1]))
            assert raw[2] < raw[1] < raw[0] and extra < TOL
            _, refined = exchange(3, rtol=2e-13, atol=2e-15, max_step=.0005)
            assert np.linalg.norm(refined[:, -1]-u[:, -1]) < TOL
            assert abs(eta-.702063959238) < 5.1e-13
            records.append(dict(check='three_site_refinement', midpoint_steps=[500, 1000, 2000],
                                raw_amplitude_residuals=raw, richardson_residual=extra))
        records.append(dict(sites=sites, memory_sites=list(range(1, sites)),
                            no_loss_memory_population=eta, retained_probability=p,
                            norm_residual=norm_error, uniform_loss_residual=loss_error))
    for dose in [.12, .30]:
        a, _ = noisy_protocol(np.exp(-dose), 'A')
        b, _ = noisy_protocol(np.exp(-dose), 'B')
        assert (a['bell_fidelity'] > b['bell_fidelity']) == (dose == .30)
    a, _ = noisy_protocol(5/6, 'A')
    b, _ = noisy_protocol(5/6, 'B')
    assert abs(a['bell_fidelity']-b['bell_fidelity']) < TOL
    return dict(pulse_area=5*np.pi/9*area, receiver_checks=records,
                channel_checks=channel_checks(), crossover_survival=5/6,
                hold_guarantee=(-np.log(.25/.5972)-.1)/.05,
                hold_obstruction=(np.log(3)-.1)/.05)
