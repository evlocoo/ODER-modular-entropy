"""Controlled catcher and fixed-hardware delivery from the November equations.

This is a current equation-based implementation. Comparison with the later
original study is pending access to that archive; no recovery claim is made.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh_tridiagonal
from cavity import mask_weights
from recovery_certificates import catcher, statewise


def weight_curves(probabilities):
    return np.array([mask_weights(row) for row in probabilities])


def caught_run(source, run, epsilon=1e-6):
    t = run['times']
    cavity = abs(run['cavities'])**2
    output = run['observer_carriers']-cavity
    memory, leakage = catcher(output, epsilon)
    rows = statewise(source, memory[-1])
    full = source[:, :, 31]
    weights = weight_curves(memory)
    r = weights@full.T
    inventory = weight_curves(memory+cavity)@full.T
    derivative = np.zeros_like(r)
    beta = 1-(epsilon/(epsilon+output))**2
    # The symmetric detector has kappa_det=0.515/2 in the declared M=1 units.
    mdot = .2575*cavity*beta
    for j in range(5):
        for mask in range(32):
            if mask & (1 << j):
                continue
            other = np.ones(len(t))
            for k in range(5):
                if k != j:
                    other *= memory[:, k] if mask & (1 << k) else 1-memory[:, k]
            derivative += (mdot[:, j]*other)[:, None]*(full[:, mask | (1 << j)]-full[:, mask])[None, :]
    assert np.min(derivative) > -1e-10
    assert np.max(r-inventory) < 1e-10
    assert np.max(derivative-.2575*(inventory-r)*5) < 1e-10
    selected = [int(np.argmin(abs(t-x))) for x in [59., 120.]]
    gap = inventory[selected]-r[selected]
    assert np.min(gap) > 0
    effective = derivative[selected]/gap
    assert np.all(effective[1] < effective[0])
    return dict(times=t, detector_output=output, memory=memory, leakage=leakage,
                retained_information=r, joint_inventory=inventory, derivative=derivative,
                effective_rate_59_120=effective, certificates=rows)


def independent_absorber(times, detector_wave, epsilon=1e-6):
    """Integrate d, catcher amplitude and escaped output from the wave alone."""
    wave = CubicSpline(times, detector_wave)
    def rhs(t, state):
        d, b, escaped = state
        y = wave(t)
        rate = abs(y)**2
        l2 = rate/(epsilon+d)
        bout = -l2*b/2-rate/np.sqrt(epsilon+d)
        # y_out=y+ell*b, with ell=y/sqrt(epsilon+d).
        return [rate, bout, rate*(1+b/np.sqrt(epsilon+d))**2]
    fit = solve_ivp(rhs, (float(times[0]), float(times[-1])), [0., 0., 0.],
                    method='DOP853', t_eval=times, rtol=2e-11, atol=2e-13,
                    max_step=.025)
    if not fit.success:
        raise RuntimeError(fit.message)
    d, amplitude, escaped = fit.y
    exact_amplitude = -d/np.sqrt(epsilon+d)
    balance = float(np.max(abs(d-amplitude**2-escaped)))
    residual = float(np.max(abs(amplitude-exact_amplitude)))
    assert max(balance, residual) < 1e-10
    return dict(balance_residual=balance, amplitude_residual=residual,
                terminal_retained=float(amplitude[-1]**2),
                terminal_detector_output=float(d[-1]))


def deadline_scan(source, caught, total=120., omega=.025, n=62):
    t = caught['times']
    keep = (t >= 0)&(t <= total)
    cutoffs = t[keep]
    lower = weight_curves(caught['memory'][keep])@np.exp2(2*source[:, :, 31]-2).T
    duration = total-cutoffs
    x = omega*duration
    error = 2*np.sqrt(n+1)*np.tanh(x)**(n+2)
    # The max with zero is the stated analytic lower-certificate definition.
    floor = np.maximum(0., 1-np.cosh(x)**(-4)-error)
    scores = .25+floor[:, None]*(lower-.25)
    limiting = scores.min(axis=1)
    best = int(np.argmax(limiting))
    samples = []
    links = omega*np.sqrt(np.arange(1, n+1)*(np.arange(1, n+1)+1))
    values, vectors = eigh_tridiagonal(np.zeros(n+1), links)
    for cutoff in [80., 74.4]:
        i = int(np.argmin(abs(cutoffs-cutoff)))
        survival_amplitude = np.sum(vectors[0]**2*np.exp(-1j*values*duration[i]))
        probability = float(1-abs(survival_amplitude)**2)
        samples.append(dict(cutoff=float(cutoffs[i]), capture_duration=float(duration[i]),
                            exact_finite_capture=probability,
                            analytic_probability_error=float(error[i]),
                            certified_capture_floor=float(floor[i]),
                            worst_source_fidelity_lower=float(limiting[i]),
                            limiting_source_index=int(np.argmin(scores[i]))))
    assert abs(cutoffs[best]-74.4) < 1e-9
    return dict(cutoffs=cutoffs, worst_source_floor=limiting,
                best_cutoff=float(cutoffs[best]), selected=samples)
