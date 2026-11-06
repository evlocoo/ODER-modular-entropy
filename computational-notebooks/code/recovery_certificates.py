"""Recovery certificates with the order of averaging stated explicitly.

The decoder receives collection flags and quantum carriers, never Q.
The untouched maximally mixed qubit reference sets the Bell-overlap convention.
"""
import numpy as np
from scipy.optimize import brentq
from cavity import mask_weights


def binary_bound(j):
    if not 0 <= j <= 1:
        raise ValueError('Normalized information is outside [0,1].')
    if j == 0:
        return .25
    if j == 1:
        return 1.
    def divergence(f):
        return f*np.log2(f/.25)+(1-f)*np.log2((1-f)/.75) if f < 1 else 2.
    return brentq(lambda f: divergence(f)-2*j, .25, 1., xtol=5e-15)


def statewise(source, late_probabilities):
    source = np.asarray(source)
    w = mask_weights(np.asarray(late_probabilities))
    if source.ndim != 3 or source.shape[1:] != (32, 32):
        raise ValueError('Expected source, late-mask, early-mask axes.')
    if np.min(late_probabilities) < 0 or np.max(late_probabilities) > 1:
        raise ValueError('Carrier probabilities must be physical.')
    assert abs(w.sum()-1) < 1e-12
    rows = []
    for s, table in enumerate(source):
        row = dict(source_index=s)
        for early, name in [(0, 'no_early'), (31, 'full_early')]:
            values = table[:, early]
            assert np.min(values) >= -2e-12 and np.max(values) <= 1+2e-12
            # An empty late register is analytically uncorrelated with Q,
            # including when the early register is present. Check its raw
            # roundoff residual, then use that exact identity for this branch.
            assert abs(values[0]) < 2e-12
            ceilings = np.array([.25]+[binary_bound(float(v)) for v in values[1:]])
            mean = float(w@values)
            row[name+'_information'] = mean
            row[name+'_aggregate_lower'] = float(2**(2*mean-2))
            row[name+'_aggregate_upper'] = binary_bound(mean)
            row[name+'_branch_lower'] = float(w@np.exp2(2*values-2))
            row[name+'_branch_upper'] = float(w@ceilings)
            assert row[name+'_branch_lower'] >= row[name+'_aggregate_lower']-1e-10
            assert row[name+'_branch_upper'] <= row[name+'_aggregate_upper']+1e-10
        rows.append(row)
    return rows


def branch_tables(source):
    """All late/early masks; exact empty-late identity checked explicitly."""
    source = np.asarray(source)
    assert np.max(abs(source[:, 0, :])) < 2e-12
    lower = np.exp2(2*source-2)
    upper = np.empty_like(source)
    upper[:, 0, :] = .25
    for s in range(len(source)):
        for late in range(1, 32):
            upper[s, late] = [binary_bound(float(j)) for j in source[s, late]]
    return lower, upper


def catcher(detector_output_energy, epsilon=1e-6):
    d = np.asarray(detector_output_energy)
    if epsilon <= 0 or d.min() < -1e-12:
        raise ValueError('Invalid detector energy or startup regularizer.')
    memory = d*d/(epsilon+d)
    leakage = epsilon*d/(epsilon+d)
    assert np.max(abs(memory+leakage-d)) < 1e-10
    assert np.max(d-memory) <= epsilon+1e-12
    return memory, leakage
