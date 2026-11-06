"""Count-event certification for fixed ten-carrier marginals.

The declared class contains D>=a and E>=b for a,b=1,...,5,
and D+E>=k for k=1,...,10. Identical indicators are counted once.
LPs use 1e-8 primal/dual feasibility tolerances without changing inputs.
"""
from math import comb
import numpy as np
from scipy.optimize import linprog

TOL = 1e-8
GUARD = 1e-6


def headroom_sort_key(headroom, name):
    """Tie only unresolved zero headroom; leave reported numerical values intact."""
    if not np.isfinite(headroom):
        raise ValueError('Headroom must be finite.')
    return (-(0. if abs(headroom) <= TOL else headroom), name)


def catalog():
    d, e = np.indices((6, 6))
    candidates = [('D%d_E%d' % (a, b), (d >= a) & (e >= b))
                  for a in range(1, 6) for b in range(1, 6)]
    candidates += [('E%d' % k, d+e >= k) for k in range(1, 11)]
    unique = []
    for name, indicator in candidates:
        if not any(np.array_equal(indicator, old[1]) for old in unique):
            unique.append((name, indicator))
    assert len(unique) == 34
    return unique


def count_channel(epsilon):
    """P(observed count | true count) for five independently flipped flags."""
    if not 0 <= epsilon <= .5:
        raise ValueError('Invalid flag-flip probability.')
    channel = np.zeros((6, 6))
    for k in range(6):
        for retained in range(k+1):
            for false in range(6-k):
                channel[k, retained+false] += (
                    comb(k, retained)*(1-epsilon)**retained*epsilon**(k-retained)
                    * comb(5-k, false)*epsilon**false*(1-epsilon)**(5-k-false))
    np.testing.assert_allclose(channel.sum(axis=1), 1, rtol=0, atol=1e-14)
    return channel


def constraints(p):
    bits = np.array([[(m >> j) & 1 for m in range(32)] for j in range(5)])
    a = np.vstack([np.ones(1024), np.repeat(bits, 32, axis=1), np.tile(bits, (1, 32))])
    return a, np.r_[1, p, np.full(5, .65)], bits.sum(axis=0)


def checked_lp(cost, a, b, aub=None, bub=None, method='highs-ds'):
    fit = linprog(cost, A_eq=a, b_eq=b, A_ub=aub, b_ub=bub,
                  bounds=(0, None), method=method,
                  options={'primal_feasibility_tolerance': TOL,
                           'dual_feasibility_tolerance': TOL,
                           'ipm_optimality_tolerance': 1e-10})
    if not fit.success:
        raise RuntimeError(fit.message)
    y = fit.eqlin.marginals
    z = fit.ineqlin.marginals if aub is not None else np.empty(0)
    reduced = cost-a.T@y-(aub.T@z if aub is not None else 0)
    dual = b@y+(bub@z if aub is not None else 0)
    residuals = dict(equality=float(np.max(abs(a@fit.x-b))),
                     inequality=float(max(0, np.max(aub@fit.x-bub))) if aub is not None else 0.,
                     nonnegativity=float(max(0, -fit.x.min())),
                     dual_feasibility=float(max(0, -reduced.min(), z.max() if z.size else 0)),
                     duality_gap=float(abs(cost@fit.x-dual)))
    if max(residuals.values()) > TOL:
        raise AssertionError(residuals)
    return fit, residuals


def search(source, probabilities, epsilon, evidence_path=None):
    """Evaluate all candidates before ranking; keep full/even/odd source splits."""
    a, b, counts = constraints(probabilities)
    costs = np.exp2(2*np.asarray(source)-2).reshape(len(source), -1)
    channel = count_channel(epsilon)
    records, evidence = [], {}
    splits = {'all': np.arange(len(costs)), 'even': np.arange(0, len(costs), 2),
              'odd': np.arange(1, len(costs), 2)}
    for name, indicator in catalog():
        score = (channel@indicator.astype(float)@channel.T)[counts[:, None], counts[None, :]].ravel()
        maximum, maximum_check = checked_lp(-score, a, b)
        evidence[name+'_maximum_law'] = maximum.x
        evidence[name+'_maximum_dual'] = maximum.eqlin.marginals
        frontiers, checks, laws, duals = [], [], [], []
        for cost in costs:
            fit, check = checked_lp(-score, a, b, cost[None, :], np.array([.5]))
            frontiers.append(float(score@fit.x)); checks.append(check)
            laws.append(fit.x); duals.append(np.r_[fit.eqlin.marginals, fit.ineqlin.marginals])
        evidence[name+'_frontier_laws'] = np.array(laws)
        evidence[name+'_frontier_duals'] = np.array(duals)
        feasible = float(score@maximum.x)
        split_values = {key: dict(threshold=max(frontiers[i] for i in indices),
                                  headroom=feasible-max(frontiers[i] for i in indices))
                        for key, indices in splits.items()}
        # Direct minimization checks the certificate, independently of frontier inversion.
        threshold = split_values['all']['threshold']
        guarded = threshold+GUARD
        direct, direct_checks, direct_laws, direct_duals = [], [], [], []
        if guarded < feasible-TOL:
            for cost in costs:
                fit, check = checked_lp(cost, a, b, -score[None, :], np.array([-guarded]))
                direct.append(float(cost@fit.x)); direct_checks.append(check)
                direct_laws.append(fit.x)
                direct_duals.append(np.r_[fit.eqlin.marginals, fit.ineqlin.marginals])
            assert min(direct) > .5
        evidence[name+'_direct_laws'] = np.array(direct_laws)
        evidence[name+'_direct_duals'] = np.array(direct_duals)
        records.append(dict(event=name, indicator_by_counts=indicator.astype(int).tolist(),
                            feasible_maximum=feasible, sourcewise_thresholds=frontiers,
                            splits=split_values, frontier_residuals=checks,
                            feasible_maximum_residuals=maximum_check,
                            guarded_rate=guarded, direct_lower_certificates=direct,
                            direct_residuals=direct_checks,
                            direct_status='verified' if direct else 'guard exceeds feasible headroom'))
    rankings = {key: [r['event'] for r in sorted(records,
                 key=lambda r: headroom_sort_key(r['splits'][key]['headroom'], r['event']))] for key in splits}
    chosen = next(r for r in records if r['event'] == rankings['all'][0])
    # Check the selected frontier with the independent HiGHS interior-point algorithm.
    indicator = np.array(chosen['indicator_by_counts'])
    score = (channel@indicator@channel.T)[counts[:, None], counts[None, :]].ravel()
    cross = []
    for index, cost in enumerate(costs):
        fit, check = checked_lp(-score, a, b, cost[None, :], np.array([.5]), 'highs-ipm')
        difference = abs(float(score@fit.x)-chosen['sourcewise_thresholds'][index])
        assert difference < TOL
        cross.append(dict(source_index=index, objective_difference=difference, residuals=check))
    sufficient = .3759 if epsilon == 0 else .3410
    direct_sufficient = []
    for cost in costs:
        fit, check = checked_lp(cost, a, b, -score[None, :], np.array([-sufficient]))
        direct_sufficient.append(dict(lower_certificate=float(cost@fit.x), residuals=check))
    assert min(r['lower_certificate'] for r in direct_sufficient) > .5
    if evidence_path is not None:
        np.savez_compressed(evidence_path, **evidence)
    return dict(flip_probability=epsilon, candidate_count=len(records),
                LP_feasibility_tolerance=TOL, direct_guard=GUARD,
                objective='feasible maximum minus maximum sourcewise bad-law event rate',
                catalog=records, rankings=rankings, selected_event=chosen['event'],
                selected_cross_algorithm=cross, sufficient_population_rate=sufficient,
                sufficient_rate_direct_checks=direct_sufficient)
