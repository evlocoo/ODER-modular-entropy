"""Collection-law LPs for the retained branch-resolved recovery statements."""
import numpy as np
from scipy.optimize import linprog
from recovery_certificates import branch_tables


def solve(c, a, b, aub=None, bub=None, bounds=(0, None), method='highs-ds'):
    r = linprog(c, A_eq=a, b_eq=b, A_ub=aub, b_ub=bub, bounds=bounds,
                method=method, options={'primal_feasibility_tolerance': 1e-9,
                                        'dual_feasibility_tolerance': 1e-9})
    if not r.success:
        raise RuntimeError(r.message)
    equality = float(np.max(abs(a@r.x-b)))
    inequality = float(max(0., np.max(aub@r.x-bub))) if aub is not None else 0.
    dual_value = b@r.eqlin.marginals
    if aub is not None:
        dual_value += bub@r.ineqlin.marginals
    duality = abs(float(c@r.x-dual_value))
    assert max(equality, inequality, duality) < 1e-8
    return r, dict(equality_residual=equality, inequality_residual=inequality,
                   primal_dual_gap=duality)


def evaluate(source, late_probabilities, evidence_path=None):
    lower, upper = branch_tables(source)
    bits = np.array([[(m >> j) & 1 for m in range(32)] for j in range(5)])
    a = np.vstack([np.ones(32), bits])
    b = np.r_[1, late_probabilities]
    late_rows = []
    evidence = {}
    cross_checks = []
    for s in range(len(source)):
        hi, check_hi = solve(-upper[s, :, 0], a, b)
        lo, check_lo = solve(lower[s, :, 31], a, b)
        paired, check_pair = solve(lower[s, :, 31]-upper[s, :, 0], a, b)
        for name, cost, result in [('upper', -upper[s, :, 0], hi),
                                   ('lower', lower[s, :, 31], lo),
                                   ('paired', lower[s, :, 31]-upper[s, :, 0], paired)]:
            evidence[f'{name}_{s}_primal'] = result.x
            evidence[f'{name}_{s}_dual'] = result.eqlin.marginals
            other, check = solve(cost, a, b, method='highs-ipm')
            difference = float(abs(other.fun-result.fun))
            assert difference < 1e-8
            cross_checks.append(dict(source_index=s, calculation=name,
                                     objective_difference=difference, **check))
        late_rows.append(dict(source_index=s, no_early_upper=-hi.fun,
                              full_early_lower=lo.fun, paired_advantage_lower=paired.fun,
                              maximum_LP_residual=max(max(x.values()) for x in [check_hi, check_lo, check_pair])))
    # Flatten masks in late-major, early-minor order, exactly as J_s(S,T).
    a10 = np.vstack([np.ones(1024), np.repeat(bits, 32, axis=1), np.tile(bits, (1, 32))])
    b10 = np.r_[1, late_probabilities, np.full(5, .65)]
    law = common_obstruction(late_probabilities)
    obstruction_values = upper.reshape(len(source), -1) @ law
    check_obstruction = dict(marginal_residual=float(np.max(abs(a10@law-b10))),
                             normalization_residual=float(abs(law.sum()-1)))
    assert max(check_obstruction.values()) < 1e-10
    event = np.logical_and(np.repeat(bits.sum(axis=0), 32) >= 2,
                           np.tile(bits.sum(axis=0), 32) >= 4).astype(float)
    repair_rows = []
    for s in range(len(source)):
        fit, residual = solve(lower[s].ravel(), a10, b10, -event[None, :], np.array([-.60]))
        evidence[f'repair_{s}_primal'] = fit.x
        evidence[f'repair_{s}_dual'] = np.r_[fit.eqlin.marginals, fit.ineqlin.marginals]
        other, check = solve(lower[s].ravel(), a10, b10, -event[None, :], np.array([-.60]), method='highs-ipm')
        difference = float(abs(other.fun-fit.fun))
        assert difference < 1e-8
        cross_checks.append(dict(source_index=s, calculation='repair',
                                 objective_difference=difference, **check))
        repair_rows.append(dict(source_index=s, repaired_lower=fit.fun,
                                joint_event_probability=float(event@fit.x),
                                maximum_LP_residual=max(residual.values())))
    assert law.min() >= -1e-9
    assert max(x['no_early_upper'] for x in late_rows) < .4441
    assert min(x['full_early_lower'] for x in late_rows) > .5866
    assert min(x['paired_advantage_lower'] for x in late_rows) > .1427
    assert abs(obstruction_values.max()-.480481020407) < 5.1e-13
    assert min(x['repaired_lower'] for x in repair_rows) > .5190
    if evidence_path is not None:
        np.savez_compressed(evidence_path, **evidence)
    return dict(late_dependence=late_rows, joint_repair=repair_rows,
                cross_algorithm_checks=cross_checks,
                common_obstruction_upper=float(obstruction_values.max()),
                common_obstruction_checks=check_obstruction,
                common_obstruction_statewise_upper=(upper.reshape(len(source), -1)@law).tolist(),
                common_obstruction_law=law.tolist())


def common_obstruction(late_probabilities):
    """Integrate one common uniform variable exactly over its mask intervals."""
    p = np.asarray(late_probabilities, dtype=float)
    if p.shape != (5,) or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected five finite probabilities.')
    cuts = np.unique(np.r_[0., .35, p, 1.])
    law = np.zeros(1024)
    for left, right in zip(cuts[:-1], cuts[1:]):
        u = (left+right)/2
        late = sum(1 << j for j in range(5) if u < p[j])
        early = 31 if u >= .35 else 0
        law[32*late+early] += right-left
    return law

