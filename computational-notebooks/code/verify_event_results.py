"""Independently replay saved primal/dual witnesses without running an LP.

Run from the notebook directory:
python code/verify_event_results.py reference/observer
"""
import argparse
import json
from pathlib import Path
import numpy as np
from event_certificates import headroom_sort_key


def rank_from_witness_values(records, indices):
    """Recalculate raw headroom before applying the declared zero-tie rule."""
    return sorted(records, key=lambda x: headroom_sort_key(
        x['feasible_maximum']-max(x['sourcewise_thresholds'][i] for i in indices), x['event']))


def verify(directory, root):
    source = np.load(root/'inputs/hp_joint_information.npy')
    p = np.load(root/'inputs/closure_reproduction_inputs.npz')['observer_carriers'][-1]
    masks = [(d, e) for d in range(32) for e in range(32)]
    a = np.array([[1]+[(d >> j) & 1 for j in range(5)]+[(e >> j) & 1 for j in range(5)]
                  for d, e in masks], dtype=float).T
    b = np.r_[1., p, [.65]*5]
    costs = 2**(2*source.reshape(32, 1024)-2)
    maximum_residual = 0.
    checked = 0

    def check(c, law, dual, row=None, limit=None):
        nonlocal maximum_residual, checked
        reduced = c-a.T@dual[:11]
        dual_value = b@dual[:11]
        residuals = [np.max(abs(a@law-b)), max(0., -law.min())]
        if row is not None:
            reduced -= row*dual[11]
            dual_value += limit*dual[11]
            residuals.extend([max(0., row@law-limit), max(0., dual[11])])
        residuals.extend([max(0., -reduced.min()), abs(c@law-dual_value)])
        assert np.all(np.isfinite(law)) and np.all(np.isfinite(dual))
        maximum_residual = max(maximum_residual, *residuals)
        assert max(residuals) <= 1e-8, residuals
        checked += 1

    results = json.loads((directory/'count_event_frontiers.json').read_text(encoding='utf-8'))
    for result in results:
        eps = result['flip_probability']
        # Enumerate individual observed flags independently of the binomial implementation.
        transition = np.zeros((6, 6))
        for true_count in range(6):
            true_mask = (1 << true_count)-1
            for observed in range(32):
                flips = (true_mask ^ observed).bit_count()
                transition[true_count, observed.bit_count()] += eps**flips*(1-eps)**(5-flips)
        arrays = np.load(directory/f'event_witnesses_{eps:.2f}.npz')
        assert len(result['catalog']) == 34
        for event in result['catalog']:
            name = event['event']; indicator = np.array(event['indicator_by_counts'])
            table = transition@indicator@transition.T
            score = np.array([table[d.bit_count(), e.bit_count()] for d, e in masks])
            law = arrays[name+'_maximum_law']
            check(-score, law, arrays[name+'_maximum_dual'])
            assert abs(score@law-event['feasible_maximum']) < 1e-10
            for i, cost in enumerate(costs):
                law = arrays[name+'_frontier_laws'][i]
                check(-score, law, arrays[name+'_frontier_duals'][i], cost, .5)
                assert abs(score@law-event['sourcewise_thresholds'][i]) < 1e-10
            for i, law in enumerate(arrays[name+'_direct_laws']):
                check(costs[i], law, arrays[name+'_direct_duals'][i], -score, -event['guarded_rate'])
                assert costs[i]@law > .5
                assert abs(costs[i]@law-event['direct_lower_certificates'][i]) < 1e-10
        for split, indices in [('all', range(32)), ('even', range(0, 32, 2)), ('odd', range(1, 32, 2))]:
            ranking = rank_from_witness_values(result['catalog'], indices)
            assert [x['event'] for x in ranking] == result['rankings'][split]
            assert ranking[0]['event'] == 'E7'
    return dict(primal_dual_witnesses_checked=checked, maximum_raw_residual=maximum_residual,
                tolerance=1e-8, passed=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, Path(__file__).resolve().parents[1]), indent=2))
