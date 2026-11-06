"""Checks of event definitions, noisy readout, and the common collection law."""
import unittest
import numpy as np
from event_certificates import catalog, count_channel, constraints, checked_lp, headroom_sort_key, TOL
from verify_event_results import rank_from_witness_values
from joint_certificates import common_obstruction


class EventCertificates(unittest.TestCase):
    def test_zero_headroom_perturbations_preserve_both_rankings(self):
        import copy
        for tiny in [(0., 0., 0.), (3e-12, -8e-13, 7e-12), (-5e-10, 9e-10, -2e-10)]:
            records=[dict(event=name, feasible_maximum=.5,
                          sourcewise_thresholds=[.5-headroom]*4,
                          headroom=headroom)
                     for name,headroom in zip(['Z','A','M','E7'], [*tiny,.34])]
            before=copy.deepcopy(records)
            search_order=sorted(records,key=lambda x:headroom_sort_key(x['headroom'],x['event']))
            self.assertEqual([x['event'] for x in search_order],['E7','A','M','Z'])
            for indices in [range(4),range(0,4,2),range(1,4,2)]:
                checked_order=rank_from_witness_values(records,indices)
                self.assertEqual([x['event'] for x in checked_order],['E7','A','M','Z'])
            self.assertEqual(records,before)

    def test_tie_boundary_and_resolved_values(self):
        self.assertEqual(headroom_sort_key(TOL,'A'),headroom_sort_key(-TOL,'A'))
        self.assertLess(headroom_sort_key(1.01*TOL,'Z'),headroom_sort_key(0.,'A'))
        self.assertLess(headroom_sort_key(0.,'Z'),headroom_sort_key(-1.01*TOL,'A'))
        self.assertLess(headroom_sort_key(.34+1e-12,'Z'),headroom_sort_key(.34,'A'))
        for value in [float('nan'),float('inf'),-float('inf')]:
            with self.assertRaises(ValueError):headroom_sort_key(value,'A')

    def test_catalog_is_the_declared_union(self):
        actual = {tuple(x.ravel()) for _, x in catalog()}
        expected = set()
        for a in range(1, 6):
            for b in range(1, 6):
                expected.add(tuple(int(d >= a and e >= b) for d in range(6) for e in range(6)))
        for k in range(1, 11):
            expected.add(tuple(int(d+e >= k) for d in range(6) for e in range(6)))
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 34)

    def test_count_channel_against_individual_bit_enumeration(self):
        for epsilon in [0., .02, .5]:
            channel = count_channel(epsilon)
            for true in range(32):
                row = np.zeros(6)
                for observed in range(32):
                    distance = (true ^ observed).bit_count()
                    row[observed.bit_count()] += epsilon**distance*(1-epsilon)**(5-distance)
                np.testing.assert_allclose(channel[true.bit_count()], row, rtol=0, atol=1e-14)

    def test_obstruction_marginals_and_shared_early_flags(self):
        p = np.array([.12, .28, .41, .63, .87])
        law = common_obstruction(p)
        a, b, _ = constraints(p)
        np.testing.assert_allclose(a@law, b, rtol=0, atol=1e-14)
        self.assertEqual(np.count_nonzero(law), 7)
        self.assertTrue(set(np.flatnonzero(law) % 32) <= {0, 31})
        self.assertTrue(np.all(law >= 0))
        self.assertAlmostEqual(law[31::32].sum(), .65)

    def test_obstruction_boundary_coincidences(self):
        p = np.array([0., .35, .35, 1., 1.])
        law = common_obstruction(p)
        a, b, _ = constraints(p)
        np.testing.assert_allclose(a@law, b, rtol=0, atol=1e-14)
        for invalid in [[np.nan]*5, [-.1]*5, [1.1]*5, [0.]*4]:
            with self.assertRaises(ValueError):
                common_obstruction(invalid)

    def test_lp_checks_infeasibility_and_known_objective(self):
        a = np.ones((1, 2)); b = np.ones(1)
        fit, residual = checked_lp(np.array([1., 2.]), a, b)
        self.assertAlmostEqual(fit.fun, 1.)
        self.assertLess(max(residual.values()), 1e-10)
        with self.assertRaises(RuntimeError):
            checked_lp(np.ones(2), a, -b)


if __name__ == '__main__':
    unittest.main()
