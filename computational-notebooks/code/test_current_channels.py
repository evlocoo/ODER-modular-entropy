"""Independent invariants for current receiver, noise, and preparation results."""
import unittest
import numpy as np
from receiver_storage import (validate, survival, terminal_bounds, noisy_protocol,
                              erasure_kraus, channel_checks)
from preparation_variance import checks, transfer
from recovery_certificates import binary_bound
from reference_verification import discrimination


class CurrentChannels(unittest.TestCase):
    def test_pulse_evolution_and_three_site_refinement(self):
        result = validate()
        self.assertAlmostEqual(result['pulse_area'], np.pi/2, places=13)

    def test_erasure_channel_preserves_arbitrary_coherence(self):
        self.assertEqual(len(channel_checks()), 4)
        for p in [0., .2, .8, 1.]:
            k = erasure_kraus(p, dimension=3)
            np.testing.assert_allclose(sum(x.conj().T@x for x in k), np.eye(3), atol=1e-14)

    def test_loss_limits(self):
        self.assertEqual(survival(0, 0, 12), 1.)
        self.assertEqual(survival(.1, 0, 0), survival(.1, 0, 100))
        self.assertLess(survival(.1, 3, .4), survival(.1, .05, .4))
        self.assertLess(.25+.75*survival(.1, 3, .4), .5)
        with self.assertRaises(ValueError):
            survival(-.1)

    def test_affine_fidelity_endpoints(self):
        self.assertEqual(terminal_bounds(0, .8472, .3306)['full_early_lower'], .25)
        self.assertEqual(terminal_bounds(1, .8472, .3306)['full_early_lower'], .8472)

    def test_branch_probabilities_and_reversal(self):
        for s in [0., .3, 5/6, np.exp(-.12), np.exp(-.30), 1.]:
            a, law_a = noisy_protocol(s, 'A')
            b, law_b = noisy_protocol(s, 'B')
            self.assertAlmostEqual(a['individual_terminal_efficiency'], b['individual_terminal_efficiency'])
            self.assertAlmostEqual(a['expected_active_links'], b['expected_active_links'])
            self.assertEqual(a['expected_integrated_g2_over_Omega'],b['expected_integrated_g2_over_Omega'])
        a, _ = noisy_protocol(1, 'A');b, _ = noisy_protocol(1, 'B')
        self.assertLess(a['bell_fidelity'], b['bell_fidelity'])

    def test_fixed_variance_and_overlap_integrals(self):
        self.assertEqual(len(checks()), 36)
        self.assertEqual(float(transfer(0, 1, .25)), 0.)

    def test_binary_bound_endpoints(self):
        self.assertEqual(binary_bound(0), .25)
        self.assertEqual(binary_bound(1), 1.)
        with self.assertRaises(ValueError):
            binary_bound(-.01)

    def test_reference_assisted_discrimination(self):
        for q in [0.,.2,.6,1.]:
            for p in [0.,.3,1.]:
                result=discrimination(q,p)
                self.assertAlmostEqual(result['reference_assisted_success'],.5+.375*p*q)
        self.assertEqual(discrimination(0)['reference_assisted_success'],.5)


if __name__ == '__main__':
    unittest.main()
