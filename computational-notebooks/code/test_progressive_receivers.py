"""Channel, timing and quadrature checks for recorded count-gate receivers."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigvalsh

from progressive_receivers import (RecordedCollection, capture_probability,
    count_tail, finite_chain_floor, gate_weights, recorded_cdf, statewise_scores)


ROOT = Path(__file__).resolve().parents[1]


def entropy(rho):
    values = eigvalsh(rho)
    values = values[values > 1e-14]
    return float(-np.sum(values*np.log2(values)))


def information(rho, q_dimension=2):
    m_dimension = len(rho)//q_dimension
    tensor = rho.reshape(q_dimension, m_dimension, q_dimension, m_dimension)
    q = np.trace(tensor, axis1=1, axis2=3)
    m = np.trace(tensor, axis1=0, axis2=2)
    return (entropy(q)+entropy(m)-entropy(rho))/2


def erasure_memory(payload, probability):
    rho = np.zeros((6, 6), complex)
    data = [0, 1, 3, 4]
    rho[np.ix_(data, data)] = probability*payload
    rho[2, 2] = rho[5, 5] = (1-probability)/2
    return rho


def linear_record(probabilities, times=(0., 1.)):
    times = np.asarray(times, float)
    end = np.asarray(probabilities, float)
    cdf = (times[:, None]-times[0])/(times[-1]-times[0])*end
    return RecordedCollection(times, cdf, cdf.copy(), np.ones_like(cdf), 0.)


class RecordedReceiverTests(unittest.TestCase):
    def test_instrument_cdf_and_validation(self):
        times = np.linspace(0., 2., 9)
        dose = times[:, None]*np.array([[.1, .3]])
        cavity = np.full(dose.shape, .02j)
        run = dict(times=times, observer_carriers=dose+abs(cavity)**2, cavities=cavity)
        record = recorded_cdf(run, epsilon=.02)
        np.testing.assert_allclose(record.cdf, dose**2/(dose+.02), atol=1e-15)
        np.testing.assert_allclose(record.acceptance_beta, 1-(.02/(.02+dose))**2)
        h = 1e-6
        middle = .2
        derivative = (((middle+h)**2/(middle+h+.02))-
                      ((middle-h)**2/(middle-h+.02)))/(2*h)
        self.assertAlmostEqual(derivative, 1-(.02/(.02+middle))**2, places=10)
        bad = dict(run, observer_carriers=run["observer_carriers"].copy())
        bad["observer_carriers"][5, 0] = bad["observer_carriers"][4, 0]-.01
        with self.assertRaises(ValueError):
            recorded_cdf(bad)
        with self.assertRaises(ValueError):
            recorded_cdf(dict(run, times=times[::-1]))
        with self.assertRaises(ValueError):
            recorded_cdf(dict(run, cavities=cavity[:, :1]))
        with self.assertRaises(ValueError):
            recorded_cdf(run, epsilon=0.)

    def test_gate_density_integrates_exact_bin_probability(self):
        record = linear_record([.1, .3, .5, .6, .8], times=(0., .19, .62, 1.))
        for gate in range(1, 6):
            result = gate_weights(record, gate, deadline=1., mode="common_constant", order=4)
            self.assertLess(result["gate_mass_residual"], 2e-14)
            self.assertAlmostEqual(result["gate_law"].sum(),
                                   count_tail(record.cdf[-1], gate), places=13)
            masks = np.flatnonzero(result["gate_law"])
            self.assertTrue(all(int(mask).bit_count() == gate for mask in masks))

    def test_one_uniform_arrival_has_closed_capture_integral(self):
        rate = 1.7
        result = gate_weights(linear_record([1.]), 1, deadline=1.,
                              mode="common_constant", rate=rate, order=12)
        expected = 1-(-np.expm1(-rate))/rate
        self.assertAlmostEqual(result["weights"][1], expected, places=13)

    def test_cutoff_and_loading_split_inside_a_time_bin(self):
        record = linear_record([.7], times=(0., 120.))
        result = gate_weights(record, 1, loading_delay=2.13, order=12)
        expected = quad(lambda time: .7/120*finite_chain_floor(120-2.13-time),
                        0., 120-2.13, points=[120-2.13-57.15], epsabs=1e-12)[0]
        self.assertAlmostEqual(result["weights"][1], expected, places=10)
        ages = np.linspace(0., 120., 1001)
        self.assertGreaterEqual(np.diff(finite_chain_floor(ages)).min(), -1e-13)
        self.assertAlmostEqual(float(finite_chain_floor(80.)),
                               float(finite_chain_floor(120.)), places=14)

    def test_late_or_missing_arrival_and_invalid_parameters(self):
        record = linear_record([0., 0.])
        result = gate_weights(record, 1, deadline=1., loading_delay=0., mode="common_constant")
        self.assertEqual(result["captured_probability"], 0.)
        result = gate_weights(linear_record([1.]), 1, deadline=1., loading_delay=1.)
        self.assertEqual(result["gate_probability"], 0.)
        with self.assertRaises(ValueError):
            gate_weights(record, 0, deadline=1.)
        with self.assertRaises(ValueError):
            gate_weights(record, 1, deadline=2.)
        with self.assertRaises(ValueError):
            gate_weights(record, 1, deadline=1., loading_delay=-1.)
        with self.assertRaises(ValueError):
            capture_probability(1., "unrecognized")

    def test_sourcewise_minimum_is_after_mask_average(self):
        # The two sources prefer different masks. Minima do not commute.
        source = np.array([[0., 1., .2, 1.], [0., .2, 1., 1.]])
        weights = np.array([0., .5, .5, 0.])
        _, fidelity = statewise_scores(source, weights)
        branch_lower = np.exp2(2*source-2)
        self.assertGreater(fidelity.min(), weights@branch_lower.min(axis=0))
        with self.assertRaises(ValueError):
            statewise_scores(source+2., weights)

    def test_explicit_recorded_channel_and_common_clock_law(self):
        vectors = [np.array(x)/np.sqrt(2) for x in
                   ([1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0])]
        bells = [np.outer(v, v) for v in vectors]
        payloads = [.75*bells[0]+.25*bells[1], .75*bells[0]+.25*bells[2]]
        weights, gates, rate = [.4, .6], [.2, .7], 3.
        js = [1-entropy(payload)/2 for payload in payloads]
        def retained(time):
            # Start in record,Q,M order and permute to Q,record,M order.
            rho = np.zeros((12, 12), complex)
            for i, (w, gate, payload) in enumerate(zip(weights, gates, payloads)):
                probability = float(capture_probability(time-gate, "common_constant", rate=rate))
                rho[6*i:6*(i+1), 6*i:6*(i+1)] = w*erasure_memory(payload, probability)
            rho = rho.reshape(2, 2, 3, 2, 2, 3).transpose(1, 0, 2, 4, 3, 5).reshape(12, 12)
            return information(rho)
        time, dt = 1.4, 1e-5
        expected = sum(w*(-np.expm1(-rate*(time-gate)))*j
                       for w, gate, j in zip(weights, gates, js))
        self.assertAlmostEqual(retained(time), expected, places=12)
        derivative = (retained(time+dt)-retained(time-dt))/(2*dt)
        self.assertAlmostEqual(derivative, rate*(np.dot(weights, js)-retained(time)), places=8)
        # Deleting the record can lose reference information even at full capture.
        unlabeled = 1-entropy(.5*payloads[0]+.5*payloads[1])/2
        self.assertAlmostEqual(js[0]-unlabeled, .125, places=12)

    def test_reset_age_requires_the_cohort_distribution(self):
        # Two equally weighted cohorts admitted at times 0 and 1, hazard 2*age.
        # At t=1, one age is 1 and the other is 0.
        actual_derivative = np.exp(-1.)
        retained = .5*(1-np.exp(-1.))
        common_absolute_clock_prediction = 2*(1-retained)
        self.assertAlmostEqual(common_absolute_clock_prediction-actual_derivative, 1., places=13)

    def test_frozen_detector_quadrature_convergence(self):
        run = dict(np.load(ROOT/"reference/upstream/baseline.npz"))
        record = recorded_cdf(run)
        for mode in ("common_constant", "finite_chain_reset"):
            a = gate_weights(record, 2, loading_delay=2., mode=mode, order=8)
            b = gate_weights(record, 2, loading_delay=2., mode=mode, order=4)
            self.assertLess(np.max(abs(a["weights"]-b["weights"])), 1e-10)
            self.assertLess(a["gate_mass_residual"], 1e-11)


class ProgressiveReplayTests(unittest.TestCase):
    def test_saved_score_must_be_finite(self):
        command = [sys.executable, str(ROOT/'code/verify_progressive_results.py')]
        clean = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(clean.returncode, 0, clean.stderr)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['inputs/hp_joint_information.npy', 'reference/progressive/summary.csv']:
                (root/name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/name, root/name)
            shutil.copytree(ROOT/'reference/upstream', root/'reference/upstream')
            name = 'reference/progressive/statewise_scores.npz'
            with np.load(ROOT/name) as original:
                scores = dict(original)
            key = 'baseline__gate1__load0__finite_chain_reset__fidelity_lower'
            scores[key][0] = np.nan
            np.savez(root/name, **scores)
            bad = subprocess.run(command + ['--root', folder], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('Nonfinite saved array ' + key, bad.stderr)


if __name__ == "__main__":
    unittest.main()
