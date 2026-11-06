"""Check the printed minimax construction and reject corrupted certificates."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from minimax_certificates import constraints,seven_outcome_law,evaluate,PRINTED_Y
from verify_minimax_results import verify

ROOT=Path(__file__).resolve().parents[1]


class MinimaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=tempfile.TemporaryDirectory();cls.output=Path(cls.folder.name)
        cls.p=np.load(ROOT/'inputs/closure_reproduction_inputs.npz')['observer_carriers'][-1]
        cls.source=np.load(ROOT/'inputs/hp_joint_information.npy')
        cls.result=evaluate(cls.source,cls.p,cls.output/'minimax_witnesses.npz')
        (cls.output/'minimax_obstruction.json').write_text(json.dumps(cls.result),encoding='utf-8')

    @classmethod
    def tearDownClass(cls):cls.folder.cleanup()

    def test_explicit_law_marginals_and_support(self):
        a,b=constraints(self.p);w=seven_outcome_law(self.p)
        self.assertEqual(np.flatnonzero(w).tolist(),sorted([31,1023]+[32*(31^(1<<j)) for j in range(5)]))
        self.assertGreaterEqual(w.min(),0.)
        np.testing.assert_allclose(a@w,b,rtol=0,atol=5e-16)

    def test_printed_value_and_both_solver_pairs(self):
        self.assertAlmostEqual(self.result['upper_certificate'],.4469473873729414,places=13)
        self.assertEqual(self.result['limiting_source_ordinal'],11)
        self.assertEqual([x['method'] for x in self.result['solvers']],['highs-ds','highs-ipm'])
        self.assertTrue(verify(self.output,ROOT)['passed'])
        with np.load(self.output/'minimax_witnesses.npz') as z:np.testing.assert_array_equal(z['printed_y'],PRINTED_Y)

    def test_corrupted_primal_and_dual_witnesses_fail(self):
        with np.load(self.output/'minimax_witnesses.npz') as z:original={k:z[k] for k in z.files}
        for key in ['printed_law','printed_y','printed_lambda']:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as name:
                folder=Path(name);data={k:v.copy() for k,v in original.items()};data[key].flat[0]+=.01
                np.savez(folder/'minimax_witnesses.npz',**data)
                (folder/'minimax_obstruction.json').write_text(json.dumps(self.result),encoding='utf-8')
                with self.assertRaises(AssertionError):verify(folder,ROOT)

    def test_invalid_marginals_rejected(self):
        for p in ([.4]*4,[np.nan]*5,[-.1]*5,[1.1]*5,[0.]*5):
            with self.subTest(p=p),self.assertRaises(ValueError):seven_outcome_law(p)
