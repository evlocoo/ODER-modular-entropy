"""Regression tests for ordered legs, structural support, and null reductions."""
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from notebook_support import validation as v


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pair,_=v.load_pair(Path(__file__).resolve().parents[1])

    def test_periodic_operator_keeps_ordered_legs(self):
        actual=v.embed(np.kron(v.PAULI['X'],v.PAULI['Y']),['B7','B0'],v.BOUNDARY)
        expected=np.kron(np.kron(v.PAULI['Y'],np.eye(64)),v.PAULI['X'])
        np.testing.assert_array_equal(actual,expected)

    def test_ascent_does_not_build_encoding_or_prune_zero_operator(self):
        with patch.object(v,'graph_encoding',side_effect=AssertionError('Full encoding forbidden')):
            zero,steps=v.ascend(self.pair['mera'],np.zeros((2,2)),['B0'])
            nonzero,other=v.ascend(self.pair['mera'],v.PAULI['Y'],['B0'])
        np.testing.assert_array_equal(zero,np.zeros((2,2)))
        self.assertEqual(steps,other)
        self.assertTrue(any(x['action'].startswith('canceled') for x in steps))
        self.assertEqual(steps[-1]['support_after'],['A'])

    def test_periodic_ascent_against_dense_product(self):
        for model,tensors in self.pair.items():
            logical,_=v.ascend(tensors,np.kron(v.PAULI['X'],v.PAULI['Y']),['B7','B0'],model=='tree')
            encoding=v.graph_encoding(tensors)
            full=np.kron(np.kron(v.PAULI['Y'],np.eye(64)),v.PAULI['X'])
            self.assertLess(np.linalg.norm(logical-encoding.conj().T@full@encoding),v.TOL)

    def test_tree_refuses_nonidentity_gate(self):
        with self.assertRaises(AssertionError):
            v.ascend(self.pair['mera'],v.PAULI['X'],['B0'],tree=True)

    def test_mixed_reduction_bell_and_ordered_keep(self):
        bell=np.array([1,0,0,1])/np.sqrt(2);rho=np.outer(bell,bell)
        np.testing.assert_allclose(v.mixed_reduce(rho,[0],2),np.eye(2)/2,atol=1e-15,rtol=0)
        np.testing.assert_allclose(v.mixed_reduce(rho,[],2),[[1]],atol=1e-15,rtol=0)
        product=np.kron(np.diag([.2,.8]),np.diag([.7,.3]))
        np.testing.assert_array_equal(v.mixed_reduce(product,[1,0],2),np.kron(np.diag([.7,.3]),np.diag([.2,.8])))


if __name__=='__main__':
    unittest.main()
