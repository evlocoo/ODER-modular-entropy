"""Exact matched-calibration witness with independent identical Gaussian rails."""
from fractions import Fraction as F
import numpy as np
from .gaussian_dualrail import gaussian_dualrail

def contraction(w,axis):
    return w['I']+w[axis]-sum(w[a] for a in ('X','Y','Z') if a != axis)

def calculate():
    t=F(24688,16875); q=F(1616,3375); V=F(5,6)
    disc=float(t*t-4*q*q)
    x2=(float(t)+np.sqrt(disc))/2;y2=(float(t)-np.sqrt(disc))/2
    A=gaussian_dualrail(np.eye(2)/np.sqrt(2),np.eye(2)/2)
    B=gaussian_dualrail(np.diag(np.sqrt([x2,y2])),np.diag([float(V)-x2/2,float(V)-y2/2]))
    expectedA={'I':F(152,625),'X':F(24,625),'Y':F(24,625),'Z':F(24,625)}
    expectedB={'I':F(152,625),'X':F(2069,40000),'Y':F(47,4000),'Z':F(2069,40000)}
    for C,expected in [(A,expectedA),(B,expectedB)]:
        for k,v in expected.items():assert abs(C['pauli_weights'][k]-float(v))<2e-15
        assert abs(C['survival']-float(F(224,625)))<2e-15
        assert abs(C['deterministic_bell_fidelity']-float(F(1009,2500)))<2e-15
        w=C['pauli_weights'];cx=contraction(w,'X');cy=contraction(w,'Y');cz=contraction(w,'Z');p=C['survival']
        reconstructed={'I':(p+2*cx+cy)/4,'X':(p-cy)/4,'Y':(p-2*cx+cy)/4,'Z':(p-cy)/4}
        assert max(abs(reconstructed[k]-w[k]) for k in w)<1e-15
        assert abs(cx-cz)<1e-15
        C['signed_correlations']={'X':cx,'Y':cy,'Z':cz}
        C['optimal_binary_success']={a:(1+abs(contraction(w,a)))/2 for a in ('X','Y','Z')}
    cp_surplus=V**2-V*t/2+q*q/4-(1-q)**2/4
    assert cp_surplus>0 and min(float(V)-x2/2,float(V)-y2/2)>0
    out={'construction':'Exact rational invariants with positive diagonal X',
         'B_invariants':{'x2_plus_y2':str(t),'xy':str(q),'vacuum_variance':str(V),'cp_determinant_surplus':str(cp_surplus)},
         'A':A,'B':B,'matching_residual':max(abs(A[k]-B[k]) for k in ('survival','deterministic_bell_fidelity','unrecorded_pauli_error'))}
    return out
