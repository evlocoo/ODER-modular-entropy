"""Reference-assisted discrimination for the declared five-share code.

Authorized sets transfer the logical qubit; unauthorized sets are constant
channels. The orthogonal flag records authorization and receiving failure.
"""
import numpy as np


def discrimination(authorized_probability, capture_probability=1.):
    q,p=authorized_probability,capture_probability
    if not 0<=q<=1 or not 0<=p<=1:
        raise ValueError('Probabilities must be physical.')
    success=p*q
    bell=np.array([1,0,0,1],complex)/np.sqrt(2)
    preserving=np.zeros((6,6),complex);erased=np.zeros_like(preserving)
    # Basis: Q tensor (logical labels 0,1 plus orthogonal absence flag).
    indices=[0,1,3,4]
    preserving[np.ix_(indices,indices)]=success*np.outer(bell,bell.conj())
    erased[np.ix_(indices,indices)]=success*np.eye(4)/4
    for state in [preserving,erased]:
        state[2,2]=state[5,5]=(1-success)/2
        assert abs(np.trace(state)-1)<1e-12
        assert np.linalg.eigvalsh(state).min()>=-1e-12
    difference=preserving-erased
    distance=float(np.sum(abs(np.linalg.eigvalsh(difference)))/2)
    score=(1+distance)/2
    memory_difference=np.trace(difference.reshape(2,3,2,3),axis1=0,axis2=2)
    assert np.linalg.norm(memory_difference)<1e-12
    assert abs(score-(.5+.375*p*q))<1e-12
    return dict(authorized_probability=q,capture_probability=p,
                reference_assisted_success=score,trace_distance=distance,
                memory_only_state_difference=float(np.linalg.norm(memory_difference)))
