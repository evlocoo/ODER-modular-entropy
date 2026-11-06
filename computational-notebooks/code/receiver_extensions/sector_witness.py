"""Two-photon amplifier-sector extraction; achieved fidelity, not global optimum."""
import numpy as np

def calculate():
    I=np.eye(2)
    paulis=[I,np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
    A0=np.array([[np.sqrt(2),0],[0,1],[0,0]])/np.sqrt(3)
    A1=np.array([[0,0],[1,0],[0,np.sqrt(2)]])/np.sqrt(3)
    W=np.array([[1,0,0],[0,1/np.sqrt(2),0],[0,1/np.sqrt(2),0],[0,0,1]])
    # Kraus operators of partial trace of second symmetric-qubit factor.
    D=[np.kron(I,np.array([[1,0]]))@W,np.kron(I,np.array([[0,1]]))@W]
    K=[d@a for d in D for a in [A0,A1]]
    assert np.allclose(sum(a.conj().T@a for a in [A0,A1]),I)
    assert np.allclose(sum(d.conj().T@d for d in D),np.eye(3))
    assert np.allclose(sum(k.conj().T@k for k in K),I)
    maxres=0.
    for i in range(2):
     for j in range(2):
      E=np.zeros((2,2)); E[i,j]=1
      actual=sum(k@E@k.conj().T for k in K)
      expected=2*E/3+np.trace(E)*I/6
      maxres=max(maxres,float(np.max(np.abs(actual-expected))))
    F2=sum(abs(np.trace(k))**2 for k in K)/4
    assert abs(F2-.75)<1e-14
    res=[]
    for G in [1,1.1,1.5,2,3,10]:
     p1=G**-3;p2=3*(G-1)/G**4
     F_coarse=.25+.75*p1
     F_fine=p1+.75*p2+.25*(1-p1-p2)
     weights=[p1+.75*p2,p2/12,p2/12,p2/12,1-p1-p2]
     assert abs(sum(weights)-1)<1e-14
     assert abs(F_fine-F_coarse-p2/2)<1e-14
     res.append(dict(G=G,p_one=p1,p_two=p2,coarse_F=F_coarse,sector_two_F=F_fine,gain=F_fine-F_coarse,pauli_erasure_weights=weights))
    out=dict(matrix_residual=maxres,two_photon_conditional_F=F2,rows=res)
    return out
