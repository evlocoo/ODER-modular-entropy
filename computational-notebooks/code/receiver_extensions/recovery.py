"""Constructive 32-source bounds for independent Pauli-plus-erasure carriers."""
import numpy as np
from scipy.optimize import brentq

def ceiling(j):
    if j<1e-12:return .25
    if j>1-1e-12:return 1.
    def residual(f):
        return f*np.log2(4*f)+(1-f)*np.log2((1-f)/.75)-2*j
    return brentq(residual,.25,1-1e-15,xtol=1e-14)

def certify(channels,source):
    p=np.array([c['survival'] for c in channels]);i=np.array([c['pauli_weights']['I'] for c in channels])
    erasure=1-p;noise=p-i
    masks=(np.arange(32)[:,None]//2**np.arange(5)[None,:])%2
    lossweights=np.prod(np.where(masks,p,erasure),axis=1)
    goodweights=np.prod(np.where(masks,i,erasure),axis=1)
    assert abs(lossweights.sum()-1)<1e-8
    assert abs(goodweights.sum()-np.prod(1-noise))<1e-8
    output={'survival':p.tolist(),'identity_weights':i.tolist(),'unrecorded_pauli_errors':noise.tolist(),
            'any_error_probability':float(1-np.prod(1-noise)),'all_no_error_probability':float(goodweights.sum())}
    for e,label in [(31,'full_early'),(0,'no_early')]:
        table=source[:,:,e];lower=2**(2*table-2)
        upper=np.array([[ceiling(float(x)) for x in row] for row in table])
        # A decoder chosen for the ideal erased branch guarantees its KRS value
        # on every identity/erasure history; other histories contribute >=0.
        goodlower=lower@goodweights
        subtractlower=lower@lossweights-(1-goodweights.sum())
        # Ignoring the apparatus and outputting I/2 always attains 1/4.
        lo=np.maximum(.25,np.maximum(goodlower,subtractlower))
        hi=upper@lossweights
        assert np.all(lo<=hi+1e-8)
        output[label]={'lower_min':float(lo.min()),'lower_max':float(lo.max()),
                       'upper_min':float(hi.min()),'upper_max':float(hi.max()),
                       'statewise_lower':lo.tolist(),'statewise_upper':hi.tolist(),
                       'good_history_lower_min':float(goodlower.min())}
    return output

