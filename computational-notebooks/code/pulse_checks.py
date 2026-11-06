from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad


def pulse_checks():
    def f(t): return 1/(np.sqrt(2)*(1+1j*t)**1.5)
    def integral(fun,L):
        return quad(lambda t: float(np.real(fun(t))),-L,L,epsabs=1e-11,limit=300)[0]+1j*quad(lambda t: float(np.imag(fun(t))),-L,L,epsabs=1e-11,limit=300)[0]
    rows=[]
    for L in [10.,100.]:
        delta=1-L/np.sqrt(1+L*L)
        for u in [0.,1.,2.,4.]:
            def fu(t): return np.exp(u/2)*f(np.exp(u)*t)
            def dfu(t):
                z=1j*np.exp(u)*t
                return fu(t)*(.5-1.5*z/(1+z))
            overlap=integral(lambda t: np.conj(f(t))*fu(t),L)
            derivative=integral(lambda t: np.conj(f(t))*dfu(t),L)
            x=np.exp(u)*L
            q=x/np.sqrt(1+x*x)
            dq=q/(1+x*x)
            actual=q-abs(overlap)**2
            dr=dq-2*np.real(np.conj(overlap)*derivative)
            target=1-np.cosh(u/2)**(-4)
            activation=dr/(2*(1-actual))
            floor=np.cosh(u/2)**(-4)
            abound=6*delta/(floor-3*delta) if floor>3*delta else None
            err=abs(actual-target)
            aerr=abs(activation-np.tanh(u/2))
            assert err <= 3*delta+1e-10
            if abound is not None: assert aerr <= abound+1e-9
            rows.append(dict(L=L,u=u,delta=delta,retained_fraction=float(actual),
              target_retained_fraction=float(target),retained_error=float(err),
              uniform_retained_error_bound=3*delta,normalized_parameter_hazard=float(activation),
              target_activation=float(np.tanh(u/2)),activation_error=float(aerr),
              uniform_activation_bound_to_u=abound))
    out=dict(model='h=1, ell=1, rectangular acceptance, effective Markov input-output',
      warning='No microscopic relativistic field-device approximation error included.',rows=rows)
    return out
