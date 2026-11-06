"""Frozen-packet test of an explicitly effective chiral field channel.

This is NOT a local 3+1 Schwarzschild scalar detector calculation.  It
assumes a one-way, vacuum, number-preserving flux channel beyond a source
worldtube inside the chosen minimum radius; the upstream transmitted
packets are frozen. Retarded-time reparametrization is implemented
unitarily on temporal modes. Both dual rails have the same transformation.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from .paths import DATA as INPUT_DIRECTORY
DATA=INPUT_DIRECTORY/'frozen_inputs.npz'
INPUT=np.load(DATA)

def mask_weights(probabilities):
    # Same mask convention as the manuscript companion: jth bit labels carrier j.
    return np.array([np.prod([p if mask&(1<<j) else 1-p for j,p in enumerate(probabilities)]) for mask in range(32)])

def binary_bound(j):
    if j <= 0: return .25
    if j >= 1: return 1.
    def divergence(f):
        return f*np.log2(f/.25)+(1-f)*np.log2((1-f)/.75) if f<1 else 2.
    return brentq(lambda f:divergence(f)-2*j,.25,1.,xtol=5e-15)
Z=np.sqrt(.9); ETA=.6336; KAPPA=.515; OMEGA=.16/Z; Y0=np.sqrt(10.)
SOURCE=INPUT['source']
LOWER=np.exp2(2*SOURCE-2)
UPPER={e:np.array([[.25]+[binary_bound(float(v)) for v in row[1:,e]] for row in SOURCE]) for e in [0,31]}

def F(y): return y**3/3+y*y/2+y+np.log(y-1)
def tau_end(r): return 4/3*(Y0**3-(r/2)**1.5)
def geometry(t):
 y=(Y0**3-.75*np.asarray(t))**(1/3)
 q=4*Z*(F(Y0)-F(y))
 qp=Z*y/(y-1)
 return q,qp,2*y*y

def scores(p):
 w=mask_weights(p)
 out={}
 for e,name in [(31,'full_early'),(0,'no_early')]:
  val=SOURCE[:,:,e]@w;lo=LOWER[:,:,e]@w;hi=UPPER[e]@w
  out[name]={'information_min':float(val.min()),'information_mean':float(val.mean()),'information_max':float(val.max()),'bell_lower_min':float(lo.min()),'bell_lower_max':float(lo.max()),'bell_upper_min':float(hi.min()),'bell_upper_max':float(hi.max())}
 return out

def simulate(envelope, duration, mode='static', step=.05):
 # State: cavity a[5], incoming norm G[5], detector norm D[5], reflection R[5].
 def rhs(t,y):
  if mode=='static': q=t; qp=1.
  else: q,qp,_=geometry(t)
  u=np.sqrt(qp)*envelope(q)
  mult=qp if mode=='adaptive' else 1.
  k=KAPPA*mult;omega=OMEGA*mult; a=y[:5]
  v=np.sqrt(ETA)*u
  da=-(k/2+1j*omega)*a+np.sqrt(k/2)*v
  reflected=v-np.sqrt(k/2)*a
  return np.concatenate([da,abs(u)**2,k/2*abs(a)**2,abs(reflected)**2])
 times=np.linspace(0,duration,max(1001,int(np.ceil(duration/.025))+1))
 sol=solve_ivp(rhs,(0,duration),np.zeros(20,dtype=complex),method='DOP853',rtol=2e-11,atol=2e-13,max_step=step,t_eval=times)
 assert sol.success
 A=sol.y[:5];G=sol.y[5:10].real;D=sol.y[10:15].real;R=sol.y[15:20].real
 P=abs(A)**2+D
 assert P.min()>-1e-10 and P.max()<1+1e-10
 balance=np.max(abs(ETA*G-P-R))
 out={'proper_duration':float(duration),'mode':mode,'p':P[:,-1].tolist(),'G':G[:,-1].tolist(),'cavity_population':(abs(A[:,-1])**2).tolist(),'detector_output':D[:,-1].tolist(),'reflected':R[:,-1].tolist(),'max_balance_error':float(balance),'scores':scores(P[:,-1]),'broadband_ideal_p':(ETA*G[:,-1]).tolist(),'broadband_scores':scores(ETA*G[:,-1])}
 return out

def run(grid='baseline',step=.05):
 envelope=CubicSpline(INPUT['times_'+grid],INPUT['envelopes_'+grid],axis=0,extrapolate=False)
 out={'input_grid':grid,'max_solver_step':step,'model':'effective outgoing chiral, number-preserving vacuum channel; fixed upstream packet; not full scalar-field local detector','eta':ETA,'kappa_static':KAPPA,'omega_static':OMEGA,'cases':{}}
 for r in [6.,4.]:
  T=tau_end(r);Q,QP,_=geometry(T)
  cases={k:simulate(envelope,d,mode,step) for k,d,mode in [('infall_fixed',T,'fixed'),('infall_adaptive',T,'adaptive'),('static_same_retarded_support',Q,'static'),('static_same_proper_duration',T,'static')]}
  cases['geometry']={'r_end':r,'proper_duration':T,'static_equivalent_duration':float(Q),'clock_rate_initial':float(geometry(0)[1]),'clock_rate_final':float(QP),'adaptive_kappa_final':float(KAPPA*QP),'adaptive_omega_final':float(OMEGA*QP)}
  pp=np.array(cases['infall_adaptive']['p']);ps=np.array(cases['static_same_retarded_support']['p'])
  cases['control_errors']={'adaptive_vs_static_p':float(np.max(abs(pp-ps))),'input_norm_equal_support':float(np.max(abs(np.array(cases['infall_fixed']['G'])-np.array(cases['static_same_retarded_support']['G']))))}
  fixed=cases['infall_fixed']; static=cases['static_same_proper_duration']
  assert fixed['scores']['full_early']['bell_lower_min'] > static['scores']['full_early']['bell_upper_max']
  assert np.all(np.array(cases['static_same_retarded_support']['p']) > np.array(fixed['p']))
  assert cases['control_errors']['adaptive_vs_static_p'] < 1e-8
  assert cases['control_errors']['input_norm_equal_support'] < 1e-8
  assert max(cases[k]['max_balance_error'] for k in ['infall_fixed','infall_adaptive','static_same_retarded_support','static_same_proper_duration']) < 1e-8
  out['cases'][str(r)]=cases
 out['static_120']=simulate(envelope,120.,'static',step)
 return out

