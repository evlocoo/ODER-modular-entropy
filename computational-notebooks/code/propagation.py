"""Scalar Schwarzschild propagation with ingoing horizon data, complex transmission 1/A_in, and calibrated packets."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.special import ndtr

@dataclass(frozen=True)
class Packet:
    center_momega: float = .16
    fractional_width: float = .2
    arrival_center_coordinate_M: float = 0.
    ell: int = 0
    normalization: str = "positive_frequency"
    @property
    def sigma_x(self): return max(.004, self.center_momega*self.fractional_width)

@dataclass(frozen=True)
class Observer:
    radius_r_over_2M: float = 10.
    @property
    def redshift(self): return math.sqrt(1-1/self.radius_r_over_2M)

def packet_spectral_density(x, packet):
    density=np.exp(-.5*((np.asarray(x)-packet.center_momega)/packet.sigma_x)**2)
    if packet.normalization == 'quadrature':
        return density/np.trapezoid(density,x)
    if packet.normalization == 'positive_frequency':
        return density/(packet.sigma_x*np.sqrt(2*np.pi)*ndtr(packet.center_momega/packet.sigma_x))
    raise ValueError('normalization must be quadrature or positive_frequency')

def scalar_transmission(x, ell=0, *, eps=1e-6, rmax_factor=40., rtol=2e-9, atol=2e-11):
    r0=2*(1+eps); rmax=max(120.,rmax_factor/max(x,.01))
    tortoise=lambda r:r+2*np.log(r/2-1)
    psi=np.exp(-1j*x*tortoise(r0))
    def rhs(r,z):
        f=1-2/r; v=f*(ell*(ell+1)/r**2+2/r**3)
        return [z[1]/f,-(x*x-v)*z[0]/f]
    sol=solve_ivp(rhs,(r0,rmax),[psi,-1j*x*psi],method='DOP853',rtol=rtol,atol=atol)
    if not sol.success: raise RuntimeError(sol.message)
    psi,y=sol.y[:,-1]; rs=tortoise(rmax)
    incoming=.5*(psi-y/(1j*x))*np.exp(1j*x*rs)
    outgoing=.5*(psi+y/(1j*x))*np.exp(-1j*x*rs)
    amp=1/incoming
    return amp, abs(amp)**2+abs(outgoing/incoming)**2-1

class TransmissionGrid:
    def __init__(self,x,amp,method='pchip_complex'):
        self.x=np.asarray(x); self.amp=np.asarray(amp); self.method=method
        if method=='pchip_complex':
            self.real=PchipInterpolator(x,amp.real,extrapolate=False)
            self.imag=PchipInterpolator(x,amp.imag,extrapolate=False)
        elif method=='cubic_complex':
            self.real=CubicSpline(x,amp.real,extrapolate=False)
            self.imag=CubicSpline(x,amp.imag,extrapolate=False)
        elif method=='pchip_phase':
            self.mag=PchipInterpolator(x,abs(amp),extrapolate=False)
            self.phase=PchipInterpolator(x,np.unwrap(np.angle(amp)),extrapolate=False)
        else: raise ValueError(method)
    def amplitude(self,x,ell=0):
        if ell!=0: raise ValueError('grid computed for ell=0')
        if self.method=='pchip_phase': return self.mag(x)*np.exp(1j*self.phase(x))
        return self.real(x)+1j*self.imag(x)

def propagated_envelope(times, packet, observer, greybody, spectral_points=2401):
    xs=np.linspace(max(.004,packet.center_momega-7*packet.sigma_x),packet.center_momega+7*packet.sigma_x,spectral_points)
    spec=np.sqrt(packet_spectral_density(xs,packet))*greybody.amplitude(xs,packet.ell)
    out=[]
    for chunk in np.array_split(np.asarray(times),max(1,int(np.ceil(len(times)/256)))):
        phase=np.exp(-1j*np.outer(chunk/observer.redshift-packet.arrival_center_coordinate_M,xs))
        out.append(np.trapezoid(phase*spec[None,:],xs,axis=1)/math.sqrt(2*math.pi*observer.redshift))
    return np.concatenate(out)

def peak_location(times,values):
    i=int(np.argmax(values))
    if i==0 or i==len(values)-1:return float(times[i])
    a,b,c=values[i-1:i+2]
    return float(times[i]+.5*(a-c)/(a-2*b+c)*(times[i+1]-times[i]))

def generate_envelopes(times,greybody, spectral_points=2401, *, center_momega=.16,
                       fractional_width=.2, arrivals=(28,40,52,64,76), radius_r_over_2M=10.,
                       normalization="positive_frequency"):
    """Generate registered observer-centered pulses using a physical delay calibration."""
    observer=Observer(radius_r_over_2M); ct=np.linspace(-16,20,3601)
    zero=Packet(center_momega,fractional_width,normalization=normalization)
    calibration=propagated_envelope(ct,zero,observer,greybody,spectral_points)
    delay=peak_location(ct,abs(calibration)**2)
    env=np.column_stack([propagated_envelope(times,Packet(center_momega,fractional_width,(arrival-delay)/observer.redshift,normalization=normalization),observer,greybody,spectral_points) for arrival in arrivals])
    return env,delay

