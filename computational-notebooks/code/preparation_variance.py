"""Matched-generator-variance comparison for the translated packet family."""
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


def transfer(t, h, b, sigma=1.):
    if h <= 0 or not 0 < b <= 1 or sigma <= 0 or np.any(np.asarray(t) < 0):
        raise ValueError('Require h,sigma>0, 0<b<=1 and t>=0.')
    x = sigma*np.asarray(t)/np.sqrt(2*h)
    return -np.expm1(-2*h*np.log1p(np.sinh(np.sqrt(b)*x)**2/b))


def checks():
    rows = []
    for h in [.5, 1., 1.5]:
        for b in [1., .75, .25]:
            shift = np.sqrt(1/b-1)
            # ell=1, measure dp/p; K=-2*pi*i*p*d/dp.
            norm = 2**(2*h)/gamma(2*h)
            weight = lambda p: norm*p**(2*h-1)*np.exp(-2*p)
            mean = quad(lambda p: -2*np.pi*shift*p*weight(p), 0, np.inf,
                        epsabs=2e-12, epsrel=2e-12)[0]
            second = quad(lambda p: 4*np.pi**2*((h-p)**2+(shift*p)**2)*weight(p),
                          0, np.inf, epsabs=2e-12, epsrel=2e-12)[0]
            var = second-mean**2
            expected = 2*h*np.pi**2/b
            sigma = 1.
            clock = sigma*np.sqrt(b)/(np.pi*np.sqrt(2*h))
            assert abs(var-expected) < 1e-10
            assert abs(clock**2*var-sigma**2) < 1e-10
            for time in [0., .2, 1., 2.]:
                u = 2*np.pi*clock*time
                def integrand(p):
                    return norm*np.exp(-h*u)*p**(2*h-1)*np.exp(-(1+np.exp(-u))*p+1j*shift*(1-np.exp(-u))*p)
                overlap = quad(lambda p: integrand(p).real, 0, np.inf,
                               epsabs=2e-12, epsrel=2e-12)[0]+1j*quad(
                               lambda p: integrand(p).imag, 0, np.inf,
                               epsabs=2e-12, epsrel=2e-12)[0]
                actual = float(transfer(time, h, b, sigma))
                centered = float(transfer(time, h, 1., sigma))
                residual = abs((1-abs(overlap)**2)-actual)
                assert residual < 1e-10 and actual <= centered+1e-10
                if b < 1 and time > 0:
                    assert actual < centered
                else:
                    assert abs(actual-centered) < 1e-14
                rows.append(dict(h=h, b=b, sigma=sigma, time=time,
                                 variance_K_quadrature=var, variance_K_formula=expected,
                                 c_b=clock, physical_variance=clock**2*var,
                                 transfer=actual, centered_transfer=centered,
                                 overlap_probability_residual=residual))
    return rows
