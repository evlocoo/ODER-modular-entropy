from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


def translated_checks():
    rows = []
    for h in [.5, 1., 1.5]:
        for ell, l in [(1., 0.), (1., .5), (1., 1.), (1., 2.)]:
            for s in [.1, .5, 1.]:
                u = 2 * np.pi * s
                prefactor = (2 * ell)**(2 * h) / gamma(2 * h) * np.exp(-h * u)

                def integrand(p):
                    return (prefactor * p**(2 * h - 1)
                            * np.exp(-ell * (1 + np.exp(-u)) * p)
                            * np.exp(1j * l * (1 - np.exp(-u)) * p))

                overlap = (quad(lambda p: integrand(p).real, 0, np.inf,
                                epsabs=1e-11)[0]
                           + 1j * quad(lambda p: integrand(p).imag, 0, np.inf,
                                       epsabs=1e-11)[0])
                b = ell**2 / (ell**2 + l**2)
                expected = (b / (b + np.sinh(np.pi * s)**2))**(2 * h)
                rows.append(dict(h=h, ell=ell, l=l, s=s, b=b,
                                 quadrature_survival=abs(overlap)**2,
                                 analytic_survival=float(expected),
                                 error=float(abs(abs(overlap)**2 - expected))))
    maximum_error = max(row['error'] for row in rows)
    assert maximum_error < 1e-9, maximum_error
    result = dict(maximum_error=maximum_error, samples=rows)
    return result
