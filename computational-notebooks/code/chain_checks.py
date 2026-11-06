#!/usr/bin/env python3
"""Time-independent nearest-neighbor receiver for a sech survival orbit.

The finite matrices are explicit hardware-model Hamiltonians. Their proposed
modular provenance is external to this computation. No switching is fitted.
Reusable calculations with NumPy and SciPy; no files are written on import.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import eigh_tridiagonal, expm
from scipy.special import gammaln
from scipy.integrate import quad
from scipy.stats import nbinom


def analytic_amplitudes(h, size, time):
    n = np.arange(size)
    if time == 0:
        a = np.zeros(size, dtype=complex)
        a[0] = 1
        return a
    log_magnitudes = (.5*(gammaln(2*h+n)-gammaln(2*h)-gammaln(n+1))
                      +n*np.log(np.tanh(np.pi*time))
                      -2*h*np.log(np.cosh(np.pi*time)))
    return (-1j)**n * np.exp(log_magnitudes)


def diagonalization(h, size):
    n = np.arange(1,size)
    links = np.pi*np.sqrt(n*(n+2*h-1))
    values, vectors = eigh_tridiagonal(np.zeros(size), links)
    return links, values, vectors


def evolution(values, vectors, times):
    return vectors @ (vectors[0,:,None]*np.exp(-1j*values[:,None]*times[None,:]))


def one_case(h, size):
    links, values, vectors = diagonalization(h,size)
    # Times are implementation times for H with the stated numerical links.
    # No geometric proper-time identification is inferred.
    times = np.linspace(0.,2.,801)
    states = evolution(values,vectors,times)
    survival = abs(states[0])**2
    derivative = 2*np.real(np.conj(states[0])*(-1j*links[0]*states[1]))
    exact_survival = np.cosh(np.pi*times)**(-4*h)
    exact_hazard = 4*np.pi*h*np.tanh(np.pi*times)
    trustworthy_denominator = survival > 1e-10
    hazards = np.full(len(times),np.nan)
    hazards[trustworthy_denominator] = -derivative[trustworthy_denominator]/survival[trustworthy_denominator]
    samples=[]
    for time in [.25,.5,.75,1.]:
        i=int(round(time/(times[1]-times[0])))
        finite=states[:,i]
        analytic=analytic_amplitudes(h,size,time)
        negative_binomial_tail=float(nbinom.sf(size-1,2*h,1/np.cosh(np.pi*time)**2))
        boundary_link=np.pi*np.sqrt(size*(size+2*h-1))
        # Duhamel bound for the finite state minus projection of infinite state.
        # The omitted node N supplies forcing b_N*a_N into node N-1.
        forcing_integral, forcing_quad_error=quad(
            lambda t: boundary_link*abs(analytic_amplitudes(h,size+1,t)[size]),
            0.,time,epsabs=1e-11,epsrel=1e-11)
        l2_error=float(np.linalg.norm(finite-analytic))
        assert l2_error <= forcing_integral+1e-10
        samples.append(dict(time=time,
            finite_a0_real=float(finite[0].real), finite_a0_imag=float(finite[0].imag),
            infinite_a0=float(analytic[0].real),
            amplitude_absolute_error=float(abs(finite[0]-analytic[0])),
            finite_retrieval=float(1-survival[i]),
            infinite_retrieval=float(1-exact_survival[i]),
            retrieval_absolute_error=float(abs(survival[i]-exact_survival[i])),
            finite_hazard=float(hazards[i]) if np.isfinite(hazards[i]) else None,
            infinite_hazard=float(exact_hazard[i]),
            hazard_absolute_error=float(abs(hazards[i]-exact_hazard[i])) if np.isfinite(hazards[i]) else None,
            finite_vs_truncated_infinite_l2_error=l2_error,
            duhamel_l2_upper_bound=float(forcing_integral),
            duhamel_quadrature_reported_error=float(forcing_quad_error),
            infinite_probability_beyond_finite_chain=negative_binomial_tail))
    positive_derivative=np.flatnonzero(derivative > 1e-8)
    failed_accuracy=np.flatnonzero(abs(states[0]-np.cosh(np.pi*times)**(-2*h)) > 1e-6)
    conservation=float(np.max(abs(np.sum(abs(states)**2,axis=0)-1)))
    assert conservation < 1e-11
    intervals=[]
    for stop in [.5,.75,1.]:
        mask=times<=stop
        intervals.append(dict(end=stop,
            max_amplitude_error=float(np.max(abs(states[0,mask]-np.cosh(np.pi*times[mask])**(-2*h)))),
            max_retrieval_error=float(np.max(abs(survival[mask]-exact_survival[mask]))),
            max_hazard_error=float(np.nanmax(abs(hazards[mask]-exact_hazard[mask])))))
    return dict(h=h,size=size,links_formula='b_n=pi*sqrt(n*(n+2h-1)), n=1,...,N-1',
        max_link=float(max(links)), operator_norm=float(max(abs(values))),
        state_norm_error=conservation,
        first_sampled_amplitude_error_above_1e_6=float(times[failed_accuracy[0]]) if len(failed_accuracy) else None,
        first_sampled_survival_increase=float(times[positive_derivative[0]]) if len(positive_derivative) else None,
        diagnostic_time_interval=[0,2], diagnostic_time_step=float(times[1]-times[0]),
        sampled_interval_errors=intervals,samples=samples)


def independent_checks():
    # Compare spectral evolution to a separate dense matrix exponential.
    h,size,time=1.,16,.31
    links,values,vectors=diagonalization(h,size)
    hamiltonian=np.diag(links,1)+np.diag(links,-1)
    initial=np.eye(size)[:,0]
    spectral=evolution(values,vectors,np.array([time]))[:,0]
    direct=expm(-1j*time*hamiltonian)@initial
    direct_error=float(np.linalg.norm(spectral-direct))
    # The candidate infinite amplitudes obey i da_n/dt=b_n*a_(n-1)+b_(n+1)*a_(n+1).
    residuals=[]
    step=1e-6
    for h in [.5,1.,2.]:
        a=analytic_amplitudes(h,21,.2)
        da=(analytic_amplitudes(h,21,.2+step)-analytic_amplitudes(h,21,.2-step))/(2*step)
        for n in range(20):
            rhs=np.pi*np.sqrt((n+1)*(n+2*h))*a[n+1]
            if n:
                rhs+=np.pi*np.sqrt(n*(n+2*h-1))*a[n-1]
            residuals.append(abs(1j*da[n]-rhs))
    residual=float(max(residuals))
    assert direct_error < 1e-12
    assert residual < 1e-8
    return dict(eigh_vs_dense_expm_l2_error=direct_error,
                infinite_schrodinger_equation_finite_difference_residual=residual)


def certified_design_rows():
    # Here N is the largest node index: dimension=N+1, unlike size elsewhere.
    # h=1, physical Omega=1, so b_n=sqrt(n*(n+1)), x=Omega*t.
    rows=[]
    for end,largest_node in [(1.,62),(2.,549)]:
        n=np.arange(1,largest_node+1)
        links=np.sqrt(n*(n+1))
        values,vectors=eigh_tridiagonal(np.zeros(largest_node+1),links)
        times=np.linspace(0.,end,801)
        state=evolution(values,vectors,times)
        survival=abs(state[0])**2
        derivative=2*np.real(np.conj(state[0])*(-1j*links[0]*state[1]))
        normalized_activation=-derivative/(4*survival)
        exact_survival=np.cosh(times)**(-4)
        exact_activation=np.tanh(times)
        # Independent quadrature of omitted-boundary forcing verifies closed E.
        error_bound=np.sqrt(largest_node+1)*np.tanh(end)**(largest_node+2)
        integral,quadrature_error=quad(lambda x:
            (largest_node+2)*np.sqrt(largest_node+1)
            *np.tanh(x)**(largest_node+1)/np.cosh(x)**2,
            0.,end,epsabs=1e-15,epsrel=1e-11)
        amplitude_floor=1/np.cosh(end)**2
        activation_bound=(np.sqrt(2)+2)*error_bound/(2*(amplitude_floor-error_bound))
        observed_r_error=float(np.max(abs(survival-exact_survival)))
        observed_a_error=float(np.max(abs(normalized_activation-exact_activation)))
        assert error_bound < amplitude_floor
        assert abs(integral-error_bound) < 1e-13
        assert activation_bound < 1e-6
        assert observed_r_error <= 2*error_bound+1e-12
        assert observed_a_error <= activation_bound+1e-12
        rows.append(dict(h=1,Omega=1,X=end,N_largest_node=largest_node,
            chain_dimension=largest_node+1,max_coupling=float(links[-1]),
            norm_difference_analytic_bound_E=float(error_bound),
            boundary_forcing_integral=float(integral),
            quadrature_reported_error=float(quadrature_error),
            uniform_retrieval_error_bound=float(2*error_bound),
            uniform_normalized_activation_error_bound=float(activation_bound),
            sampled_max_retrieval_error=observed_r_error,
            sampled_max_normalized_activation_error=observed_a_error,
            endpoint_retrieval=float(1-survival[-1]),
            endpoint_exact_retrieval=float(1-exact_survival[-1]),
            samples=801,sampling_interval=[0,end],
            qualification='Analytic uniform bounds with floating-point numerical '
                'evaluation; observed maxima are sampled, not interval-certified. '
                'Retention beyond endpoint requires a separate storage operation.'))
    return rows


