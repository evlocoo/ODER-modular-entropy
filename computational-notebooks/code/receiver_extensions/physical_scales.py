"""Stationary scalar flux, specified mode occupations and alternative clock scales."""
import hashlib
import numpy as np
from scipy.special import ndtr
from .paths import DATA

def calculate():
    Z=np.sqrt(.9);R=20.;TAU=120.;ETA=.6336
    data=np.load(DATA/'scalar_flux_grid_384.npz')
    x,w,g=data['x'],data['weights'],data['gamma']
    n=1/np.expm1(8*np.pi*x)
    
    def entropy(n):
        return (np.log1p(n)+n*np.log1p(1/n))/np.log(2)
    
    def rates(gamma,ell):
        d=(2*ell+1)/(2*np.pi)
        return dict(number=float(d*np.sum(w*gamma*n)),
                    energy=float(d*np.sum(w*x*gamma*n)),
                    entropy_bits=float(d*np.sum(w*entropy(gamma*n))))
    
    per=[rates(gg,ell) for ell,gg in enumerate(g)]
    total={k:sum(p[k] for p in per) for k in per[0]}
    rows=[]
    for name,rate in [('ell0',per[0]),('ell0to2',total)]:
        for copies in [1,10]:
            N=copies*rate['number'];S=copies*rate['entropy_bits'];L=copies*rate['energy']
            rows.append(dict(sector=name,field_copies=copies,
                             number_rate_infinity=N,entropy_bit_rate_infinity=S,power_infinity=L,
                             photon_mean_interval_coordinate=1/N,photon_mean_interval_proper_r20=Z/N,
                             entropy_bit_interval_coordinate=1/S,entropy_bit_interval_proper_r20=Z/S,
                             expected_photons_in_120_proper=N*TAU/Z,
                             stationary_entropy_budget_in_120_proper=S*TAU/Z,
                             expected_infinity_energy_in_120_proper=L*TAU/Z,
                             photon_interval_over_prepared_peak_spacing=Z/N/12,
                             entropy_bit_interval_over_prepared_peak_spacing=Z/S/12))
    
    q=np.exp(-.5*((x-.16)/.032)**2)/(.032*np.sqrt(2*np.pi)*ndtr(5))
    trans=float(np.sum(w*q*g[0])); nout=g[0]*n
    qmode=float(np.sum(w*q*nout))
    transmode=float(np.sum(w*q*g[0]*nout)/trans)
    omega=.16/Z;kappa=.515
    # This extra spectral mode is only an asymptotic stationary filtered choice,
    # not the finite-window multimode detector output or a local-field channel.
    power_filter=(kappa/2)**2/((kappa/2)**2+(x/Z-omega)**2)
    filtered_weight=q*g[0]*power_filter
    filtered_norm=float(np.sum(w*filtered_weight))
    filteredmode=float(np.sum(w*filtered_weight*nout)/filtered_norm)
    outputmatchedmode=float(np.sum(w*filtered_weight*power_filter*nout)/filtered_norm)
    TH=1/(8*np.pi);Tloc=TH/Z;a=1/(R**2*Z);TU=a/(2*np.pi)
    clock=dict(Hawking_temperature_infinity=TH,UP_Tolman_temperature_r20=Tloc,
               static_proper_acceleration_r20=a,acceleration_Unruh_temperature_r20=TU,
               acceleration_temperature_over_UP_temperature=TU/Tloc,
               thermal_clock_char_proper=1/(np.pi*Tloc),thermal_clock_char_coordinate=1/(np.pi*Tloc)/Z,
               boost_clock_char_proper=2/a,boost_clock_char_coordinate=2/a/Z,
               lab_Omega025_char_proper=1/.025,lab_Omega025_char_coordinate=1/.025/Z,
               thermal_latency_proper=np.log(2)/(np.pi*Tloc),
               thermal_latency_coordinate=8*np.log(2),
               cavity_inverse_kappa_proper=1/kappa,cavity_omega=omega,
               cavity_linewidth_over_resonance=kappa/omega)
    out=dict(units='G=c=hbar=kB=M=1; infinity fluxes per Schwarzschild time',
             source='Prepared five-qubit HP register, exactly one excitation per dual rail, ten distinguishable scalar channels',
             proper_lapse_r20=Z,proper_window=TAU,coordinate_window=TAU/Z,
             proper_peak_spacing=12.,coordinate_peak_spacing=12/Z,
             proper_peaks=[28,40,52,64,76],coordinate_peaks=(np.array([28,40,52,64,76])/Z).tolist(),
             flux_rows=rows,
             packet=dict(center=.16,sigma=.032,norm=float(np.sum(w*q)),transmission=trans,
                         emitted_mean_energy=float(np.sum(w*x*q)),
                         transmitted_mean_energy_unnormalized=float(np.sum(w*x*q*g[0])),
                         gaussian_mode_thermal_occupation=qmode,
                         normalized_transmitted_mode_thermal_occupation=transmode,
                         normalized_stationary_cavity_filtered_mode_occupation=filteredmode,
                         stationary_matched_output_thermal_occupation_before_aperture=outputmatchedmode,
                         gaussian_mode_thermal_occupation_after_aperture=ETA*qmode,
                         transmitted_mode_thermal_occupation_after_aperture=ETA*transmode,
                         filtered_mode_thermal_occupation_after_aperture=ETA*filteredmode,
                         stationary_matched_output_thermal_occupation_after_aperture=ETA*outputmatchedmode,
                         stationary_cavity_filtered_signal_norm=filtered_norm),
             clocks=clock,
             source_grid_sha256=hashlib.sha256((DATA/'scalar_flux_grid_384.npz').read_bytes()).hexdigest(),
             limitations=['ell>=3 omitted from base flux table; existing quadrature refinement checks are not a total radial error bound',
                          'Stationary entropy flux times a finite interval is a thermodynamic budget estimate, not exact finite-window entropy',
                          'No entropy-bit-to-diary-qubit or photon-to-logical-carrier identification',
                          'Thermal-clock and acceleration-clock rows are alternative stipulated realizations; KMS alone selects neither receiver nor hazard',
                          'Mode occupations are distinct calibrated asymptotic modes, not a universal local detector noise rate'])
    assert abs(clock['thermal_clock_char_coordinate']-8)<1e-12
    assert abs(clock['boost_clock_char_coordinate']-800)<1e-10
    assert abs(clock['acceleration_temperature_over_UP_temperature']-.01)<1e-12
    assert abs(trans-.7222117972236304)<1e-12
    assert abs(out['coordinate_peak_spacing']*Z-12)<1e-12
    return out
