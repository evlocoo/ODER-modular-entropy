"""Static asymptotic LC comparison; no finite-deadline memory equivalence."""
import numpy as np
from .paths import DATA

def diagnostic():
    z=np.sqrt(.9);Omega=.16/z;kappa=.515
    data_path=DATA/'scalar_flux_grid_384.npz'
    a=np.load(data_path);x=a['x'];quad=a['weights'];Gamma=a['gamma'][0]
    q=np.exp(-.5*((x-.16)/.032)**2);q/=np.sum(quad*q)
    w=x/z
    legacy=1/(1-2j*(w-Omega)/kappa)
    lc=kappa*w/(kappa*w-1j*(w*w-Omega*Omega))
    def compare(weight):
     W=quad*weight;W/=W.sum()
     return {'legacy_transmitted_probability':float(np.sum(W*abs(legacy)**2)),
     'lc_transmitted_probability':float(np.sum(W*abs(lc)**2)),
     'weighted_complex_amplitude_rms':float(np.sqrt(np.sum(W*abs(lc-legacy)**2))),
     'twoport_scattering_vector_rms':float(np.sqrt(2*np.sum(W*abs(lc-legacy)**2))),
     'conditional_output_mode_overlap':float(abs(np.sum(W*lc.conj()*legacy))**2/(np.sum(W*abs(lc)**2)*np.sum(W*abs(legacy)**2)))}
    r=1-lc
    balance=float(np.max(abs(abs(r)**2+abs(lc)**2-1)))
    assert balance<1e-12
    out={'scope':'Asymptotic static scattering only; no finite-deadline stored-cavity equivalence or moving-field coupling claim.',
     'Omega':Omega,'kappa':kappa,'linewidth_over_Omega':kappa/Omega,
     'source_gaussian':compare(q),'transmitted_gaussian':compare(q*Gamma),
     'unitarity_diagonal_residual':balance,
     'unitarity_offdiagonal_residual':float(np.max(abs(r.conj()*(-lc)+(-lc).conj()*r))),
     'network_parameters_normalized':{'C':1.,'L':float(1/Omega**2),'each_port_Z':float(2/kappa)},
     'pole_values':[[float(t.real),float(t.imag)] for t in np.roots([1,kappa,Omega**2])],
     'finite_time_warning':'Do not reinterpret classical LC energy as bare-photon probability; quantum memory is dressed at strong coupling.'}
    return out

def reference_certificate():
    data=np.load(DATA/'scalar_flux_grid_384.npz')
    x=data['x'];weights=data['weights'];Gamma=data['gamma'][0]
    z=np.sqrt(.9);Omega=.16/z;kappa=.515;eta=.6336;n=5
    q=np.exp(-.5*((x-.16)/.032)**2);q/=sum(weights*q)
    nu=x/z
    old=1/(1-2j*(nu-Omega)/kappa)
    new=kappa*nu/(kappa*nu-1j*(nu**2-Omega**2))
    # Standard physical node S has reflection t-1, transmission t. Apply the
    # SAME constant reflection-output phase convention to both implementations.
    # First input-port columns then are (1-h,h), each of norm one.
    assert np.max(abs(abs(1-new)**2+abs(new)**2-1))<1e-12
    assert np.max(abs(abs(1-old)**2+abs(old)**2-1))<1e-12
    G=float(sum(weights*q*Gamma));W=weights*q*Gamma/G
    c=np.sum(W*((1-new).conj()*(1-old)+new.conj()*old))
    rows=[]
    for label,overlap in [('conditional_transmitted_packet',c),('common_greybody',1-G+G*c),('common_greybody_and_eta',1-eta*G+eta*G*c)]:
     rows.append({'scope':label,'one_carrier_overlap':[float(overlap.real),float(overlap.imag)],'one_carrier_global_trace_distance':float(np.sqrt(max(0,1-abs(overlap)**2))),'five_carrier_global_trace_distance':float(np.sqrt(max(0,1-abs(overlap)**(2*n))))})
    # Explicit finite passive completion witness for nonidentifiability from
    # complete static calibration. The two models agree identically at r=20,
    # but their spatial coupling profiles differ away from the calibration line.
    g0=.02;r0=20.;rend=4.
    T=2/(3*np.sqrt(2))*(r0**1.5-rend**1.5)
    weighted_time=2/(5*np.sqrt(2)*r0)*(r0**2.5-rend**2.5)
    p_static=np.sin(g0*T)**2
    p_infall_profile2=np.sin(g0*weighted_time)**2
    assert Omega-g0>0
    out={'scope':'Static stationary scattering of the full prepared positive-frequency packet, not switch-on-at-zero or finite-deadline possession. Each carrier is one photon in two identical independent rails. All environments vacuum; greybody and eta dilations identical in both comparisons.','eta':eta,'greybody_probability':G,'carriers':n,'certificate':rows,'embedding_nonidentifiability_witness':{'model':'Two resonant positive-frequency modes with H=Omega(Na+Nb)+g(r)(a†b+b†a); profile1 g0, profile2 g0*r/r0 on [4,20]. Positive single-particle Hamiltonian, vacuum invariant. Distinct possible spatial embeddings, not same fully specified hardware.','g0':g0,'minimum_single_particle_eigenvalue':Omega-g0,'proper_duration':T,'profile2_weighted_duration':weighted_time,'static_both_capture_probability':float(p_static),'moving_profile1_capture_probability':float(p_static),'moving_profile2_capture_probability':float(p_infall_profile2),'interpretation':'All static dynamics coincide because g1(20)=g2(20). Static calibration cannot select their off-worldline coupling profiles. This is an identifiability witness, not a Schwarzschild detector prediction.'}}
    return out
