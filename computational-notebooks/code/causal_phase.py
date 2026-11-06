"""Three causal propagation arms. Run from any directory with Python 3.12.

Inputs are frozen source tables and transmission grids from the accompanying
black-hole package. No source rephasing, peak alignment, or parameter search.
The all-pass is a positive-frequency quantum envelope input-output model.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT/"inputs/causal_phase"
sys.path.insert(0, str(ROOT/'code'))
from propagation import Packet, Observer, TransmissionGrid, packet_spectral_density
from causal_filter import FilterParameters, channel_for_packets
from schwarzschild_controls import source_scores

def synth(times, xs, coeff, alpha):
    weights = np.full(len(xs), (xs[-1]-xs[0])/(len(xs)-1))
    weights[[0,-1]] *= .5
    weighted = coeff * weights[:,None] / np.sqrt(2*np.pi*alpha)
    result = []
    for chunk in np.array_split(times, max(1,int(np.ceil(len(times)/128)))):
        result.append(np.exp(-1j*np.outer(chunk/alpha,xs)) @ weighted)
    return np.concatenate(result)

def run_experiment(output):
    out=ROOT/Path(output); out.mkdir(exist_ok=True,parents=True)
    plan=json.loads((INPUTS/'parameters.json').read_text())
    for name,digest in plan['inputs_sha256'].items():
        assert hashlib.sha256((INPUTS/name).read_bytes()).hexdigest()==digest
    reg=json.loads((INPUTS/'observer_registry.json').read_text(encoding='utf-8'))
    old=json.loads((INPUTS/'control_summary.json').read_text(encoding='utf-8'))
    source=np.load(INPUTS/'control_source_information.npy')
    obs=Observer(reg['observer']['radius_r_over_2M']); alpha=obs.redshift
    pkt=Packet(reg['packet']['centers_Momega'][0],reg['packet']['fractional_width'])
    emissions=np.array(old['settings']['frozen_source_coordinate_centers_M'])
    eta=old['settings']['eta']; end=reg['time']['terminal_M']
    filt=FilterParameters(reg['filter']['center_local_Momega'],reg['filter']['kappa_total_per_M'],reg['filter']['kappa_in_fraction'],reg['filter']['kappa_detector_fraction'])
    x0=plan['resonance_Momega']; gamma=plan['gamma_Momega']; g=gamma/alpha
    runs={}; summary=[]; packets=[]; states=[]; validations=[]; bookkeeping=[]
    delay=None
    for setting,dt,nfreq in [('baseline',.1,2401),('refined',.05,4801)]:
        grid=np.load(INPUTS/f'control_transmission_{setting}.npz')
        trans=TransmissionGrid(grid['frequency'],grid['transmission'])
        xs=np.linspace(.004,.384,nfreq); T=trans.amplitude(xs)
        density=packet_spectral_density(xs,pkt); weight=density*abs(T)**2
        mean_delay=float(np.trapezoid(weight*2*gamma/((xs-x0)**2+gamma**2),xs)/np.trapezoid(weight,xs))
        if delay is None:
            delay=mean_delay
            (out/'frozen_delay_before_recovery.json').write_text(json.dumps(dict(coordinate_delay_M=delay,proper_delay_M=alpha*delay,rule=plan['delay_rule']),indent=2)+'\n')
        A=(xs-x0-1j*gamma)/(xs-x0+1j*gamma)
        phases=np.exp(1j*np.outer(xs,emissions))
        coeff=np.sqrt(density)[:,None]*T[:,None]*phases
        state_coeff=coeff*(np.sqrt(2*g)/(g+1j*(x0-xs)/alpha))[:,None]
        times=np.arange(round(end/dt)+1)*dt
        for arm,factor in [('schwarzschild',np.ones_like(A)),('matched_delay',np.exp(1j*xs*delay)),('causal_allpass',A)]:
            env=synth(times,xs,coeff*factor[:,None],alpha)
            incident,collected,diags,cav=channel_for_packets(times,env,eta=eta,filter_parameters=filt)
            info,ceil,floor=source_scores(source,collected[-1])
            supply,_,_=source_scores(source,incident[-1])
            assert np.max(collected-eta*incident)<2e-9
            assert np.max(info-supply)<2e-9
            assert max(d['maximum_energy_balance_residual'] for d in diags)<2e-8
            assert collected.min()>-2e-12 and incident.max()<1+2e-8
            result=dict(times=times,envelopes=env,incident=incident,collected=collected,information=info,ceilings=ceil,floors=floor,cavities=cav)
            runs[setting,arm]=result
            np.savez_compressed(out/f'{setting}_{arm}.npz',**result)
            summary.append(dict(setting=setting,arm=arm,spectral_transmission=float(np.trapezoid(density*abs(T*factor)**2,xs)),mean_collected=float(collected[-1].mean()),mean_no_early=float(info[:,0].mean()),mean_full_early=float(info[:,-1].mean()),fidelity_ceiling=float(ceil.max()),fidelity_floor=float(floor.min()),maximum_detector_energy_residual=max(d['maximum_energy_balance_residual'] for d in diags),maximum_information_minus_incident_supply=float(np.max(info-supply))))
            for j in range(5): packets.append(dict(setting=setting,arm=arm,packet=j+1,incident=incident[-1,j],collected=collected[-1,j],detector_memory=abs(cav[-1,j])**2))
            for j in range(32): states.append(dict(setting=setting,arm=arm,seed=reg['hp']['first_seed']+j,no_early=info[j,0],full_early=info[j,-1],ceiling=ceil[j],floor=floor[j]))
            print(setting,arm,'completed',flush=True)
        # Independent causal state evolution, with explicit prehistory.
        # Full-spectrum state at -400 is the consistent incoming packet state.
        hist=np.arange(-round(400/dt),round(end/dt)+1)*dt
        incoming=synth(hist,xs,coeff,alpha)
        drive=CubicSpline(hist,incoming,axis=0)
        initial=synth(np.array([-400.]),xs,state_coeff,alpha)[0]
        def rhs(t,z):
            b=z[:5]; u=drive(t); y=u-np.sqrt(2*g)*b
            return np.concatenate((-(g+1j*x0/alpha)*b+np.sqrt(2*g)*u,abs(u)**2,abs(y)**2))
        sol=solve_ivp(rhs,(hist[0],hist[-1]),np.concatenate((initial,np.zeros(10))),t_eval=hist,method='DOP853',rtol=2e-11,atol=2e-13,max_step=.05)
        assert sol.success
        b=sol.y[:5].T; E_in=sol.y[5:10].real.T; E_out=sol.y[10:15].real.T
        balance=E_in+abs(initial)**2-E_out-abs(b)**2
        zero=round(400/dt)
        y=incoming[zero:]-np.sqrt(2*g)*b[zero:]
        spec_state=synth(times,xs,state_coeff,alpha)
        diff=float(np.max(abs(y-runs[setting,'causal_allpass']['envelopes'])))
        state_diff=float(np.max(abs(b[zero:]-spec_state)))
        far=synth(np.array([-800.,-400.]),xs,state_coeff,alpha)
        history_bounds=[float(np.max(abs(v))*np.exp(-g*duration)) for v,duration in zip(far,[800,400])]
        baseline=np.load(INPUTS/f'control_{setting}_schwarzschild.npz')
        reproduction=float(np.max(abs(runs[setting,'schwarzschild']['collected'][-1]-baseline['observer_carriers'][-1])))
        val=dict(setting=setting,maximum_unit_modulus_error=float(np.max(abs(abs(A)**2-1))),maximum_pointwise_transmission_probability_difference=float(np.max(abs(abs(T*A)**2-abs(T)**2))),recomputed_mean_delay_M=mean_delay,frozen_delay_M=delay,maximum_upstream_energy_residual=float(np.max(abs(balance))),maximum_spectral_vs_causal_waveform_error=diff,maximum_spectral_vs_causal_state_error=state_diff,zero_start_minus800_state_error_bound_at0=history_bounds[0],zero_start_minus400_state_error_bound_at0=history_bounds[1],baseline_reproduction_error=reproduction)
        validations.append(val)
        assert diff<2e-8 and state_diff<2e-8 and np.max(abs(balance))<2e-8 and reproduction<2e-8
        for j in range(5):
            bookkeeping.append(dict(setting=setting,packet=j+1,incoming_energy_minus400_to0=E_in[zero,j],outgoing_energy_minus400_to0=E_out[zero,j],upstream_storage_at_minus400=abs(initial[j])**2,upstream_storage_at0=abs(b[zero,j])**2,upstream_storage_at_deadline=abs(b[-1,j])**2,incoming_energy0_to_deadline=E_in[-1,j]-E_in[zero,j],outgoing_energy0_to_deadline=E_out[-1,j]-E_out[zero,j]))
        np.savez_compressed(out/f'{setting}_causal_state_check.npz',times=hist,upstream_state=b,input_energy=E_in,output_energy=E_out,energy_balance=balance)
        print(setting,'independent causal check:',json.dumps(val),flush=True)
    refinement=[]
    for arm in ['schwarzschild','matched_delay','causal_allpass']:
        a=runs['baseline',arm]; b=runs['refined',arm]
        refinement.append(dict(arm=arm,maximum_terminal_collection_change=float(np.max(abs(a['collected'][-1]-b['collected'][-1]))),maximum_information_change=float(np.max(abs(a['information']-b['information']))),maximum_bound_change=float(max(np.max(abs(a['ceilings']-b['ceilings'])),np.max(abs(a['floors']-b['floors']))))))
    for name,rows in [('summary',summary),('packets',packets),('states',states),('validation',validations),('upstream_bookkeeping',bookkeeping),('refinement',refinement)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    (out/'results.json').write_text(json.dumps(dict(summary=summary,validation=validations,refinement=refinement,upstream_bookkeeping=bookkeeping),indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


