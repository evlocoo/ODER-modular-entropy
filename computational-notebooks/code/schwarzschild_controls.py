"""Fixed-source transmission controls for the Schwarzschild diary benchmark.

Every condition uses the same source spectral normalization, emission phases,
observer clock, collection interval and initially empty passive receiver.
The magnitude-only condition is a mathematical diagnostic, not a causal model.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
from propagation import (Observer, Packet, TransmissionGrid, scalar_transmission,
                         propagated_envelope, peak_location, packet_spectral_density)
from causal_filter import channel_for_packets, FilterParameters
from cavity import mask_weights
from source_tables import generate_source_tables

CONDITIONS = ('schwarzschild', 'unit_transmission', 'magnitude_only')

class ControlTransmission:
    def __init__(self, transmission, condition):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.transmission, self.condition = transmission, condition
    def amplitude(self, x, ell=0):
        if self.condition == 'unit_transmission':
            return np.ones_like(x, dtype=complex)
        values = self.transmission.amplitude(x, ell)
        return abs(values) if self.condition == 'magnitude_only' else values

def source_scores(source, probabilities):
    """Evaluate each fixed HP source; bounds are uniform over the 32 seeds."""
    information = np.einsum('d,sde->se', mask_weights(probabilities), source)
    def binary_divergence(f):
        if f == 1.:
            return 2.
        return f*np.log2(f/.25)+(1-f)*np.log2((1-f)/.75)
    ceiling = np.array([brentq(lambda f:binary_divergence(f)-2*e,.25,1.)
                        for e in information[:,0]])
    floor = 2**(2*information[:,31]-2)
    return information, ceiling, floor

def run_controls(root='.', output='outputs/controls'):
    root = Path(root)
    out = root/output
    out.mkdir(parents=True, exist_ok=True)
    registry = json.loads((root/'inputs/observer_registry.json').read_text())
    assert len(set(registry['packet']['centers_Momega'])) == 1
    with threadpool_limits(limits=1):
        source, _, _, diagnostics = generate_source_tables(
            registry['hp']['first_seed'],registry['hp']['ensemble_size'])
    assert source.shape == (32,32,32)
    assert max(d['normalization_error'] for d in diagnostics)<2e-12
    assert max(d['unitary_residual'] for d in diagnostics)<2e-12
    np.save(out/'control_source_information.npy', source)
    observer = Observer(registry['observer']['radius_r_over_2M'])
    center = registry['packet']['centers_Momega'][0]
    width = registry['packet']['fractional_width']
    eta = registry['observer']['aperture_fraction']*registry['observer']['efficiency']
    terminal = registry['time']['terminal_M']
    filt = FilterParameters(registry['filter']['center_local_Momega'],
                           registry['filter']['kappa_total_per_M'],
                           registry['filter']['kappa_in_fraction'],
                           registry['filter']['kappa_detector_fraction'])
    zero = Packet(center,width)
    settings = [('baseline',321,.1,2401,{}),
                ('refined',641,.05,4801,
                 dict(eps=3e-7,rmax_factor=65.,rtol=3e-11,atol=3e-13))]
    runs, summaries, packets, source_rows, solver_rows = {}, [], [], [], []
    emission_coordinates = None
    for setting,nodes,dt,nfreq,options in settings:
        started = time.perf_counter()
        frequencies = np.linspace(.004,.384,nodes)
        solved = np.array([scalar_transmission(float(w),**options) for w in frequencies])
        flux = float(np.max(abs(solved[:,1])))
        assert flux<2e-8
        transmission = TransmissionGrid(frequencies,solved[:,0])
        np.savez_compressed(out/('control_transmission_'+setting+'.npz'),
                            frequency=frequencies,transmission=solved[:,0])
        # Calibrate once. The resulting source phases remain fixed for every
        # control and for numerical refinement; no output peaks are realigned.
        if emission_coordinates is None:
            ct=np.linspace(-16,20,3601)
            calibration=propagated_envelope(ct,zero,observer,transmission,2401)
            delay=peak_location(ct,abs(calibration)**2)
            emission_coordinates=(np.array(registry['packet']['arrival_centers_observer_input_M'])-delay)/observer.redshift
        times=np.arange(0,terminal+dt/2,dt)
        xs=np.linspace(.004,.384,nfreq)
        density=packet_spectral_density(xs,zero)
        input_band_mass=float(np.trapezoid(density,xs))
        for condition in CONDITIONS:
            control=ControlTransmission(transmission,condition)
            wavepackets=[Packet(center,width,float(t)) for t in emission_coordinates]
            envelopes=np.column_stack([propagated_envelope(times,p,observer,control,nfreq)
                                        for p in wavepackets])
            global_curves,observer_curves,diags,cavities=channel_for_packets(
                times,envelopes,eta=eta,filter_parameters=filt)
            # No clipping or renormalization of carrier probabilities.
            assert np.min(observer_curves)>-2e-12
            assert np.max(observer_curves-eta*global_curves)<2e-9
            assert np.max(global_curves)<1.+2e-8
            assert max(d['maximum_energy_balance_residual'] for d in diags)<2e-8
            info,ceiling,floor=source_scores(source,observer_curves[-1])
            spectral_transmission=float(np.trapezoid(density*abs(control.amplitude(xs))**2,xs))
            run=dict(times=times,envelopes=envelopes,global_carriers=global_curves,
                     observer_carriers=observer_curves,cavities=cavities,information=info,
                     ceilings=ceiling,floors=floor)
            runs[setting,condition]=run
            np.savez_compressed(out/('control_'+setting+'_'+condition+'.npz'),**run)
            summaries.append(dict(setting=setting,condition=condition,
                spectral_transmission=spectral_transmission,input_band_mass=input_band_mass,
                mean_terminal_incident=float(global_curves[-1].mean()),
                mean_terminal_collected=float(observer_curves[-1].mean()),
                mean_no_early=float(info[:,0].mean()),mean_full_early=float(info[:,31].mean()),
                fidelity_ceiling=float(ceiling.max()),fidelity_floor=float(floor.min()),
                maximum_energy_balance_residual=max(d['maximum_energy_balance_residual'] for d in diags)))
            for j in range(5):
                packets.append(dict(setting=setting,condition=condition,packet=j+1,
                    source_coordinate_center_M=float(emission_coordinates[j]),
                    input_peak_M=peak_location(times,abs(envelopes[:,j])**2),
                    terminal_incident=float(global_curves[-1,j]),
                    terminal_collected=float(observer_curves[-1,j]),
                    terminal_cavity=float(abs(cavities[-1,j])**2)))
            for s in range(32):
                source_rows.append(dict(setting=setting,condition=condition,
                    seed=registry['hp']['first_seed']+s,no_early=float(info[s,0]),
                    full_early=float(info[s,31]),ceiling=float(ceiling[s]),floor=float(floor[s])))
        solver_rows.append(dict(setting=setting,frequency_nodes=nodes,time_step_M=dt,
            spectral_points=nfreq,maximum_flux_residual=flux,seconds=time.perf_counter()-started))
        print(setting,'controls complete',flush=True)
    refinement=[]
    for condition in CONDITIONS:
        a,b=runs['baseline',condition],runs['refined',condition]
        refinement.append(dict(condition=condition,
            maximum_terminal_probability_change=float(np.max(abs(a['observer_carriers'][-1]-b['observer_carriers'][-1]))),
            maximum_information_change=float(np.max(abs(a['information']-b['information']))),
            maximum_fidelity_bound_change=float(max(np.max(abs(a['ceilings']-b['ceilings'])),np.max(abs(a['floors']-b['floors']))))))
    # Positive-frequency Gaussian packets have nonzero pre-window tails.
    # These diagnostics do not feed or prepopulate the receiver at time zero.
    # The finite intervals are reported explicitly, not as exact (-inf,0) integrals.
    prewindow=[]
    for condition in CONDITIONS:
        control=ControlTransmission(transmission,condition)
        p=Packet(center,width,float(emission_coordinates[0]))
        past=np.arange(-800.,.1,.1)
        envelope=propagated_envelope(past,p,observer,control,4801)
        before400=past<=-400.
        prewindow.append(dict(condition=condition,packet=1,
            incident_energy_minus800_to0=float(np.trapezoid(abs(envelope)**2,past)),
            incident_energy_minus800_to_minus400=float(np.trapezoid(abs(envelope[before400])**2,past[before400]))))
    tables=dict(summary=pd.DataFrame(summaries),packets=pd.DataFrame(packets),
                states=pd.DataFrame(source_rows),refinement=pd.DataFrame(refinement),
                solver=pd.DataFrame(solver_rows),prewindow=pd.DataFrame(prewindow))
    for name,frame in tables.items():frame.to_csv(out/('control_'+name+'.csv'),index=False)
    meta=dict(source_calibration_delay_M=float(delay),
        frozen_source_coordinate_centers_M=emission_coordinates.tolist(),
        observer_lapse=observer.redshift,eta=eta,collection_interval_M=[0.,terminal],
        source_generated=True,conditions=list(CONDITIONS),
        refinement='Joint refinement of time, spectral quadrature, transmission grid, ODE tolerances and radial boundaries; source phases fixed.',
        interpretation='Unit transmission is a declared potential-free transfer reference at the same lapse and source phase convention. Magnitude-only is a phase-removal diagnostic and need not be causal. The receiver is empty at time zero in every condition; positive-frequency packet prehistory is not collected.')
    (out/'control_summary.json').write_text(json.dumps(dict(settings=meta,
        results=tables['summary'].to_dict('records'),refinement=tables['refinement'].to_dict('records'),
        prewindow=tables['prewindow'].to_dict('records')),indent=2)+'\n')
    return tables,meta
