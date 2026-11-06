"""Replay minimax witnesses independently, without optimization.

python code/verify_minimax_results.py reference/observer
"""
import argparse,json
from pathlib import Path
import numpy as np

TOL=1e-10


def upper_tables(source):
    """Invert binary relative entropy by vectorized bisection, independently of Brent roots."""
    source=np.asarray(source)
    if source.shape!=(32,32,32) or not np.isfinite(source).all():raise ValueError('Invalid source table.')
    if np.max(abs(source[:,0,:]))>2e-12:raise AssertionError('Empty late register is not decorrelated.')
    j=source[:,1:,:]
    if np.any((j<0)|(j>1)):raise ValueError('Invalid normalized information.')
    lo=np.full_like(j,.25);hi=np.ones_like(j)
    for _ in range(60):
        mid=(lo+hi)/2
        with np.errstate(divide='ignore',invalid='ignore'):
            value=mid*np.log2(mid/.25)+np.where(mid==1.,0.,(1-mid)*np.log2((1-mid)/.75))
        below=value<2*j
        lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
    upper=np.empty_like(source);upper[:,0,:]=.25;upper[:,1:,:]=(lo+hi)/2
    return upper.reshape(32,1024)


def verify(directory,root):
    source=np.load(root/'inputs/hp_joint_information.npy')
    p=np.load(root/'inputs/closure_reproduction_inputs.npz')['observer_carriers'][-1]
    masks=[(late,early) for late in range(32) for early in range(32)]
    a=np.array([[1]+[(late>>j)&1 for j in range(5)]+[(early>>j)&1 for j in range(5)] for late,early in masks],dtype=float).T
    b=np.r_[1.,p,[.65]*5];d=upper_tables(source)
    summary=json.loads((directory/'minimax_obstruction.json').read_text(encoding='utf-8'))
    reports=[]
    with np.load(directory/'minimax_witnesses.npz') as arrays:
        for key in ('printed','highs_ds','highs_ipm'):
            w,y,lam=arrays[key+'_law'],arrays[key+'_y'],arrays[key+'_lambda'];t=float(arrays[key+'_t'])
            if w.shape!=(1024,) or y.shape!=(11,) or lam.shape!=(32,):raise ValueError('Witness shape mismatch.')
            if not all(np.isfinite(v).all() for v in (w,y,lam,t)):raise ValueError('Nonfinite witness.')
            scores=d@w;slack=d.T@lam-a.T@y
            residuals=dict(marginal_linf=float(np.max(abs(a@w-b))),
                primal_nonnegativity=float(max(0.,-w.min())),source_inequality=float(max(0.,np.max(scores-t))),
                dual_nonnegativity=float(max(0.,-lam.min())),dual_normalization=float(abs(lam.sum()-1)),
                dual_inequality=float(max(0.,-slack.min())),primal_dual_gap=float(abs(t-b@y)),
                mask_complementarity_linf=float(np.max(abs(w*slack))),
                source_complementarity_linf=float(np.max(abs(lam*(t-scores)))))
            assert max(residuals.values())<=TOL,residuals
            assert abs(t-summary['upper_certificate'])<=TOL
            if key=='printed':
                assert np.argmax(scores)==10 and np.flatnonzero(lam).tolist()==[10]
                assert np.count_nonzero(w)==7
                np.testing.assert_allclose(scores,summary['statewise_upper'],rtol=0,atol=TOL)
                np.testing.assert_allclose(scores,arrays['printed_statewise_upper'],rtol=0,atol=TOL)
                explicit=np.zeros(1024);weight=(sum(p)-4*.35)/5
                explicit[31]=.65-weight;explicit[1023]=weight
                for j in range(5):explicit[32*sum(1<<k for k in range(5) if k!=j)]=weight+.35-p[j]
                np.testing.assert_allclose(w,explicit,rtol=0,atol=TOL)
                assert abs(max(scores)-.4469473873729414)<=TOL and max(scores)<.4470
            reports.append(dict(witness=key,upper_certificate=float(max(scores)),dual_value=float(b@y),residuals=residuals))
    return dict(passed=True,witness_pairs_checked=3,upper_certificate_tables='Independent 60-step binary-divergence bisection',
                tolerance=TOL,maximum_raw_residual=max(max(x['residuals'].values()) for x in reports),witnesses=reports)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args();print(json.dumps(verify(args.directory,Path(__file__).resolve().parents[1]),indent=2))
