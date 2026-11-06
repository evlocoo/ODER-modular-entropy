"""Seven-outcome obstruction and minimax upper-certificate optimization."""
import numpy as np
from scipy.optimize import linprog
from recovery_certificates import branch_tables

TOL = 1e-10
PRINTED_VALUE = .4469473873729414
PRINTED_Y = np.array([-.14072569712960725, .15449592044778254,
    .15407693269134443, .14973914230954194, .14364190332411880,
    .14796987439815912, .13539516867481916, .12956281188976515,
    .12338269739576435, .0005830648658530224, .001801954303405573])


def constraints(p, alpha=.65):
    p=np.asarray(p,dtype=float)
    if p.shape!=(5,) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Expected five finite late probabilities in [0,1].')
    if not np.isfinite(alpha) or not 0<=alpha<=1:
        raise ValueError('Invalid early probability.')
    bits=((np.arange(32)[None,:] >> np.arange(5)[:,None]) & 1)
    return np.vstack([np.ones(1024),np.repeat(bits,32,axis=1),np.tile(bits,(1,32))]),np.r_[1.,p,[alpha]*5]


def seven_outcome_law(p, alpha=.65):
    constraints(p,alpha)
    p=np.asarray(p,dtype=float)
    a=(p.sum()-4*(1-alpha))/5
    w=np.zeros(1024)
    w[31]=alpha-a
    w[1023]=a
    for j in range(5):w[32*(31 ^ (1<<j))]=a+1-alpha-p[j]
    if w.min()<0:raise ValueError('These marginals do not give a nonnegative seven-outcome law.')
    return w


def check_witness(a,b,d,w,t,y,lam):
    w,y,lam=map(np.asarray,(w,y,lam))
    if w.shape!=(1024,) or y.shape!=(11,) or lam.shape!=(len(d),):
        raise ValueError('Witness dimensions do not match the problem.')
    if not all(np.isfinite(v).all() for v in (a,b,d,w,t,y,lam)):
        raise ValueError('Nonfinite witness or problem data.')
    slack=d.T@lam-a.T@y
    values=d@w
    residuals=dict(marginal_linf=float(np.max(abs(a@w-b))),
        primal_nonnegativity=float(max(0.,-w.min())),
        source_inequality=float(max(0.,np.max(values-t))),
        dual_nonnegativity=float(max(0.,-lam.min())),
        dual_normalization=float(abs(lam.sum()-1)),
        dual_inequality=float(max(0.,-slack.min())),
        primal_dual_gap=float(abs(t-b@y)),
        mask_complementarity_linf=float(np.max(abs(w*slack))),
        source_complementarity_linf=float(np.max(abs(lam*(t-values)))))
    if max(residuals.values())>TOL:raise AssertionError(residuals)
    return dict(primal_value=float(t),dual_value=float(b@y),residuals=residuals)


def evaluate(source,p,evidence_path=None):
    a,b=constraints(p)
    _,upper=branch_tables(source)
    d=upper.reshape(len(source),1024)
    if len(d)!=32:raise ValueError('The printed witness specifies the ordered 32-source ensemble.')
    w=seven_outcome_law(p);values=d@w;t=float(values.max())
    lam=np.zeros(32);lam[10]=1
    printed=check_witness(a,b,d,w,t,PRINTED_Y,lam)
    if abs(t-PRINTED_VALUE)>TOL:raise AssertionError('Printed obstruction value does not reproduce.')
    if int(np.argmax(values))!=10:raise AssertionError('The limiting source differs from the printed witness.')
    evidence=dict(printed_law=w,printed_t=np.array(t),printed_y=PRINTED_Y,
                  printed_lambda=lam,printed_statewise_upper=values)
    solutions=[]
    options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
    for method in ('highs-ds','highs-ipm'):
        primal=linprog(np.r_[np.zeros(1024),1.],
            A_eq=np.column_stack([a,np.zeros(11)]),b_eq=b,
            A_ub=np.column_stack([d,-np.ones(32)]),b_ub=np.zeros(32),
            bounds=[(0,None)]*1024+[(None,None)],method=method,options=options)
        dual=linprog(np.r_[-b,np.zeros(32)],A_ub=np.column_stack([a.T,-d.T]),
            b_ub=np.zeros(1024),A_eq=np.r_[np.zeros(11),np.ones(32)][None,:],
            b_eq=[1.],bounds=[(None,None)]*11+[(0,None)]*32,method=method,options=options)
        if not primal.success or not dual.success:raise RuntimeError((primal.message,dual.message))
        y,weights=dual.x[:11],dual.x[11:]
        check=check_witness(a,b,d,primal.x[:-1],primal.x[-1],y,weights)
        if abs(primal.fun-t)>TOL:raise AssertionError('Explicit law differs from minimax optimum.')
        solutions.append(dict(method=method,**check))
        key=method.replace('-','_')
        evidence.update({key+'_law':primal.x[:-1],key+'_t':primal.x[-1],key+'_y':y,key+'_lambda':weights})
    difference=abs(solutions[0]['primal_value']-solutions[1]['primal_value'])
    if difference>TOL:raise AssertionError('Solver objectives disagree.')
    if evidence_path is not None:np.savez_compressed(evidence_path,**evidence)
    support=np.flatnonzero(w)
    return dict(passed=True,tolerance=TOL,early_marginal=.65,late_marginals=np.asarray(p).tolist(),
        a=float((np.sum(p)-4*.35)/5),upper_certificate=t,conservative_upper=.4470,
        limiting_source_index=10,limiting_source_ordinal=11,limiting_seed=2026082110,
        law=[dict(late_mask=int(i//32),early_mask=int(i%32),probability=float(w[i])) for i in support],
        statewise_upper=values.tolist(),printed_witness=printed,solvers=solutions,
        cross_solver_objective_difference=float(difference),
        residual_norms='Maximum absolute component for constraints; absolute scalar objective gaps.',
        interpretation='Optimum of the upper-certificate problem, not equality with optimal quantum decoding fidelity.')
