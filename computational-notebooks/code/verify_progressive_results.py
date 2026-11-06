"""Independently replay the progressive receiver table from frozen inputs.

Uses direct order-statistic densities and 12-point Gaussian quadrature; no
progressive_receivers functions are imported. Pass --output to save its audit.
"""
import argparse,hashlib
import numpy as np,csv,json
from pathlib import Path
from itertools import combinations
from numpy.polynomial.legendre import leggauss
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--output',type=Path)
args=parser.parse_args()
root=args.root.resolve()
def finite(label, values):
 if not np.isfinite(values).all():
  raise ValueError(f'Nonfinite {label}')
 return values

j=finite('source information', np.load(root/'inputs/hp_joint_information.npy'))[:,:,31]
rows=list(csv.DictReader(open(root/'reference/progressive/summary.csv')))
for row in rows:
 for field,value in row.items():
  if field not in ('grid','receiving_mode') and value != '':
   finite(f'summary {field}',float(value))
ref={(r['grid'],r['receiving_mode'],int(r['gate_count']),float(r['loading_delay_M'])):r for r in rows}
saved=np.load(root/'reference/progressive/statewise_scores.npz')
for key in saved.files:
 finite(f'saved array {key}',saved[key])
results=[];maxdiff=0.;max_sourcewise_diff=0.
gx,gw=leggauss(12)
for grid in ['baseline','grid_refined','physical_refined']:
 a=np.load(root/f'reference/upstream/{grid}.npz'); t=finite('times',a['times']);d=finite('observer carriers',a['observer_carriers'])-abs(finite('cavities',a['cavities']))**2
 cdf=d*d/(1e-6+d); slopes=np.diff(cdf,axis=0)/np.diff(t)[:,None]
 for delay in [0.,5.,10.]:
  end=120.-delay
  breaks=np.unique(np.r_[t[t<end],end,end-57.15])
  lo=breaks[:-1];hi=breaks[1:]
  nodes=((lo+hi)[:,None]/2+(hi-lo)[:,None]*gx/2).ravel();qw=((hi-lo)[:,None]*gw/2).ravel()
  bins=np.searchsorted(t,nodes,side='right')-1
  f=cdf[bins]+(nodes-t[bins])[:,None]*slopes[bins]; fd=slopes[bins]
  x=.025*np.minimum(120-delay-nodes,57.15)
  p=np.maximum(0.,1-np.cosh(x)**(-4)-2*np.sqrt(63)*np.tanh(x)**64)
  for m in range(1,6):
   weights=np.zeros(32);law=np.zeros(32)
   for subset in combinations(range(5),m):
    mask=sum(1<<q for q in subset);outside=set(range(5))-set(subset)
    density=np.zeros(len(nodes))
    for last in subset:
     prod=fd[:,last].copy()
     for k in subset:
      if k!=last:prod*=f[:,k]
     for k in outside:prod*=1-f[:,k]
     density+=prod
    law[mask]=qw@density;weights[mask]=(qw*p)@density
   v=.25+(np.exp2(2*j-2)-.25)@weights
   finite('recomputed scores',v)
   rr=ref[(grid,'finite_chain_reset',m,delay)]
   dif=abs(float(v.min())-float(rr['worst_source_fidelity_lower']));finite('summary comparison residual',dif);maxdiff=max(maxdiff,dif)
   key=f'{grid}__gate{m}__load{delay:g}__finite_chain_reset__fidelity_lower'
   sourcewise_diff=float(np.max(finite('sourcewise comparison residuals',abs(v-saved[key]))))
   max_sourcewise_diff=max(max_sourcewise_diff,sourcewise_diff)
   if max(dif,sourcewise_diff)>1e-11: raise AssertionError('Progressive reference replay exceeds tolerance')
   results.append(dict(grid=grid,gate=m,delay=delay,fidelity_min=float(v.min()),source=int(v.argmin()),max_ref_difference=dif,max_sourcewise_difference=sourcewise_diff))
print('max reference score diff',maxdiff)
print('minimum over three grids')
for m in range(1,6):
 print(m,{delay:min(x['fidelity_min'] for x in results if x['gate']==m and x['delay']==delay) for delay in [0.,5.,10.]})
x=.025*57.15
print('stop_capture_floor',1-np.cosh(x)**(-4)-2*np.sqrt(63)*np.tanh(x)**64)
print('largest_link',.025*np.sqrt(62*63))
print('max sourcewise score difference',max_sourcewise_diff)
files=['inputs/hp_joint_information.npy','reference/progressive/summary.csv','reference/progressive/statewise_scores.npz']+[f'reference/upstream/{grid}.npz' for grid in ['baseline','grid_refined','physical_refined']]
report=dict(cases=len(results),sources=int(j.shape[0]),quadrature_order=12,maximum_summary_difference=maxdiff,maximum_sourcewise_difference=max_sourcewise_diff,source_files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},results=results)
if args.output:
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(report,indent=2)+'\n')
