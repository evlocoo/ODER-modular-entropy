"""Check aggregation and primary bootstrap intervals from the saved subset tables."""
import csv,json,math,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
residuals={}
def check(name,observed,expected):
    residual=abs(float(observed)-float(expected));residuals[name]=residual
    if not math.isfinite(residual) or residual>1e-10:raise AssertionError(f'{name}: {residual}')
def percentile(values,p):
    values=sorted(map(float,values));position=(len(values)-1)*p
    low=math.floor(position);high=math.ceil(position)
    return values[low]+(position-low)*(values[high]-values[low])
groups=defaultdict(list)
with (OUT/'all_subset_scores.csv').open(newline='',encoding='utf-8') as f:
    for row in csv.DictReader(f):groups[row['encoder'],int(row['seed'])].append(row)
if len(groups)!=192:raise AssertionError('Expected 192 encodings')
computed={}
for key,rows in groups.items():
    if len(rows)!=56 or len({r['subset'] for r in rows})!=56:raise AssertionError('Incorrect subset inventory')
    contiguous=[float(r['r_network']) for r in rows if r['contiguous']=='True']
    if len(contiguous)!=8:raise AssertionError('Expected eight windows')
    c=math.fsum(contiguous)/8;u=math.fsum(float(r['r_network']) for r in rows)/56
    computed[key]={'C':c,'U':u,'gap':c-u}
with (OUT/'per_encoding.csv').open(newline='',encoding='utf-8') as f:
    for r in csv.DictReader(f):
        key=r['encoder'],int(r['seed']);values=computed[key]
        for field,name in [('contiguous_mean','C'),('uniform_mean','U'),('gap','gap')]:check(str(key)+'_'+name,r[field],values[name])
mera=[computed['mera',s]['gap'] for s in range(101001,101065)]
tree=[computed['tree',s]['gap'] for s in range(101001,101065)]
gamma=[a-b for a,b in zip(mera,tree,strict=True)]
with np.load(OUT/'bootstrap_results.npz',allow_pickle=False) as z:
    rng=np.random.Generator(np.random.PCG64(303001))
    pi=rng.integers(0,64,size=(20000,64),dtype=np.int64)
    hi=rng.integers(0,64,size=(20000,64),dtype=np.int64)
    if not np.array_equal(pi,z['pair_indices']) or not np.array_equal(hi,z['haar_indices']):raise AssertionError('Resampling indices differ from frozen convention')
    if np.array_equal(pi,hi):raise AssertionError('Haar index array was reused from pairs')
    intervals={}
    for name,values in [('Delta_MERA',mera),('Gamma',gamma)]:
        means=[math.fsum(values[int(i)] for i in row)/64 for row in pi]
        check(name+'_bootstrap_max_disagreement',max(abs(a-b) for a,b in zip(means,z[name],strict=True)),0.)
        intervals[name]=(percentile(means,.025),percentile(means,.975))
with (OUT/'descriptive_statistics.csv').open(newline='',encoding='utf-8') as f:
    for row in csv.DictReader(f):
        name=row['estimand']
        if name not in intervals:continue
        values=mera if name=='Delta_MERA' else gamma
        for field,value in [('mean',statistics.mean(values)),('sample_sd',statistics.stdev(values)),
                            ('median',statistics.median(values)),('q1',percentile(values,.25)),('q3',percentile(values,.75)),
                            ('percentile95_low',intervals[name][0]),('percentile95_high',intervals[name][1])]:
            check(name+'_'+field,row[field],value)
report={'passed':True,'encoding_means_checked':192,'subset_rows_checked':10752,
        'primary_bootstrap_means_recomputed':40000,'resampling_index_arrays_match_frozen_seed':True,
        'method':'Reaggregate CSV subset scores using math.fsum; recompute primary bootstrap means with Python loops and percentile interpolation on sorted values. Check sample SD with statistics.stdev.',
        'maximum_absolute_disagreement':max(residuals.values()),'checks':residuals}
(OUT/'statistical_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
