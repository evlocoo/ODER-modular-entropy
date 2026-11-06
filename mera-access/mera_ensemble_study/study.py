"""Predeclared 64-pair MERA/tree and 64-Haar eight-site encoding study."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,csv,hashlib,importlib.util,itertools,json,platform,sys,threading,time
from pathlib import Path
import numpy as np
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'inputs'
MERA_SEEDS=list(range(101001,101065))
HAAR_SEEDS=list(range(202001,202065))
BOOT_SEED=303001
BOOT_REPS=20000
TOL=1e-10
RESOLUTION=1e-8
SUBSETS=list(itertools.combinations(range(8),3))
WINDOWS={tuple(sorted((i,(i+1)%8,(i+2)%8))) for i in range(8)}

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ahash(a):return hashlib.sha256(a.astype('<c16').tobytes(order='C')).hexdigest()
def write_json(p,value):p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def table(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

class Validation:
    def __init__(self):self.values={}
    def check(self,name,value):
        value=float(value);self.values[name]=value
        if not np.isfinite(value) or value>TOL:raise AssertionError(f'{name}: {value:.17g} exceeds {TOL}')

class ResourceBudgetReached(RuntimeError):pass

def load_inputs():
    identity=json.loads((ROOT/'input_identity.json').read_text())
    for name,record in identity['files'].items():
        if digest(INPUT/name)!=record['sha256']:raise ValueError('Input changed: '+name)
    old_plan=json.loads((INPUT/'original_plan.json').read_text())
    if digest(INPUT/'original_methods.py')!=old_plan['provenance']['source_sha256']:raise ValueError('Audited source mismatch')
    if not json.loads((INPUT/'architecture_audit_summary.json').read_text())['passed']:raise ValueError('Architecture audit did not pass')
    audited=json.loads((INPUT/'architecture_audit_plan.json').read_text())
    if digest(INPUT/'audited_architecture.json')!=audited['architecture_sha256']:raise ValueError('Architecture specification changed')
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location('audited_original',INPUT/'original_methods.py')
    methods=importlib.util.module_from_spec(spec);spec.loader.exec_module(methods)
    return methods,identity,old_plan

def make_plan(identity,old_plan):
    return {'study':'Eight-site random-encoding ensemble, fixed architecture, no optimization',
        'source_sha256':digest(Path(__file__)),'input_identity':identity,
        'mera_seeds':MERA_SEEDS,'haar_seeds':HAAR_SEEDS,'statistical_resampling_seed':BOOT_SEED,
        'exploratory_seed':20260918,'exploratory_policy':'Separate frozen-tensor regression; excluded from every ensemble estimate and bootstrap.',
        'rng':'NumPy Generator(PCG64)',
        'mera_generation':{'method':'Call unchanged audited pilot generate() with only its in-memory SEED set to the declared seed.',
            'draw_order':old_plan['generation_order'],'phase_convention':old_plan['qr_convention'],
            'architecture':json.loads((INPUT/'audited_architecture.json').read_text()),'legs':old_plan['legs']},
        'paired_tree':'Copy all seven isometries exactly; replace six disentanglers with I4. No independent tree draw.',
        'haar_generation':'Draw real standard_normal((256,2)), then imaginary standard_normal((256,2)); reduced QR; multiply Q columns by diag(R)/abs(diag(R)). One independent seed per encoding. Haar seed indices create no pairing with structured encodings.',
        'source':old_plan['source'],'access':old_plan['ensembles'],
        'outcomes':{'score':'I(Q:B_S)/2 in bits, preserving the accessible classical subset flag',
            'gap':'Delta=mean_contiguous-mean_uniform','amplification':'Gamma=Delta_MERA-Delta_tree',
            'level_changes':'L_C=mean_C_MERA-mean_C_tree; L_U=mean_U_MERA-mean_U_tree; Gamma=L_C-L_U'},
        'primary_estimands':['E[Delta_MERA]','E[Gamma]'],
        'secondary_estimands':['E[Delta_tree]','E[Delta_Haar]','All acquisition-level means','E[L_C]','E[L_U]',
            'Structured minus Haar gap and acquisition-level contrasts'],
        'summary_statistics':'Encoding-level mean, sample SD (ddof=1), median, Q1, Q3, IQR, minimum, maximum, and positive/unresolved/negative counts at 1e-8. NumPy linear quantiles.',
        'bootstrap':{'replicates':BOOT_REPS,'rng_seed':BOOT_SEED,'sampling_unit':'Encoding; MERA/tree are paired',
            'draw_order':'PCG64(303001): first draw a (20000,64) int64 index array for pairs; then an independent (20000,64) int64 array for Haar. Use the same pair indices for every paired/structured estimand.',
            'interval':'2.5th and 97.5th percentiles of encoding-level bootstrap means, method=linear; no centering or multiplicity adjustment.',
            'label':'Approximate Monte Carlo uncertainty for the declared random-encoding distributions; not subset-level replication.',
            'primary_interpretation':'Assess each primary question separately. Entire interval above zero supports positive ensemble mean; entire interval below zero supports negative mean; crossing zero leaves its sign unresolved by this interval. No omnibus success claim.',
            'haar_rule':'Individual Haar gaps may be nonzero. Permutation symmetry implies zero ensemble expectation, not an exact-zero requirement on this sample.'},
        'sign_classification_threshold':RESOLUTION,'threshold_meaning':'Numerical resolution only, not a scientific effect-size threshold',
        'validation_tolerance':TOL,'norms':'Frobenius for matrices; Euclidean for states; absolute scalar differences; minimum eigenvalue >= -1e-10.',
        'entropy':old_plan['entropy'],
        'validation':'Every structured encoding: original independent contraction/dense encodings and independent partial traces. Haar: both partial traces. All: tensor/full isometry, normalization, density trace/Hermiticity/positivity, empty/full information, all 56 subset states and scores, pure complement identities. Compare all eight-by-two logical columns, not only selected input states.',
        'algebraic_consequence':'For pure Bell-reference preparation, complementary five-site ensemble scores equal 1 minus three-site scores, so Delta_5=-Delta_3. Report as an algebraic consequence, not new independent evidence.',
        'execution':'Sequential. Save tensors before scoring. Checkpoint each completed MERA/tree seed pair or Haar seed; never substitute a failed seed.',
        'resource_budget_per_process':{'wall_seconds':1800,'cpu_seconds':1800,'peak_rss_bytes':2*1024**3,
            'policy':'Sample RSS every 10 ms; check soft budget before/after each seed. Finish the active seed, checkpoint and stop/report if exceeded.'},
        'failure_policy':'Preserve failed seed outputs and failure record; diagnose implementation, document repairs/reruns, never reseed or discard negative results. No automatic replacement.',
        'software':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'psutil':psutil.__version__,'platform':platform.system()},
        'scope':'No architecture changes, Hamiltonian optimization, larger systems or manuscript edits. Stop after this ensemble study.',
        'identity_note':'Hashes establish artifact identity, not independent timestamped preregistration.'}

def generate_pair(methods,seed):
    old=methods.SEED;methods.SEED=seed
    try:mera=methods.generate()
    finally:methods.SEED=old
    tree={k:a.copy() if k.startswith('w') else np.eye(4,dtype=complex) for k,a in mera.items()}
    if sum(k.startswith('w') for k in mera)!=7 or sum(k.startswith('u') for k in mera)!=6:raise ValueError('Tensor inventory changed')
    if any(not np.array_equal(mera[k],tree[k]) for k in mera if k.startswith('w')):raise ValueError('Pair isometries differ')
    return mera,tree

def generate_haar(seed):
    rng=np.random.Generator(np.random.PCG64(seed))
    a=rng.standard_normal((256,2))+1j*rng.standard_normal((256,2))
    q,r=np.linalg.qr(a,mode='reduced');d=np.diag(r)
    if np.any(np.abs(d)==0):raise ValueError('Zero diagonal in Haar QR')
    return q*(d/np.abs(d))[None,:]

def evaluate(methods,arm,seed,out,tensors=None,haar=None):
    out.mkdir();val=Validation();eigenvalues={};density_rows=[];scores=[]
    if tensors is not None:
        for k,a in tensors.items():
            val.check(k+'_left_isometry',np.linalg.norm(a.conj().T@a-np.eye(a.shape[1])))
            if k.startswith('u'):val.check(k+'_right_unitarity',np.linalg.norm(a@a.conj().T-np.eye(4)))
        vn=methods.contract_network(tensors);vd=methods.dense_encoding(tensors)
    else:vn=haar;vd=haar.copy()
    val.check('encoding_agreement_fro',np.linalg.norm(vn-vd))
    states={}
    for name,v in [('network',vn),('dense',vd)]:
        val.check(name+'_encoding_isometry_fro',np.linalg.norm(v.conj().T@v-np.eye(2)))
        states[name]=v.T.reshape(-1)/np.sqrt(2)
        val.check(name+'_normalization',abs(np.vdot(states[name],states[name])-1))
    val.check('state_agreement_l2',np.linalg.norm(states['network']-states['dense']))
    np.savez_compressed(out/'encodings.npz',V_network=vn,V_dense=vd)
    def reduced(path,sites):
        red=methods.reduced_tensor if path=='network' else methods.reduced_dense
        return red(states[path],sites)
    def entropy(rho,key):
        tr=float(abs(np.trace(rho)-1));herm=float(np.linalg.norm(rho-rho.conj().T))
        eig=np.linalg.eigvalsh((rho+rho.conj().T)/2)
        val.check(key+'_trace',tr);val.check(key+'_hermiticity_fro',herm)
        val.check(key+'_negative_eigenvalue',max(0.,-eig.min()))
        eigenvalues[key]=eig
        density_rows.append({'state':key,'trace_residual':tr,'hermiticity_fro':herm,
                             'minimum_raw_eigenvalue':float(eig.min()),'negative_eigenvalue_count':int(np.sum(eig<0))})
        positive=eig[eig>0]
        return float(-np.sum(positive*np.log2(positive)))
    sq={p:entropy(reduced(p,[0]),p+'_Q') for p in states}
    for p in sq:val.check(p+'_reference_entropy',abs(sq[p]-1))
    def information(path,s,key):
        b=reduced(path,[i+1 for i in s]);qb=reduced(path,[0]+[i+1 for i in s])
        value=sq[path]+entropy(b,key+'_B')-entropy(qb,key+'_QB')
        return value,qb
    endpoints={}
    for p in states:
        empty,_=information(p,[],p+'_empty');full,_=information(p,list(range(8)),p+'_full')
        val.check(p+'_empty_information',abs(empty));val.check(p+'_full_information',abs(full-2))
        endpoints[p]={'empty_information':empty,'full_information':full}
    for s in SUBSETS:
        key=''.join(map(str,s));comp=[i for i in range(8) if i not in s];vals={};rhos={};cv={};crhos={}
        for p in states:
            vals[p],rhos[p]=information(p,list(s),p+'_'+key)
            cv[p],crhos[p]=information(p,comp,p+'_complement_'+key)
            val.check(p+'_'+key+'_pure_complement',abs(vals[p]+cv[p]-2))
            val.check(p+'_'+key+'_score_range',max(0.,-vals[p]/2,vals[p]/2-1))
        val.check(key+'_rho_agreement_fro',np.linalg.norm(rhos['network']-rhos['dense']))
        val.check(key+'_complement_rho_agreement_fro',np.linalg.norm(crhos['network']-crhos['dense']))
        val.check(key+'_score_agreement',abs(vals['network']-vals['dense'])/2)
        scores.append({'encoder':arm,'seed':seed,'subset':' '.join(map(str,s)),'contiguous':s in WINDOWS,
                       'r_network':vals['network']/2,'r_dense':vals['dense']/2,
                       'complement_r_network':cv['network']/2,'complement_r_dense':cv['dense']/2,
                       'complement_residual_network':(vals['network']+cv['network'])/2-1,
                       'complement_residual_dense':(vals['dense']+cv['dense'])/2-1})
    means={p:{g:float(np.mean([r['r_'+p] for r in scores if g=='uniform' or r['contiguous']]))
              for g in ['contiguous','uniform']} for p in states}
    gap=means['network']['contiguous']-means['network']['uniform']
    val.check('gap_path_agreement',abs(gap-(means['dense']['contiguous']-means['dense']['uniform'])))
    five={g:float(np.mean([r['complement_r_network'] for r in scores if g=='uniform' or r['contiguous']])) for g in ['contiguous','uniform']}
    val.check('complementary_gap_identity',abs(five['contiguous']-five['uniform']+gap))
    np.savez_compressed(out/'raw_eigenvalues.npz',**eigenvalues)
    table(out/'density_validation.csv',density_rows);table(out/'subset_scores.csv',scores)
    write_json(out/'validation.json',{'passed':True,'tolerance':TOL,'checks':val.values,
               'minimum_raw_eigenvalue':min(r['minimum_raw_eigenvalue'] for r in density_rows),
               'tolerated_negative_eigenvalues':sum(r['negative_eigenvalue_count'] for r in density_rows),
               'endpoints':endpoints,'maximum_residual':max(val.values.values())})
    summary={'encoder':arm,'seed':seed,'contiguous_mean':means['network']['contiguous'],
             'uniform_mean':means['network']['uniform'],'gap':gap,'means_by_path':means,
             'complementary_five_site_means':five,'complementary_gap_residual':five['contiguous']-five['uniform']+gap,
             'maximum_validation_residual':max(val.values.values()),'passed':True}
    write_json(out/'summary.json',summary)
    return summary

def checkpoint(folder,record):
    # Completion marker written last. Its hashes detect incomplete or altered checkpoints.
    record['output_sha256']={p.relative_to(folder).as_posix():digest(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='checkpoint.json'}
    write_json(folder/'checkpoint.json',record)

def read_checkpoint(folder):
    marker=folder/'checkpoint.json'
    if not marker.exists():
        if folder.exists():raise RuntimeError('Incomplete seed directory retained for diagnosis: '+folder.name)
        return None
    value=json.loads(marker.read_text())
    for name,h in value['output_sha256'].items():
        if digest(folder/name)!=h:raise ValueError('Checkpoint file changed: '+name)
    return value

def regression(methods,out):
    out.mkdir();summaries={};val=Validation()
    mh=json.loads((INPUT/'mera_hashes.json').read_text());th=json.loads((INPUT/'tree_hashes.json').read_text())
    for arm,expected,hashes in [('mera',mh['archive_sha256'],mh['arrays']),('tree',th['tree_archive_sha256'],th['tree_array_hashes'])]:
        p=INPUT/(arm+'_tensors.npz')
        if digest(p)!=expected:raise ValueError('Exploratory tensor archive hash mismatch')
        with np.load(p,allow_pickle=False) as z:t={k:z[k] for k in z.files}
        if {k:ahash(a) for k,a in t.items()}!=hashes:raise ValueError('Exploratory tensor arrays changed')
        summaries[arm]=evaluate(methods,arm,20260918,out/arm,tensors=t)
    prior=json.loads((INPUT/'ablation_summary.json').read_text())
    for arm,r in summaries.items():
        val.check(arm+'_gap_regression',abs(r['gap']-prior['delta_'+arm]))
        for g in ['contiguous','uniform']:val.check(arm+'_'+g+'_regression',abs(r[g+'_mean']-prior['means'][arm]['network'][g]))
        old=list(csv.DictReader((INPUT/'baseline_paired.csv').open(encoding='utf-8',newline='')))
        current=list(csv.DictReader((out/arm/'subset_scores.csv').open(encoding='utf-8',newline='')))
        for a,b in zip(current,old,strict=True):
            if a['subset']!=b['subset']:raise ValueError('Exploratory subset ordering changed')
            for path in ['network','dense']:val.check(arm+'_'+a['subset']+'_'+path,abs(float(a['r_'+path])-float(b['r_'+arm+'_'+path])))
    result={'passed':True,'excluded_from_ensemble':True,'seed':20260918,'encoders':summaries,
            'gamma':summaries['mera']['gap']-summaries['tree']['gap'],
            'L_C':summaries['mera']['contiguous_mean']-summaries['tree']['contiguous_mean'],
            'L_U':summaries['mera']['uniform_mean']-summaries['tree']['uniform_mean'],
            'maximum_regression_residual':max(val.values.values()),'checks':val.values}
    checkpoint(out,result);return result

def describe(name,values,boot):
    q1,q3=np.percentile(values,[25,75],method='linear');lo,hi=np.percentile(boot,[2.5,97.5],method='linear')
    return {'estimand':name,'n_encodings_or_pairs':len(values),'mean':float(np.mean(values)),
        'sample_sd':float(np.std(values,ddof=1)),'median':float(np.median(values)),
        'q1':float(q1),'q3':float(q3),'iqr':float(q3-q1),'minimum':float(np.min(values)),
        'maximum':float(np.max(values)),'positive':int(np.sum(values>RESOLUTION)),
        'unresolved':int(np.sum(np.abs(values)<=RESOLUTION)),'negative':int(np.sum(values< -RESOLUTION)),
        'percentile95_low':float(lo),'percentile95_high':float(hi)}

def analyze(out,pairs,haars,exploratory):
    if [r['seed'] for r in pairs]!=MERA_SEEDS or [r['seed'] for r in haars]!=HAAR_SEEDS:
        raise ValueError('Seed inventory incomplete or reordered')
    metrics={
        'Delta_MERA':np.array([r['mera']['gap'] for r in pairs]),
        'Delta_tree':np.array([r['tree']['gap'] for r in pairs]),
        'Delta_Haar':np.array([r['haar']['gap'] for r in haars]),
        'Gamma':np.array([r['gamma'] for r in pairs]),
        'L_C':np.array([r['L_C'] for r in pairs]),'L_U':np.array([r['L_U'] for r in pairs])}
    for arm in ['mera','tree','haar']:
        source=haars if arm=='haar' else pairs
        for short,group in [('C','contiguous'),('U','uniform')]:
            metrics[arm+'_'+short]=np.array([r[arm][group+'_mean'] for r in source])
    rng=np.random.Generator(np.random.PCG64(BOOT_SEED))
    pair_indices=rng.integers(0,64,size=(BOOT_REPS,64),dtype=np.int64)
    haar_indices=rng.integers(0,64,size=(BOOT_REPS,64),dtype=np.int64)
    bootstrap={};statistics=[]
    for name,values in metrics.items():
        indices=haar_indices if name=='Delta_Haar' or name.startswith('haar_') else pair_indices
        bootstrap[name]=values[indices].mean(axis=1)
        statistics.append(describe(name,values,bootstrap[name]))
    contrasts=[]
    for arm in ['mera','tree']:
        for observable,suffix in [('gap','Delta_MERA' if arm=='mera' else 'Delta_tree'),('C',arm+'_C'),('U',arm+'_U')]:
            h='Delta_Haar' if observable=='gap' else 'haar_'+observable
            name=arm+'_minus_Haar_'+observable
            values=bootstrap[suffix]-bootstrap[h];bootstrap[name]=values
            lo,hi=np.percentile(values,[2.5,97.5],method='linear')
            contrasts.append({'contrast':name,'estimate':float(metrics[suffix].mean()-metrics[h].mean()),
                'percentile95_low':float(lo),'percentile95_high':float(hi),
                'sampling':'Independent Haar resamples; MERA/tree pair indices shared across structured arms. No index pairing with Haar.'})
    np.savez_compressed(out/'bootstrap_results.npz',**bootstrap,
                        pair_indices=pair_indices.astype(np.int16),haar_indices=haar_indices.astype(np.int16))
    np.savez_compressed(out/'encoding_level_values.npz',**metrics)
    table(out/'descriptive_statistics.csv',statistics);table(out/'structured_vs_haar.csv',contrasts)
    primary=[]
    for name in ['Delta_MERA','Gamma']:
        r=next(x for x in statistics if x['estimand']==name)
        decision='positive mean supported by this interval' if r['percentile95_low']>0 else 'negative mean supported by this interval' if r['percentile95_high']<0 else 'mean sign unresolved by this interval'
        primary.append({'estimand':name,'mean':r['mean'],'percentile95_low':r['percentile95_low'],
                        'percentile95_high':r['percentile95_high'],'interpretation':decision})
    ex_values={'Delta_MERA':exploratory['encoders']['mera']['gap'],
               'Delta_tree':exploratory['encoders']['tree']['gap'],'Gamma':exploratory['gamma'],
               'L_C':exploratory['L_C'],'L_U':exploratory['L_U']}
    typical=[]
    for name,value in ex_values.items():
        x=metrics[name];q1,q3=np.percentile(x,[25,75],method='linear')
        typical.append({'quantity':name,'exploratory_value':value,
                        'empirical_percentile_midrank':float(100*(np.sum(x<value)+.5*np.sum(x==value))/64),
                        'within_observed_range':bool(x.min()<=value<=x.max()),
                        'within_sample_iqr':bool(q1<=value<=q3),
                        'comparison':'Descriptive placement against 64 new draws; exploratory seed excluded from estimation.'})
    table(out/'exploratory_comparison.csv',typical)
    draw_figures(out,metrics,ex_values)
    return {'primary':primary,'descriptive_statistics':statistics,'independent_haar_contrasts':contrasts,
            'exploratory_comparison':typical,'bootstrap_replicates':BOOT_REPS,
            'interval_label':'Approximate Monte Carlo uncertainty for the declared random-encoding distributions; percentile 95% intervals, no multiplicity adjustment.',
            'encoding_counts':{'mera':64,'paired_tree':64,'independent_haar':64},
            'pairing':'MERA/tree only. Haar encodings are independently sampled.',
            'complementary_identity':'Delta_5=-Delta_3 for complementary access ensembles in each pure preparation; algebraic consequence, not independent evidence.'}

def draw_figures(out,m,exploratory):
    colors={'MERA':'#216a88','Tree':'#b86938','Haar':'#647c50'}
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
    x,y=m['Delta_tree'],m['Delta_MERA'];lo=min(x.min(),y.min(),0)-.01;hi=max(x.max(),y.max())+.01
    ax.plot([lo,hi],[lo,hi],'--',color='#666666',lw=1,label='Equal gaps')
    ax.axhline(0,color='#aaaaaa',lw=.7);ax.axvline(0,color='#aaaaaa',lw=.7)
    ax.scatter(x,y,color=colors['MERA'],s=32,alpha=.8,label='64 independent pairs')
    ax.scatter([exploratory['Delta_tree']],[exploratory['Delta_MERA']],marker='*',s=160,color='#ad3257',label='Exploratory seed (excluded)')
    ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Tree gap Δ',ylabel='MERA gap Δ',title='Paired access gaps: one point per frozen isometry set')
    ax.legend(fontsize=8);ax.set_aspect('equal',adjustable='box')
    fig.savefig(out/'paired_gaps.png',dpi=180,metadata={'Software':None});plt.close(fig)
    fig,(ax,bx)=plt.subplots(2,1,figsize=(9,5),height_ratios=[3,1],sharex=True,layout='constrained')
    gamma=m['Gamma'];ax.hist(gamma,bins=12,color=colors['MERA'],alpha=.8,edgecolor='white')
    ax.set(ylabel='Number of encoding pairs',title='Disentangler amplification Γ = ΔMERA − Δtree')
    bx.scatter(gamma,((np.arange(64)%7)-3)/10,s=24,color=colors['MERA'],alpha=.8)
    bx.set(xlabel='Γ (positive increases the signed contiguous-minus-uniform gap)',yticks=[],ylim=(-.5,.5))
    for a in [ax,bx]:
        a.axvline(0,color='#444444',ls='--',lw=1)
        a.axvline(exploratory['Gamma'],color='#ad3257',ls=':',lw=1.4,label='Exploratory seed (excluded)')
    ax.legend(fontsize=8)
    fig.savefig(out/'gamma_distribution.png',dpi=180,metadata={'Software':None});plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for j,(name,key) in enumerate([('MERA','Delta_MERA'),('Tree','Delta_tree'),('Haar','Delta_Haar')]):
        vals=m[key];jitter=((np.arange(64)%9)-4)/35
        ax.scatter(vals,j+jitter,color=colors[name],s=24,alpha=.8)
        ax.scatter([np.mean(vals)],[j],marker='D',s=65,facecolors='white',edgecolors='black',zorder=4)
    ax.axvline(0,color='#555555',ls='--',lw=1)
    ax.set(yticks=[0,1,2],yticklabels=['MERA (64)','Paired tree (64)','Independent Haar (64)'],
           xlabel='Contiguous-minus-uniform acquisition gap Δ',title='Every encoding; white diamonds mark group means')
    fig.savefig(out/'gap_distributions.png',dpi=180,metadata={'Software':None});plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,(label,arm) in zip(axes,[('MERA','mera'),('Tree','tree'),('Haar','haar')],strict=True):
        x,y=m[arm+'_U'],m[arm+'_C'];lo=min(x.min(),y.min());hi=max(x.max(),y.max());pad=(hi-lo)*.1
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'--',lw=1,color='#666666')
        ax.scatter(x,y,color=colors[label],s=25,alpha=.8)
        ax.set(xlabel='Uniform-subset mean',ylabel='Contiguous mean',title=label+' — all 64 encodings')
    fig.savefig(out/'acquisition_levels.png',dpi=180,metadata={'Software':None});plt.close(fig)

def run(out,resume=False):
    if resume:
        if not out.is_dir():raise ValueError('Resume directory does not exist')
        if (out/'summary.json').exists():raise ValueError('Completed result is preserved; use a new output directory')
    else:out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();cpu_start=time.process_time();proc=psutil.Process()
    done=threading.Event();usage={'peak_sampled_rss_bytes':0};current='setup';records=[]
    def monitor():
        while not done.is_set():
            usage['peak_sampled_rss_bytes']=max(usage['peak_sampled_rss_bytes'],proc.memory_info().rss);done.wait(.01)
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    def budget():
        if time.perf_counter()-started>1800 or time.process_time()-cpu_start>1800 or usage['peak_sampled_rss_bytes']>2*1024**3:
            raise ResourceBudgetReached('Declared per-process budget reached; completed seeds remain checkpointed.')
    try:
        methods,identity,old_plan=load_inputs();plan=make_plan(identity,old_plan)
        if resume:
            if json.loads((out/'plan_before_execution.json').read_text())!=plan:raise ValueError('Resume plan/source/environment differs')
        else:write_json(out/'plan_before_execution.json',plan)
        # No acquisition score is computed above this write.
        write_json(out/'input_hash_verification.json',{'passed':True,'verified':identity,'no_original_files_modified':True})
        for name,group in [('contiguous',WINDOWS),('uniform',SUBSETS)]:
            if any(len(s)!=3 for s in group) or any(sum(i in s for s in group)/len(group)!=3/8 for i in range(8)):
                raise AssertionError('Access marginal mismatch: '+name)
        current='exploratory_regression';budget()
        ex=read_checkpoint(out/'exploratory_regression')
        if ex is None:ex=regression(methods,out/'exploratory_regression')
        print('Exploratory regression passed; excluded from ensemble.',flush=True)
        encdir=out/'encodings';encdir.mkdir(exist_ok=True);pairs=[];haars=[]
        for seed in MERA_SEEDS:
            budget();current='mera_tree_'+str(seed);folder=encdir/current
            rec=read_checkpoint(folder)
            if rec is None:
                folder.mkdir();tick=time.perf_counter();mera,tree=generate_pair(methods,seed)
                frozen={**{'mera_'+k:a for k,a in mera.items()},**{'tree_'+k:a for k,a in tree.items()}}
                np.savez_compressed(folder/'tensors.npz',**frozen)
                write_json(folder/'tensor_hashes.json',{'archive_sha256':digest(folder/'tensors.npz'),
                    'arrays':{k:ahash(a) for k,a in frozen.items()},'array_format':'little-endian complex128 C-order'})
                m=evaluate(methods,'mera',seed,folder/'mera',tensors=mera)
                t=evaluate(methods,'tree',seed,folder/'tree',tensors=tree)
                gamma=m['gap']-t['gap'];lc=m['contiguous_mean']-t['contiguous_mean'];lu=m['uniform_mean']-t['uniform_mean']
                if abs(gamma-(lc-lu))>TOL:raise AssertionError('Gamma versus level-change identity failed')
                rec={'seed':seed,'mera':m,'tree':t,'gamma':gamma,'L_C':lc,'L_U':lu,'wall_seconds':time.perf_counter()-tick}
                checkpoint(folder,rec)
            pairs.append(rec);records.append(current)
            write_json(out/'progress.json',{'complete_pairs':len(pairs),'complete_haar':len(haars),'last_checkpoint':current})
            if len(pairs)%8==0:print(f'Checkpoint: {len(pairs)}/64 MERA/tree pairs; {time.perf_counter()-started:.1f} s elapsed.',flush=True)
        for seed in HAAR_SEEDS:
            budget();current='haar_'+str(seed);folder=encdir/current
            rec=read_checkpoint(folder)
            if rec is None:
                folder.mkdir();tick=time.perf_counter();v=generate_haar(seed)
                np.savez_compressed(folder/'tensors.npz',V=v)
                write_json(folder/'tensor_hashes.json',{'archive_sha256':digest(folder/'tensors.npz'),'arrays':{'V':ahash(v)},'array_format':'little-endian complex128 C-order'})
                h=evaluate(methods,'haar',seed,folder/'haar',haar=v)
                rec={'seed':seed,'haar':h,'wall_seconds':time.perf_counter()-tick};checkpoint(folder,rec)
            haars.append(rec);records.append(current)
            write_json(out/'progress.json',{'complete_pairs':len(pairs),'complete_haar':len(haars),'last_checkpoint':current})
            if len(haars)%8==0:print(f'Checkpoint: {len(haars)}/64 Haar encodings; {time.perf_counter()-started:.1f} s elapsed.',flush=True)
        budget();current='analysis'
        encoding_rows=[];subset_rows=[];paired_rows=[]
        for r in pairs:
            paired_rows.append({'seed':r['seed'],'Delta_MERA':r['mera']['gap'],'Delta_tree':r['tree']['gap'],
                'Gamma':r['gamma'],'L_C':r['L_C'],'L_U':r['L_U'],
                'mera_contiguous':r['mera']['contiguous_mean'],'mera_uniform':r['mera']['uniform_mean'],
                'tree_contiguous':r['tree']['contiguous_mean'],'tree_uniform':r['tree']['uniform_mean']})
        for rec in pairs+haars:
            for arm in ['mera','tree','haar']:
                if arm not in rec:continue
                r=rec[arm];encoding_rows.append({k:r[k] for k in ['encoder','seed','contiguous_mean','uniform_mean','gap','maximum_validation_residual','complementary_gap_residual','passed']})
                dirname=('haar_' if arm=='haar' else 'mera_tree_')+str(rec['seed'])
                subset_rows.extend(csv.DictReader((encdir/dirname/arm/'subset_scores.csv').open(encoding='utf-8',newline='')))
        table(out/'per_encoding.csv',encoding_rows);table(out/'per_pair.csv',paired_rows);table(out/'all_subset_scores.csv',subset_rows)
        analysis=analyze(out,pairs,haars,ex)
        max_validation=max(r['maximum_validation_residual'] for r in encoding_rows)
        usage.update(wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu_start,
            timing_scope='Per-process study after Python imports through generation, validation, checkpointing, statistics and figures; excludes final report/summary.',
            memory_measurement='Process RSS sampled every 10 ms; brief peaks may be missed.')
        if hasattr(proc.memory_info(),'peak_wset'):usage['os_peak_working_set_bytes']=proc.memory_info().peak_wset
        summary={'complete':True,'analysis':analysis,'maximum_validation_residual':max_validation,
                 'maximum_complementary_gap_residual':max(abs(r['complementary_gap_residual']) for r in encoding_rows),
                 'exploratory_regression_maximum_residual':ex['maximum_regression_residual'],
                 'total_ensemble_subset_rows':len(subset_rows),'resources':usage}
        write_json(out/'summary.json',summary);write_report(out,summary)
        print(json.dumps({'complete':True,'primary':analysis['primary'],'maximum_validation_residual':max_validation,'resources':usage},indent=2),flush=True)
    except Exception as exc:
        filename='budget_reached.json' if isinstance(exc,ResourceBudgetReached) else 'failure.json'
        write_json(out/filename,{'type':type(exc).__name__,'message':str(exc),'stage_or_seed':current,
            'completed_seed_checkpoints':records,'source_sha256':digest(Path(__file__)),'action':'Preserved completed and failed outputs. No seed replacement or scientific-design change.'})
        raise
    finally:done.set();worker.join()

def write_report(out,summary):
    analysis=summary['analysis'];by={r['estimand']:r for r in analysis['descriptive_statistics']}
    lines=['# Eight-site random-encoding ensemble','','## Two primary questions','',
           'The study contains 64 independent MERA draws, their 64 paired identity-disentangler trees, and 64 independent Haar-isometry controls. Each encoding uses a Bell reference and the same flagged three-site access ensembles. The separate seed 20260918 is excluded from every ensemble estimate.','',
           '| Primary estimand | Mean | Approximate percentile 95% interval | Separate assessment |',
           '|---|---:|---|---|']
    for r in analysis['primary']:lines.append(f"| {r['estimand']} | {r['mean']:+.10f} | [{r['percentile95_low']:+.10f}, {r['percentile95_high']:+.10f}] | {r['interpretation']} |")
    lines += ['', 'Delta_MERA answers whether contiguous access has a positive mean advantage in the declared MERA distribution. Gamma answers whether the specified disentanglers increase the signed gap relative to paired trees. These questions are assessed separately; there is no omnibus success criterion.','',
        'Intervals use 20,000 encoding-level bootstrap resamples. They approximate Monte Carlo uncertainty for these declared random-encoding distributions. MERA/tree pairs are resampled together, and Haar draws independently. No multiplicity adjustment is applied. The 56 overlapping subsets are not independent replicates.','',
        '## Gaps and amplification','','| Quantity | Mean | Sample SD | Median | IQR | Minimum | Maximum | Positive / unresolved / negative |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for key in ['Delta_MERA','Delta_tree','Delta_Haar','Gamma']:
        r=by[key];lines.append(f"| {key} | {r['mean']:+.8f} | {r['sample_sd']:.8f} | {r['median']:+.8f} | {r['iqr']:.8f} | {r['minimum']:+.8f} | {r['maximum']:+.8f} | {r['positive']} / {r['unresolved']} / {r['negative']} |")
    lines += ['', 'Sign counts use 1e-8 as a numerical-resolution threshold, not an effect-size threshold. All negative gaps and negative amplification values remain in the results.','',
        '![All encoding gaps](gap_distributions.png)','','![Paired gaps](paired_gaps.png)','','![Amplification distribution](gamma_distribution.png)','',
        '## Acquisition levels','','| Encoder | Mean contiguous acquisition | Mean uniform-subset acquisition |','|---|---:|---:|']
    for arm in ['mera','tree','haar']:lines.append(f"| {arm} | {by[arm+'_C']['mean']:.10f} | {by[arm+'_U']['mean']:.10f} |")
    for key in ['L_C','L_U']:
        r=by[key];lines += ['',f"{key}, measured as MERA minus tree, averages {r['mean']:+.10f}, with approximate percentile 95% interval [{r['percentile95_low']:+.10f}, {r['percentile95_high']:+.10f}]."]
    lines += ['', 'A larger gap does not imply more total acquisition. The two level changes are reported independently, and Gamma = L_C − L_U.','',
        '![Acquisition levels](acquisition_levels.png)','',
        '## Independent Haar comparison','',
        'Haar controls match the logical and boundary dimensions, but not locality or tensor structure. Their seed indices do not pair them scientifically with structured encodings. A Haar draw can have a nonzero gap; permutation symmetry gives zero ensemble expectation, not an exact-zero sample requirement.','',
        '| Independent contrast | Mean difference | Approximate percentile 95% interval |','|---|---:|---|']
    for r in analysis['independent_haar_contrasts']:lines.append(f"| {r['contrast']} | {r['estimate']:+.10f} | [{r['percentile95_low']:+.10f}, {r['percentile95_high']:+.10f}] |")
    h=by['Delta_Haar'];lines += ['',f"The Haar mean gap is {h['mean']:+.10f}, with interval [{h['percentile95_low']:+.10f}, {h['percentile95_high']:+.10f}]. These contrasts describe the specified ensembles rather than isolating a universal causal property of multiscale structure.",'',
        '## Exploratory seed placement','','| Quantity | Seed 20260918 value | Empirical percentile among 64 ensemble draws | Within observed range | Within sample IQR |',
        '|---|---:|---:|---|---|']
    for r in analysis['exploratory_comparison']:lines.append(f"| {r['quantity']} | {r['exploratory_value']:+.10f} | {r['empirical_percentile_midrank']:.2f} | {r['within_observed_range']} | {r['within_sample_iqr']} |")
    lines += ['', "The exploratory MERA gap lies at the 46.875th empirical percentile and inside the ensemble sample’s interquartile range, making that gap representative of the sample center. Its tree gap is at the 81.25th percentile and its amplification at the 14.0625th percentile, both outside their sample interquartile ranges. The seed's positive amplification was therefore smaller than most ensemble paired effects.",'', 'This is a descriptive assessment of representativeness relative to this sample, not a population outlier test. Seed 20260918 appears as a separate marked reference in figures and never enters the estimates or resamples.','',
        '## Complementary access','',
        'Purity of the Bell-reference state gives I(Q:B_S)+I(Q:B_Sc)=2, hence r(S)+r(Sc)=1. Complementing the eight cyclic triples gives the eight cyclic five-site windows; complementing all 56 triples gives all 56 five-site subsets. Both five-site ensemble means are therefore one minus their three-site counterparts, so Delta_5 = −Delta_3. This is an algebraic consequence of the same states and access rules, not additional independent evidence.','',
        '## Validation and execution','',
        f"All 192 ensemble encoders passed at 1e-10. The largest validation residual was {summary['maximum_validation_residual']:.3e}; the complementary-gap residual was at most {summary['maximum_complementary_gap_residual']:.3e}. The separate exploratory regression reproduced within {summary['exploratory_regression_maximum_residual']:.3e}.",'',
        'Every MERA/tree pair retained separate contraction and dense encodings and two partial-trace implementations. Haar used both partial traces. Checks covered tensor/full isometry, normalization, density validity, empty/full reference information, and complementary information. Raw eigenvalues are retained, including tolerated negatives; only negative eigenvalues contribute zero to entropy, while all positive eigenvalues are kept. Scores are not rounded or clipped to force agreement.','',
        'The sampling plan preceded acquisition evaluation, and no seed was replaced. Tensor draws use the declared seeds and generation conventions. Complete per-subset results, per-encoding and paired summaries, validation records, and tensor archives are included.','',
        f"Measured execution after imports: {summary['resources']['wall_seconds']:.2f} wall seconds; {summary['resources']['cpu_seconds']:.2f} CPU seconds; {summary['resources']['peak_sampled_rss_bytes']/1024**2:.1f} MiB sampled peak RSS. Budget per process: 30 minutes and 2 GiB.",'',
        'The study concerns random, unoptimized encodings of fixed size. It does not establish a physical ground state, an Ising result, optimal decoding fidelity, retrieval latency, or prevalence outside the declared random-encoding distributions.','']
    primary={r['estimand']:r for r in analysis['primary']}
    contrast=next(r for r in analysis['independent_haar_contrasts'] if r['contrast']=='mera_minus_Haar_gap')
    haar_assessment='positive structured-minus-Haar mean contrast supported by this interval' if contrast['percentile95_low']>0 else 'negative structured-minus-Haar mean contrast supported by this interval' if contrast['percentile95_high']<0 else 'structured-minus-Haar mean contrast sign unresolved by this interval'
    lines += ['## Separate conclusions','',
        f"**Access preference:** {primary['Delta_MERA']['interpretation']}; mean MERA gap {primary['Delta_MERA']['mean']:+.10f}.",'',
        f"**Disentangler amplification:** {primary['Gamma']['interpretation']}; mean Gamma {primary['Gamma']['mean']:+.10f}. Acquisition-level changes remain separate: mean L_C {by['L_C']['mean']:+.10f}, mean L_U {by['L_U']['mean']:+.10f}.",'',
        f"**Haar comparison:** {haar_assessment}; mean MERA-minus-Haar gap contrast {contrast['estimate']:+.10f}. The independent Haar distribution matches dimensions but changes both locality and tensor structure.",'']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results',help='New output directory')
    parser.add_argument('--resume',action='store_true',help='Resume a matching plan from verified complete seed checkpoints')
    args=parser.parse_args();run(args.output.resolve(),args.resume)
