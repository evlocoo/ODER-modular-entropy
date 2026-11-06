"""Eight-site frozen-seed MERA access pilot. Run: python reproduce.py"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import csv, hashlib, itertools, json, platform, sys, time, threading
from pathlib import Path
import numpy as np
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
TOL=1e-10
SEED=20260918
PAIRS={4:[(1,2),(3,0)],8:[(1,2),(3,4),(5,6),(7,0)]}
def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def generate():
    rng=np.random.Generator(np.random.PCG64(SEED)); tensors={}
    for layer,count in enumerate([1,2,4]):
        for j in range(count):
            a=rng.standard_normal((4,2))+1j*rng.standard_normal((4,2))
            q,r=np.linalg.qr(a,mode='reduced');d=np.diag(r);q=q*(d/np.abs(d))[None,:]
            tensors[f'w{layer}_{j}']=q
        n=2*count
        for j,pair in enumerate(PAIRS.get(n,[])):
            a=rng.standard_normal((4,4))+1j*rng.standard_normal((4,4))
            q,r=np.linalg.qr(a);d=np.diag(r);q=q*(d/np.abs(d))[None,:]
            tensors[f'u{n}_{j}']=q
    return tensors

def contract_network(t):
    # One explicit index network: free legs [physical 0,...,7,logical].
    wire=[0];next_leg=1;args=[];edges=[]
    for layer,count in enumerate([1,2,4]):
        new=[]
        for j in range(count):
            legs=[next_leg,next_leg+1];next_leg+=2
            args.extend([t[f'w{layer}_{j}'].reshape(2,2,2),legs+[wire[j]]]);new+=legs
        wire=new
        for j,(a,b) in enumerate(PAIRS.get(len(wire),[])):
            legs=[next_leg,next_leg+1];next_leg+=2
            args.extend([t[f'u{len(wire)}_{j}'].reshape(2,2,2,2),legs+[wire[a],wire[b]]])
            wire[a],wire[b]=legs
    return np.einsum(*args,wire+[0],optimize='greedy').reshape(256,2)

def dense_encoding(t):
    # Full expansion matrices and explicitly embedded gate matrices; no einsum.
    v=np.eye(2,dtype=complex)
    for layer,count in enumerate([1,2,4]):
        expansion=np.ones((1,1),complex)
        for j in range(count):expansion=np.kron(expansion,t[f'w{layer}_{j}'])
        v=expansion@v;n=2*count;dim=2**n
        for j,(a,b) in enumerate(PAIRS.get(n,[])):
            gate=np.zeros((dim,dim),complex);u=t[f'u{n}_{j}']
            for col in range(dim):
                bits=[(col>>(n-1-k))&1 for k in range(n)]
                inp=2*bits[a]+bits[b]
                for out in range(4):
                    new=bits.copy();new[a]=out//2;new[b]=out%2
                    row=sum(x<<(n-1-k) for k,x in enumerate(new))
                    gate[row,col]=u[out,inp]
            v=gate@v
    return v

def reduced_tensor(psi,keep):
    rest=[k for k in range(9) if k not in keep]
    a=psi.reshape([2]*9).transpose(keep+rest).reshape(2**len(keep),-1)
    return a@a.conj().T

def reduced_dense(psi,keep):
    # Explicit environment-index sums, independent of reshape/transpose reduction.
    rest=[k for k in range(9) if k not in keep];d=2**len(keep)
    rho=np.zeros((d,d),complex)
    for env in range(2**len(rest)):
        base=sum(((env>>(len(rest)-1-j))&1)<<(8-k) for j,k in enumerate(rest))
        indices=[base+sum(((a>>(len(keep)-1-j))&1)<<(8-k) for j,k in enumerate(keep)) for a in range(d)]
        v=psi[indices];rho+=np.outer(v,v.conj())
    return rho

def main():
    start=time.perf_counter();cpu_start=time.process_time();OUT.mkdir(exist_ok=True)
    if (OUT/'summary.json').exists():raise RuntimeError('Use a new --output directory to preserve previous runs.')
    usage={'peak_sampled_rss_bytes':0};done=threading.Event();proc=psutil.Process()
    def monitor():
        while not done.is_set():
            usage['peak_sampled_rss_bytes']=max(usage['peak_sampled_rss_bytes'],proc.memory_info().rss)
            if usage['peak_sampled_rss_bytes']>2*1024**3 or time.process_time()-cpu_start>600:os._exit(3)
            done.wait(.01)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    subsets=list(itertools.combinations(range(8),3))
    windows={tuple(sorted((i,(i+1)%8,(i+2)%8))) for i in range(8)}
    plan={'architecture':{'physical_dimension':2,'chi':2,'expansion_counts':[1,2,4],'disentangler_pairs':PAIRS,'top_disentangler':False},'seed':SEED,'rng':'numpy Generator(PCG64)','generation_order':'For each expansion layer: w left to right, then u in listed pair order. Each matrix draws all real entries then all imaginary entries, in C order.','qr_convention':'Reduced QR; multiply column j of Q by R[j,j]/abs(R[j,j]), giving positive real diagonal of the corresponding R. No zero diagonals occurred.','legs':'w[out_left,out_right,in]; u[out_first,out_second,in_first,in_second]. Ordered pair (7,0) means first leg site 7. C-order reshape; logical and physical site 0 are most significant bits.','source':'Bell(Q,A) then V on A; state axes [Q,B0,...,B7]. Unoptimized encoding.','ensembles':{'contiguous':[list(s) for s in sorted(windows)],'uniform':[list(s) for s in subsets],'weights':[1/8,1/56],'selection':'Input independent; subset identity retained as classical flag'},'statistic':'Delta = contiguous mean I(Q:B_S)/2 minus uniform-subset mean I(Q:B_S)/2','agreement_tolerance':TOL,'resolved_threshold':1e-8,'norms':'Frobenius for matrices; Euclidean for states; absolute error for scalars; eigenvalue lower bound -1e-10.','entropy':'Base two; record raw eigenvalues; reject values below -1e-10; set only negative eigenvalues to zero, retaining every positive eigenvalue. No trace renormalization.','completion':'All tensor, encoding, state, reduced-state, marginal, independent-path and pure-complement checks pass; deliver either signed resolved result or unchanged null.','resource_budget':{'cpu_seconds':600,'rss_bytes':2*1024**3,'monitor_interval_seconds':.01},'provenance':{'source_sha256':digest(Path(__file__)),'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'psutil':psutil.__version__,'platform':platform.system()},'identity_note':'File hashes identify artifacts; they do not prove independently timestamped preregistration.'}
    write_json(OUT/'plan_before_execution.json',plan)
    tensors=generate();np.savez(OUT/'frozen_tensors.npz',**tensors)
    write_json(OUT/'tensor_hashes.json',{'archive_sha256':digest(OUT/'frozen_tensors.npz'),'arrays':{k:hashlib.sha256(v.astype('<c16').tobytes(order='C')).hexdigest() for k,v in tensors.items()},'array_hash_format':'little-endian complex128 C-order bytes; shapes in archive'})
    # All access evaluation occurs after the preceding writes have closed.
    checks={};arrays={};eigen_rows=[]
    def check(name,value):
        checks[name]=float(value)
        if not np.isfinite(value) or value>TOL:raise AssertionError(f'{name}: {value}')
    for k,a in tensors.items():check(k+'_isometry_fro',np.linalg.norm(a.conj().T@a-np.eye(a.shape[1])))
    vt=contract_network(tensors);vd=dense_encoding(tensors)
    check('encoding_agreement_fro',np.linalg.norm(vt-vd))
    for name,v in [('network',vt),('dense',vd)]:check(name+'_encoding_isometry_fro',np.linalg.norm(v.conj().T@v-np.eye(2)))
    pt=vt.T.reshape(-1)/np.sqrt(2);pd=vd.T.reshape(-1)/np.sqrt(2)
    arrays.update(V_network=vt,V_dense=vd,psi_network=pt,psi_dense=pd)
    check('state_agreement_l2',np.linalg.norm(pt-pd))
    for name,p in [('network',pt),('dense',pd)]:check(name+'_normalization',abs(np.vdot(p,p)-1))
    def entropy(rho,key):
        check(key+'_trace',abs(np.trace(rho)-1));check(key+'_hermiticity_fro',np.linalg.norm(rho-rho.conj().T))
        eig=np.linalg.eigvalsh((rho+rho.conj().T)/2)
        check(key+'_negative_eigenvalue',max(0.,-eig.min()))
        eigen_rows.extend({'state':key,'index':i,'raw_eigenvalue':float(x)} for i,x in enumerate(eig))
        arrays[key]=rho;arrays[key+'_eigenvalues']=eig
        positive=eig[eig>0];return float(-np.sum(positive*np.log2(positive)))
    def info(p,red,s,key):
        sq=entropy(red(p,[0]),key+'_Q')
        sb=entropy(red(p,[i+1 for i in s]),key+'_B')
        sqb=entropy(red(p,[0]+[i+1 for i in s]),key+'_QB')
        check(key+'_reference_entropy',abs(sq-1))
        return sq+sb-sqb
    for name,p,red in [('network',pt,reduced_tensor),('dense',pd,reduced_dense)]:
        check(name+'_empty_information',abs(info(p,red,[],name+'_empty')))
        check(name+'_full_information',abs(info(p,red,list(range(8)),name+'_full')-2))
    rows=[]
    for s in subsets:
        key=''.join(map(str,s));a=info(pt,reduced_tensor,list(s),'network_'+key);b=info(pd,reduced_dense,list(s),'dense_'+key)
        check(key+'_rho_agreement_fro',np.linalg.norm(arrays['network_'+key+'_QB']-arrays['dense_'+key+'_QB']))
        check(key+'_score_agreement',abs(a-b)/2)
        comp=[i for i in range(8) if i not in s]
        for name,p,red,x in [('network',pt,reduced_tensor,a),('dense',pd,reduced_dense,b)]:
            c=info(p,red,comp,name+'_complement_'+key);check(name+'_'+key+'_pure_complement',abs(x+c-2))
        rows.append({'subset':' '.join(map(str,s)),'contiguous':s in windows,'r_network':a/2,'r_dense':b/2,'I_network_bits':a,'I_dense_bits':b})
    marginals={name:[sum(i in s for s in group)/len(group) for i in range(8)] for name,group in [('contiguous',windows),('uniform',subsets)]}
    for name,m in marginals.items():check(name+'_marginal',max(abs(x-3/8) for x in m))
    means={path:{'contiguous':float(np.mean([r['r_'+path] for r in rows if r['contiguous']])),'uniform':float(np.mean([r['r_'+path] for r in rows]))} for path in ['network','dense']}
    delta=means['network']['contiguous']-means['network']['uniform']
    check('delta_agreement',abs(delta-(means['dense']['contiguous']-means['dense']['uniform'])))
    summary={'validation_passed':True,'means':means,'delta':delta,'decision':'resolved difference in this encoding' if abs(delta)>1e-8 else 'no resolved difference in this encoding','marginal_inclusion_probabilities':marginals,'subset_count':56,'window_count':8,'maximum_validation_residual':max(checks.values())}
    table(OUT/'subset_scores.csv',rows);table(OUT/'raw_eigenvalues.csv',eigen_rows)
    np.savez_compressed(OUT/'supporting_arrays.npz',**arrays)
    write_json(OUT/'validation.json',{'passed':True,'tolerance':TOL,'checks':checks,'minimum_raw_eigenvalue':min(x['raw_eigenvalue'] for x in eigen_rows)})
    draw_network(OUT/'network.png')
    done.set();thread.join();usage.update(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu_start,memory_measurement='Process RSS sampled every 10 ms; includes Python and imported libraries. Sampling may miss shorter peaks.')
    if hasattr(proc.memory_info(),'peak_wset'):usage['os_peak_working_set_bytes']=proc.memory_info().peak_wset
    summary['resources']=usage;write_json(OUT/'summary.json',summary)
    report=f'''# Finite MERA access pilot\n\nThis unoptimized eight-site encoding tests whether the arrangement of acquired sites changes reference information at matched site count and marginal access. Both ensembles acquire three sites and include every site with probability 3/8. The subset identity remains available as a classical flag.\n\n## Result\n\n| Ensemble | Mean I(Q:B_S)/2 |\n|---|---:|\n| Eight cyclic windows | {means['network']['contiguous']:.12f} |\n| All 56 three-site subsets | {means['network']['uniform']:.12f} |\n\nThe signed difference (contiguous minus uniform) is **{delta:+.12f}**: {summary['decision']}. The numerical decision threshold is 1e-8, conditional on all validation checks passing; it is not a rigorous error certificate.\n\n## Reliability\n\nA single explicit index contraction and a separate dense implementation using Kronecker expansions and embedded gate matrices agree. Reduced states are calculated by tensor matricization in one path and explicit environment sums in the other. All 56 reference-plus-subset states and information scores pass the 1e-10 checks. The largest recorded validation residual is {max(checks.values()):.3e}. Checks include tensor and encoding isometry, normalization, trace, Hermiticity, positivity, empty/full access, and pure-state complementary information. Raw eigenvalues are retained; only tolerated negative eigenvalues are excluded from entropy sums. Every positive eigenvalue contributes.\n\nThe plan and tensor hashes were written before access evaluation. Hashes establish file identity, not externally timestamped preregistration. This run used seed 20260918 without optimization or outcome selection. Detailed conventions are in plan_before_execution.json.\n\nRuntime: {usage['wall_seconds']:.2f} wall seconds; {usage['cpu_seconds']:.2f} CPU seconds. Sampled peak resident memory: {usage['peak_sampled_rss_bytes']/1024**2:.1f} MiB. Budget: 600 CPU seconds and 2 GiB.\n\n## Scope\n\nThe result concerns flagged-channel reference information for this frozen MERA encoding. It does not establish an optimal-recovery fidelity difference, typical behavior across encodings, or physical retrieval latency. This is not an optimized critical-Ising state. A subsequent identity-disentangler comparison could ask how much of the effect depends on disentanglers; that experiment has not been run.\n\nSee subset_scores.csv for every score and the eight window labels, validation.json for checks, supporting_arrays.npz and raw_eigenvalues.csv for numerical evidence, and network.png for the actual connectivity.\n'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8');print(json.dumps(summary,indent=2))

def draw_network(path):
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(12,7));ax.set_xlim(-1,8);ax.set_ylim(-.5,7);ax.axis('off')
    xs=[3.5];y=6.5;ax.text(3.5,6.8,'Logical input A (Bell-paired with Q)',ha='center')
    for layer,count in enumerate([1,2,4]):
        ny=y-1.;newxs=[(j+.5)*8/(2*count)-.5 for j in range(2*count)]
        for j in range(count):
            ax.plot([xs[j],xs[j]],[y,ny+.25],color='#34495e');ax.text(xs[j],ny+.12,f'w{layer}_{j}',ha='center',bbox=dict(facecolor='#d9edf7',edgecolor='#34495e'))
            for x in newxs[2*j:2*j+2]:ax.plot([xs[j],x],[ny,ny-.35],color='#34495e')
        xs=newxs;y=ny-.35;n=len(xs)
        if n in PAIRS:
            gy=y-.45
            for x in xs:ax.plot([x,x],[y,gy-.2],color='#34495e')
            for j,(a,b) in enumerate(PAIRS[n]):
                if b==0:
                    ax.plot([xs[a],8,8,-.8,-.8,xs[b]],[gy,gy,gy-.3,gy-.3,gy,gy],color='#a33f23',lw=2)
                    ax.text(7.8,gy-.25,f'u{n}_{j}',ha='right',color='#a33f23')
                else:
                    ax.plot([xs[a],xs[b]],[gy,gy],lw=6,color='#d8ad60');ax.text((xs[a]+xs[b])/2,gy+.13,f'u{n}_{j}',ha='center',fontsize=9)
            y=gy-.35
    for i,x in enumerate(xs):ax.text(x,y-.15,f'B{i}',ha='center')
    ax.set_title('Eight-site binary MERA: seven isometries and six disentanglers\nWraparound gates connect (3,0) and (7,0); crossing lines have no junction',fontsize=12)
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--output':OUT=Path(sys.argv[2]).resolve()
    elif len(sys.argv)!=1:raise SystemExit('Usage: python reproduce.py [--output DIRECTORY]')
    main()
