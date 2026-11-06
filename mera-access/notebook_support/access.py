"""Notebook adapters for encoding, reduction, and verification."""
from pathlib import Path
import csv, hashlib, importlib.util, itertools, json, math, os, subprocess, sys, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

TOL = 1e-10
RESOLUTION = 1e-8
ARMS = ('mera', 'tree', 'haar')
LABELS = {'mera': 'MERA', 'tree': 'Paired tree', 'haar': 'Independent Haar'}
COLORS = {'mera': '#216a88', 'tree': '#b86938', 'haar': '#647c50'}
SUBSETS = list(itertools.combinations(range(8), 3))
WINDOWS = {tuple(sorted((i, (i+1) % 8, (i+2) % 8))) for i in range(8)}

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write_json(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

def require(p):
    p = Path(p)
    if not p.is_file():
        raise FileNotFoundError(f'Required file missing: {p.name}. Extract the complete companion ZIP and keep mera_ensemble_study/ beside this notebook. Expected relative path: ' + '/'.join(p.parts[-3:]))
    return p

def digest(p):
    return hashlib.sha256(require(p).read_bytes()).hexdigest()

def check(value, name, tolerance=TOL):
    value = float(value)
    if not np.isfinite(value) or value > tolerance:
        raise AssertionError(f'{name}: {value:.17g} exceeds {tolerance}')
    return value

def verify_originals(root):
    manifest = read_json(require(root/'verification/source_identity.json'))
    for name, h in manifest['files'].items():
        if digest(root/name) != h:
            raise AssertionError('Package file integrity mismatch: ' + name)
    return {'passed': True, 'files_checked': len(manifest['files'])}

def load_backend(study):
    identity = read_json(require(study/'input_identity.json'))
    for name, rec in identity['files'].items():
        if digest(study/'inputs'/name) != rec['sha256']:
            raise AssertionError('Audited input changed: ' + name)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('mera_audited_methods', study/'inputs/original_methods.py')
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)  # main() is guarded in the archived source.
    return backend

def select_results(root, mode, run_name):
    study = root/'mera_ensemble_study'
    require(study/'results/summary.json')
    if mode == 'saved':
        print('Displayed ensemble: supplied study results. Worked-example and statistical checks execute below.')
        return study/'results', None
    if mode != 'regenerate':
        raise ValueError("MODE must be 'saved' or 'regenerate'.")
    if Path(run_name).name != run_name or run_name in ('', '.', '..'):
        raise ValueError('RUN_NAME must be a single new directory name.')
    target = root/'outputs'/run_name
    if target.exists():
        raise FileExistsError('Choose a new RUN_NAME. Existing runs are preserved: ' + run_name)
    print('Regenerating 64 MERA/tree pairs and 64 independent Haar encodings; progress follows.', flush=True)
    log = root/'verification'/(run_name+'_process.log')
    with log.open('w', encoding='utf-8') as f:
        process = subprocess.Popen([sys.executable, '-u', str(root/'notebook_support/verify_regeneration.py'), '--output', str(target)],
                                   cwd=study, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            f.write(line); f.flush()
            print(line.rstrip(), flush=True)
        status = process.wait()
    if status:
        raise RuntimeError(f'Regeneration failed with exit code {status}. Inspect outputs/{run_name}/companion_verification.json for comparison locations, or verification/{log.name} for a study-process failure. Keep the result directory for diagnosis.')
    report = read_json(target/'companion_verification.json')
    if not report['passed']:
        raise AssertionError('Fresh-process comparison did not pass.')
    write_json(root/'verification/current_regeneration.json', report)
    print('Downstream analysis now uses outputs/' + run_name + '/, including its newly generated tensors.')
    return target, report

def read_csv(path):
    with require(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def collection():
    membership = np.array([[int(i in s) for s in SUBSETS] for i in range(8)], dtype=float)
    wc = np.array([1/8 if s in WINDOWS else 0 for s in SUBSETS])
    wu = np.full(56, 1/56)
    for name, w in [('contiguous', wc), ('uniform', wu)]:
        check(abs(w.sum()-1), name+' normalization')
        check(np.max(np.abs(membership@w-3/8)), name+' inclusion probabilities')
    return membership, wc, wu

def analyze(results):
    rows = read_csv(results/'all_subset_scores.csv')
    groups = {}
    for row in rows:
        groups.setdefault((row['encoder'], int(row['seed'])), []).append(row)
    expected = {(a, s) for a in ARMS for s in (range(202001, 202065) if a == 'haar' else range(101001, 101065))}
    if set(groups) != expected or len(rows) != 10752:
        raise AssertionError('Expected exactly 192 declared encodings and 10,752 subset rows.')
    residuals = {}
    means = {}
    for key, group in groups.items():
        if len(group) != 56 or {tuple(map(int, r['subset'].split())) for r in group} != set(SUBSETS):
            raise AssertionError('Incomplete subset inventory: ' + str(key))
        for r in group:
            if (r['contiguous'] == 'True') != (tuple(map(int, r['subset'].split())) in WINDOWS):
                raise AssertionError('Incorrect contiguous label')
        c = [float(r['r_network']) for r in group if r['contiguous'] == 'True']
        if len(c) != 8: raise AssertionError('Expected eight cyclic windows')
        C, U = math.fsum(c)/8, math.fsum(float(r['r_network']) for r in group)/56
        means[key] = {'C': C, 'U': U, 'gap': C-U}
    for r in read_csv(results/'per_encoding.csv'):
        key = (r['encoder'], int(r['seed']))
        for field, short in [('contiguous_mean', 'C'), ('uniform_mean', 'U'), ('gap', 'gap')]:
            residuals[str(key)+'_'+short] = check(abs(means[key][short]-float(r[field])), field)
    m = {}
    for a in ARMS:
        seeds = range(202001, 202065) if a == 'haar' else range(101001, 101065)
        for short in ('C', 'U', 'gap'):
            name = ('Delta_'+('MERA' if a == 'mera' else 'Haar' if a == 'haar' else 'tree')) if short == 'gap' else a+'_'+short
            m[name] = np.array([means[a, s][short] for s in seeds])
    m['Gamma'] = m['Delta_MERA']-m['Delta_tree']
    m['L_C'] = m['mera_C']-m['tree_C']
    m['L_U'] = m['mera_U']-m['tree_U']
    for i, r in enumerate(read_csv(results/'per_pair.csv')):
        if int(r['seed']) != 101001+i: raise AssertionError('Pair ordering differs')
        for name in ('Delta_MERA', 'Delta_tree', 'Gamma', 'L_C', 'L_U'):
            residuals[f'{i}_{name}'] = check(abs(m[name][i]-float(r[name])), name)
    rng = np.random.Generator(np.random.PCG64(303001))
    pi = rng.integers(0, 64, size=(20000, 64), dtype=np.int64)
    hi = rng.integers(0, 64, size=(20000, 64), dtype=np.int64)
    boot, stats = {}, {}
    declared = {r['estimand']: r for r in read_csv(results/'descriptive_statistics.csv')}
    with np.load(require(results/'bootstrap_results.npz'), allow_pickle=False) as z:
        if not np.array_equal(pi, z['pair_indices']) or not np.array_equal(hi, z['haar_indices']):
            raise AssertionError('Archived bootstrap indices differ from declared draws.')
        for name, values in m.items():
            ix = hi if name == 'Delta_Haar' or name.startswith('haar_') else pi
            boot[name] = values[ix].mean(axis=1)
            residuals[name+'_bootstrap'] = check(np.max(np.abs(boot[name]-z[name])), name+' bootstrap')
            q = np.percentile(values, [25, 75], method='linear')
            interval = np.percentile(boot[name], [2.5, 97.5], method='linear')
            stats[name] = dict(mean=float(values.mean()), sample_sd=float(values.std(ddof=1)), median=float(np.median(values)),
                               q1=float(q[0]), q3=float(q[1]), iqr=float(q[1]-q[0]), minimum=float(values.min()), maximum=float(values.max()),
                               positive=int(np.sum(values>RESOLUTION)), negative=int(np.sum(values < -RESOLUTION)),
                               unresolved=int(np.sum(np.abs(values)<=RESOLUTION)), percentile95_low=float(interval[0]), percentile95_high=float(interval[1]))
            for field, value in stats[name].items():
                residuals[name+'_'+field] = check(abs(value-float(declared[name][field])), name+' '+field)
    data = np.array([[float(r[k]) for k in ('r_network', 'r_dense', 'complement_r_network', 'complement_r_dense')] for r in rows])
    residuals['score_paths'] = check(np.max(np.abs(data[:, 0]-data[:, 1])), 'subset score paths')
    residuals['complement_paths'] = check(np.max(np.abs(data[:, 2]-data[:, 3])), 'complement paths')
    residuals['complement_identity'] = check(np.max(np.abs(data[:, :2]+data[:, 2:]-1)), 'pure complements')
    if data.min() < -TOL or data.max() > 1+TOL: raise AssertionError('Score outside allowed range')
    report = {'passed': True, 'encodings': 192, 'subset_rows': len(rows), 'bootstrap_replicates': 20000,
              'indices_match': True, 'maximum_absolute_disagreement': max(residuals.values()), 'checks': residuals,
              'method': 'Reaggregate CSVs with math.fsum; independent PCG64 resampling; compare all 12 bootstrap distributions and descriptive statistics.'}
    return {'rows': rows, 'groups': groups, 'metrics': m, 'statistics': stats, 'report': report, 'bootstrap': boot}

def entropy(rho):
    check(abs(np.trace(rho)-1), 'density trace')
    check(np.linalg.norm(rho-rho.conj().T), 'density Hermiticity')
    eig = np.linalg.eigvalsh((rho+rho.conj().T)/2)
    check(max(0., -eig.min()), 'negative eigenvalue')
    p = eig[eig>0]
    return float(-np.sum(p*np.log2(p))), eig

def example(study, results, data, seed=101001):
    backend = load_backend(study)
    folder = results/'encodings'/f'mera_tree_{seed}'
    hashes = read_json(require(folder/'tensor_hashes.json'))
    if digest(folder/'tensors.npz') != hashes['archive_sha256']: raise AssertionError('Frozen tensor archive mismatch')
    with np.load(folder/'tensors.npz', allow_pickle=False) as z:
        all_t = {k: z[k] for k in z.files}
    for k, a in all_t.items():
        if hashlib.sha256(a.astype('<c16').tobytes(order='C')).hexdigest() != hashes['arrays'][k]:
            raise AssertionError('Frozen tensor mismatch: '+k)
    for k in all_t:
        if k.startswith('mera_w') and not np.array_equal(all_t[k], all_t[k.replace('mera_', 'tree_')]):
            raise AssertionError('Pair isometries differ')
    residuals, scores, densities = {}, {}, {}
    for arm in ('mera', 'tree'):
        t = {k.removeprefix(arm+'_'): a for k, a in all_t.items() if k.startswith(arm+'_')}
        vn, vd = backend.contract_network(t), backend.dense_encoding(t)
        residuals[arm+'_encoding_fro'] = check(np.linalg.norm(vn-vd), 'independent encodings')
        residuals[arm+'_isometry_fro'] = check(np.linalg.norm(vn.conj().T@vn-np.eye(2)), 'full isometry')
        psi = vn.T.reshape(-1)/np.sqrt(2)
        residuals[arm+'_norm'] = check(abs(np.vdot(psi, psi)-1), 'state normalization')
        sq, _ = entropy(backend.reduced_tensor(psi, [0]))
        out = []
        original = {tuple(map(int,r['subset'].split())): float(r['r_network']) for r in data['groups'][arm, seed]}
        for s in SUBSETS:
            keep = [0]+[i+1 for i in s]
            qb = backend.reduced_tensor(psi, keep)
            qbd = backend.reduced_dense(psi, keep)
            b = backend.reduced_tensor(psi, [i+1 for i in s])
            sb, _ = entropy(b); sqb, eig = entropy(qb)
            score = (sq+sb-sqb)/2
            residuals[arm+str(s)+'_rho_fro'] = check(np.linalg.norm(qb-qbd), 'reduced states')
            residuals[arm+str(s)+'_score'] = check(abs(score-original[s]), 'saved subset score')
            out.append(score)
            if arm == 'mera' and s == (0,1,2):
                densities = {'rho_QB': qb, 'eigenvalues': eig, 'S_Q': sq, 'S_B': sb, 'S_QB': sqb, 'score': score, 'psi': psi, 'V': vn}
        scores[arm] = np.array(out)
    return {'scores': scores, 'worked': densities,
            'report': {'passed': True, 'seed': seed, 'tensor_hashes_verified': True, 'paired_isometries_identical': True,
                       'subsets_per_arm': 56, 'maximum_residual': max(residuals.values()), 'checks': residuals}}

def sufficiency(h):
    membership, wc, wu = collection()
    A = np.vstack([np.ones(56), membership])
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    cutoff = np.finfo(float).eps*max(A.shape)*s[0]
    rank = int(np.sum(s>cutoff))
    projection = vh[:rank].T@(vh[:rank]@h)
    difference = wc-wu
    null_error = check(np.linalg.norm(A@difference), 'constraint null space')
    rank_augmented = int(np.linalg.matrix_rank(np.vstack([A, h])))
    if rank != 8 or rank_augmented != 9:
        raise AssertionError('The fixed-cardinality constraint rank or information witness differs.')
    return A, wc, wu, projection, {'rank_A': rank, 'rank_with_information_row': rank_augmented,
        'singular_values': s.tolist(), 'svd_rank_cutoff': float(cutoff), 'constraint_null_residual_l2': null_error,
        'row_projection_residual_l2': float(np.linalg.norm(h-projection)), 'information_contrast': float(h@difference),
        'uniform_minimum_weight': float(wu.min())}

def save_figure(fig, root, name):
    for ext in ('png', 'svg', 'pdf'):
        metadata = {'Creator': 'Evlondo Cooper III', 'CreationDate': None, 'ModDate': None} if ext == 'pdf' else ({'Date': None} if ext == 'svg' else {'Software': None})
        fig.savefig(root/'figures'/f'{name}.{ext}', dpi=170, bbox_inches='tight', metadata=metadata)
    return fig

def masks_figure():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    for ax, mask, title in zip(axes, [(7,0,1), (0,3,6)], ['Contiguous collection: {7, 0, 1}', 'Noncontiguous collection: {0, 3, 6}']):
        theta = np.pi/2-np.arange(8)*np.pi/4
        x, y = np.cos(theta), np.sin(theta)
        ax.plot(np.r_[x,x[0]],np.r_[y,y[0]],color='#a9b5bb',lw=1.5)
        for i in range(8):
            ax.scatter(x[i],y[i],s=450,facecolor=COLORS['mera'] if i in mask else '#edf1f3',edgecolor='#50616b',zorder=3)
            ax.text(x[i],y[i],str(i),ha='center',va='center',color='white' if i in mask else '#263942',weight='bold')
        ax.set_title(title,fontsize=11,pad=10);ax.set_aspect('equal');ax.set_xlim(-1.35,1.35);ax.set_ylim(-1.3,1.3);ax.axis('off')
    return fig

def network_figure(graph):
    # Every gate, ordered pair and edge label comes from the audited JSON.
    fig, ax = plt.subplots(figsize=(11, 8), layout='constrained')
    xmap = {}; pos = {'A': (3.5, 8.5)}
    stages = graph['forward_stages']
    heights = {'expand2': 7.8, 'expand4': 6.4, 'disentangle4': 5.1, 'expand8': 3.6, 'disentangle8': 2.1}
    ax.text(3.5,8.9,'Logical diary A, Bell-paired with the untouched reference Q',ha='center',fontsize=11)
    for stage in stages:
        nodes = [n for n in graph['nodes'] if n['stage']==stage]
        y = heights[stage]
        if stage.startswith('expand'):
            nout = len(nodes)*2
            xs = np.linspace(0,7,nout)
            for j, n in enumerate(nodes):
                xo = xs[2*j:2*j+2]; center = float(np.mean(xo)); pin = pos[n['inputs'][0]]
                ax.plot([pin[0],center],[pin[1],y+.25],color='#677984',lw=1)
                ax.text((pin[0]+center)/2,(pin[1]+y+.25)/2+.04,n['inputs'][0],fontsize=8,ha='center',backgroundcolor='white')
                patch = Polygon([(center,y+.25),(xo[0]-.18,y-.15),(xo[1]+.18,y-.15)],facecolor='#dcecf3',edgecolor=COLORS['mera'],zorder=3)
                ax.add_patch(patch);ax.text(center,y+.03,n['name'],ha='center',va='center',fontsize=9,zorder=4)
                for e, x in zip(n['outputs'],xo): pos[e]=(float(x),y-.15)
        else:
            for n in nodes:
                pair = n['ordered_sites']; inputs = [pos[e] for e in n['inputs']]
                wrap = pair[-1]==0
                for k, (e, p) in enumerate(zip(n['inputs'],inputs)):
                    ax.plot([p[0],p[0]],[p[1],y+.13],color='#677984',lw=1)
                    ax.text(p[0]+.08,(p[1]+y)/2,e,fontsize=8,backgroundcolor='white')
                    pos[n['outputs'][k]]=(p[0],y-.13)
                if wrap:
                    for k, p in enumerate(inputs):
                        ax.add_patch(Rectangle((p[0]-.28,y-.13),.56,.26,facecolor='#f5e3d4',edgecolor=COLORS['tree'],zorder=3))
                        ax.text(p[0]+(-.4 if k==0 else .4),y,n['name']+(' [1]' if k==0 else ' [2]'),fontsize=8,ha='right' if k==0 else 'left',va='center')
                    ax.text(3.5,y-.38,f"{n['name']}: ordered pair {tuple(pair)}; the two end blocks are one periodic gate",ha='center',fontsize=8,color='#86512e')
                else:
                    left,right=sorted(p[0] for p in inputs)
                    ax.add_patch(Rectangle((left-.18,y-.13),right-left+.36,.26,facecolor='#f5e3d4',edgecolor=COLORS['tree'],zorder=3))
                    ax.text((left+right)/2,y,n['name'],ha='center',va='center',fontsize=8,zorder=4)
    for i in range(8):
        e='B'+str(i); x,y=pos[e]
        ax.plot([x,x],[y,.8],color='#677984');ax.text(x,.55,e,ha='center',fontsize=10)
    ax.text(3.5,.05,'Triangles: isometries w (one input, two outputs). Rectangles: two-site unitaries u.\nOrdered gate legs [1], [2] follow the JSON; all other wires have named edges.',ha='center',fontsize=9)
    ax.set(xlim=(-.65,7.65),ylim=(-.3,9.2));ax.axis('off')
    return fig

def gaps_figure(data):
    fig, ax = plt.subplots(figsize=(9, 3.8), layout='constrained')
    for j,(arm,key) in enumerate(zip(ARMS,('Delta_MERA','Delta_tree','Delta_Haar'))):
        vals=data['metrics'][key]; st=data['statistics'][key]
        jitter=((np.arange(64)%9)-4)*.032
        ax.scatter(vals,j+jitter,color=COLORS[arm],s=22,alpha=.75)
        ax.plot([st['percentile95_low'],st['percentile95_high']],[j,j],lw=4,color='#25343b',zorder=4)
        ax.scatter(st['mean'],j,s=65,marker='D',facecolor='white',edgecolor='#25343b',zorder=5)
    ax.axvline(0,color='#84939c',ls='--',lw=1)
    ax.set(yticks=range(3),yticklabels=[LABELS[a] for a in ARMS],xlabel='Access gap Δ = contiguous mean − uniform mean',title='Which collection pattern makes more reference information available?')
    ax.invert_yaxis()
    return fig

def subset_figure(scores):
    fig, ax = plt.subplots(figsize=(11, 4), layout='constrained')
    flags=np.array([s in WINDOWS for s in SUBSETS]);x=np.arange(56)
    ax.scatter(x[~flags],scores[~flags],s=24,color='#a6b4bc',label='Other three-site subsets')
    ax.scatter(x[flags],scores[flags],s=55,color=COLORS['mera'],marker='s',label='Eight cyclic windows')
    ax.axhline(scores.mean(),color='#596f7b',ls='--',label='Uniform mean')
    ax.axhline(scores[flags].mean(),color=COLORS['mera'],ls=':',lw=2,label='Contiguous mean')
    ax.set(xticks=x[::2],xticklabels=[''.join(map(str,SUBSETS[i])) for i in x[::2]],ylabel='Available reference information e(S)',xlabel='Collected subset (lexicographic order)',title='One frozen encoding: MERA seed 101001')
    ax.tick_params(axis='x',labelrotation=60,labelsize=8)
    ax.legend(ncol=4,fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.01))
    ax.set_title('One frozen encoding: MERA seed 101001',pad=42)
    return fig

def paired_figure(data):
    m=data['metrics']; g=m['Gamma'];neg=g < -RESOLUTION
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    lo=min(m['Delta_tree'].min(),m['Delta_MERA'].min())-.005;hi=max(m['Delta_tree'].max(),m['Delta_MERA'].max())+.005
    ax.plot([lo,hi],[lo,hi],ls='--',color='#84939c',label='Equal gaps')
    ax.scatter(m['Delta_tree'][~neg],m['Delta_MERA'][~neg],s=30,color=COLORS['mera'],label='Amplification')
    ax.scatter(m['Delta_tree'][neg],m['Delta_MERA'][neg],s=55,marker='v',color=COLORS['tree'],label='Attenuation')
    ax.set(xlabel='Paired-tree gap',ylabel='MERA gap',title='The intervention, pair by pair');ax.legend(fontsize=8)
    order=np.argsort(g);bx.axhline(0,color='#84939c',ls='--')
    bx.scatter(np.arange(64),g[order],color=[COLORS['tree'] if v < -RESOLUTION else COLORS['mera'] for v in g[order]],s=24)
    bx.set(xlabel='Encoding pair, ordered by Γ',ylabel='Γ = MERA gap − tree gap',title='All 64 effects, including five attenuations')
    return fig

def levels_figure(data):
    m=data['metrics'];fig,axes=plt.subplots(1,3,figsize=(11,3.8),sharey=True,layout='constrained')
    for ax,arm in zip(axes,ARMS):
        for i,(u,c) in enumerate(zip(m[arm+'_U'],m[arm+'_C'])):
            ax.plot([0,1],[u,c],color=COLORS[arm],alpha=.15,lw=.8)
        ax.scatter(np.zeros(64),m[arm+'_U'],color=COLORS[arm],marker='o',s=16,alpha=.65)
        ax.scatter(np.ones(64),m[arm+'_C'],color=COLORS[arm],marker='s',s=16,alpha=.65)
        ax.plot([0,1],[m[arm+'_U'].mean(),m[arm+'_C'].mean()],color='#263942',lw=2,marker='D',markerfacecolor='white',ms=6)
        ax.set(xticks=[0,1],xticklabels=['Uniform ○','Contiguous □'],xlim=(-.3,1.3),title=LABELS[arm])
    axes[0].set_ylabel('Available reference information (normalized)')
    return fig

def complement_figure(data):
    fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
    for j,(arm,key) in enumerate(zip(ARMS,('Delta_MERA','Delta_tree','Delta_Haar'))):
        v=data['metrics'][key]; xx=j+((np.arange(64)%9)-4)*.018
        ax.scatter(xx-.16,v,s=18,color=COLORS[arm],marker='s',alpha=.65)
        ax.scatter(xx+.16,-v,s=18,color=COLORS[arm],marker='o',alpha=.65)
        ax.scatter([j-.16,j+.16],[v.mean(),-v.mean()],marker='D',s=65,facecolor='white',edgecolor='#263942',zorder=5)
    ax.scatter([],[],marker='s',color='#596f7b',label='Three-site gap');ax.scatter([],[],marker='o',color='#596f7b',label='Complementary five-site gap')
    ax.axhline(0,color='#84939c',ls='--');ax.set(xticks=range(3),xticklabels=[LABELS[a] for a in ARMS],ylabel='Contiguous-minus-uniform gap',title='Complementary access reverses the same comparison');ax.legend(fontsize=8)
    return fig

def rowspace_figure(h, projection):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    ax.plot(h, color=COLORS['mera'],label='Actual e(S)',lw=1.3)
    ax.plot(projection,color=COLORS['tree'],ls='--',label='Row-space projection',lw=1.3)
    ax.set(xlabel='Three-site subset index',ylabel='Available information',title='Can individual inclusion statistics determine this score?');ax.legend(fontsize=8)
    membership,wc,wu=collection();d=wc-wu
    bx.bar(np.arange(56),d,color=[COLORS['mera'] if x>0 else '#a6b4bc' for x in d],width=.85)
    bx.set(xlabel='Three-site subset index',ylabel='Probability difference wC − wU',title='A change invisible to every individual efficiency')
    return fig
