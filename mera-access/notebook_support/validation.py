"""Executable causal-cone and exact-null checks for the frozen eight-site pair.

python -m notebook_support.validation --output outputs/validation_check
Matrix norms are Frobenius; scalar errors are absolute; tolerance is 1e-10.
No tensor generation, Hamiltonian optimization, or ensemble resampling occurs.
"""
from pathlib import Path
import argparse, csv, hashlib, itertools, json, math
import numpy as np
from notebook_support import access

TOL = 1e-10
# Ordered graph specified independently of the backend's PAIRS and gate loops.
# Each tuple is (tensor name, ordered inputs, ordered outputs).
GRAPH = [
    ('w0_0', ('A',), ('a0','a1')),
    ('w1_0', ('a0',), ('b0','b1')),
    ('w1_1', ('a1',), ('b2','b3')),
    ('u4_0', ('b1','b2'), ('c1','c2')),
    ('u4_1', ('b3','b0'), ('c3','c0')),
    ('w2_0', ('c0',), ('d0','d1')),
    ('w2_1', ('c1',), ('d2','d3')),
    ('w2_2', ('c2',), ('d4','d5')),
    ('w2_3', ('c3',), ('d6','d7')),
    ('u8_0', ('d1','d2'), ('B1','B2')),
    ('u8_1', ('d3','d4'), ('B3','B4')),
    ('u8_2', ('d5','d6'), ('B5','B6')),
    ('u8_3', ('d7','d0'), ('B7','B0')),
]
BOUNDARY = tuple('B'+str(i) for i in range(8))
PAULI = {'X': np.array([[0,1],[1,0]],complex),
         'Y': np.array([[0,-1j],[1j,0]],complex),
         'Z': np.diag([1.,-1.]).astype(complex)}


def check(value):
    value=float(value)
    if not np.isfinite(value) or value>TOL:
        raise AssertionError(f'Validation residual {value:.17g} exceeds {TOL}')
    return value


def embed(operator, support, labels):
    """Identity-complete an operator, preserving explicitly ordered qubit legs."""
    support=list(support); labels=list(labels)
    extras=[x for x in labels if x not in support]
    order=support+extras
    if len(set(order))!=len(order) or set(order)!=set(labels):
        raise ValueError('Invalid operator support.')
    n=len(order); axes=[order.index(x) for x in labels]
    big=np.kron(operator,np.eye(2**len(extras)))
    return big.reshape([2]*(2*n)).transpose(axes+[n+i for i in axes]).reshape(2**n,2**n)


def ascend(tensors, operator, support, tree=False):
    """Local Heisenberg ascent. This path never constructs or accepts V."""
    op=operator.copy(); support=list(support); steps=[]
    for name, inputs, outputs in reversed(GRAPH):
        before=support.copy()
        if not set(support).intersection(outputs):
            action='canceled by tensor isometry'; dimension=op.shape[0]
        elif tree and name.startswith('u'):
            if not np.array_equal(tensors[name],np.eye(4)):
                raise AssertionError('Tree gate must be exactly identity.')
            rename=dict(zip(outputs,inputs))
            support=[rename.get(x,x) for x in support]
            action='identity wires'; dimension=op.shape[0]
        else:
            rest=[x for x in support if x not in outputs]
            local=embed(op,support,list(outputs)+rest)
            transfer=np.kron(tensors[name],np.eye(2**len(rest)))
            op=transfer.conj().T@local@transfer
            support=list(inputs)+rest
            action='retained'; dimension=local.shape[0]
        steps.append(dict(tensor=name,action=action,support_before=before,
                          support_after=support.copy(),local_operator_dimension=dimension))
    assert support==['A'] and op.shape==(2,2)
    return op,steps


def graph_encoding(tensors):
    labels={x:i for i,x in enumerate(dict.fromkeys(
        x for _,ins,outs in GRAPH for x in (*ins,*outs)))}
    args=[]
    for name,ins,outs in GRAPH:
        args.extend([tensors[name].reshape([2]*(len(ins)+len(outs))),
                     [labels[x] for x in (*outs,*ins)]])
    return np.einsum(*args,[labels[x] for x in (*BOUNDARY,'A')],optimize='greedy').reshape(256,2)


def load_pair(root):
    directory=root/'mera_ensemble_study/inputs'
    pair={}; hashes={}
    for model in ['mera','tree']:
        identity=json.loads((directory/(model+'_hashes.json')).read_text(encoding='utf-8'))
        path=directory/(model+'_tensors.npz')
        expected=identity['archive_sha256'] if model=='mera' else identity['tree_archive_sha256']
        hashes[model]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashes[model]==expected
        expected_arrays=identity['arrays'] if model=='mera' else identity['tree_array_hashes']
        with np.load(path,allow_pickle=False) as archive:
            assert set(archive.files)=={n for n,_,_ in GRAPH}==set(expected_arrays)
            pair[model]={k:archive[k] for k in archive.files}
        for k,t in pair[model].items():
            assert t.shape==((4,2) if k.startswith('w') else (4,4))
            assert t.dtype==np.dtype('complex128') and np.all(np.isfinite(t))
            assert hashlib.sha256(t.astype('<c16').tobytes(order='C')).hexdigest()==expected_arrays[k]
    for name,_,_ in GRAPH:
        if name.startswith('w'):assert np.array_equal(pair['mera'][name],pair['tree'][name])
        else:assert np.array_equal(pair['tree'][name],np.eye(4))
    return pair,hashes


def mixed_reduce(rho, keep, n=9):
    rest=[i for i in range(n) if i not in keep]
    order=list(keep)+rest
    arranged=rho.reshape([2]*(2*n)).transpose(order+[n+i for i in order])
    d=2**len(keep); env=2**len(rest)
    return np.trace(arranged.reshape(d,env,d,env),axis1=1,axis2=3)


def execute(root, output):
    root=Path(root); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    pair,hashes=load_pair(root)
    backend=access.load_backend(root/'mera_ensemble_study')
    specification=access.read_json(root/'mera_ensemble_study/inputs/audited_architecture.json')
    assert [(x['name'],tuple(x['inputs']),tuple(x['outputs'])) for x in specification['nodes']]==GRAPH
    arrays={}; eigenvalues={}; density_checks=[]; invariants={}; cones=[]; nulls={}

    def entropy(rho,key):
        trace=check(abs(np.trace(rho)-1)); hermitian=check(np.linalg.norm(rho-rho.conj().T))
        eig=np.linalg.eigvalsh((rho+rho.conj().T)/2)
        assert np.all(np.isfinite(eig)); check(max(0.,-eig.min()))
        eigenvalues[key]=eig
        density_checks.append(dict(state=key,trace_residual=trace,hermiticity_fro=hermitian,
                                   minimum_eigenvalue=float(eig.min())))
        positive=eig[eig>0]
        return float(-np.sum(positive*np.log2(positive)))

    triples=list(itertools.combinations(range(8),3))
    fours=list(itertools.combinations(range(8),4))
    windows={tuple(sorted((i,(i+1)%8,(i+2)%8))) for i in range(8)}
    four_windows={tuple(sorted((i,(i+1)%8,(i+2)%8,(i+3)%8))) for i in range(8)}
    permutations=np.array(list(itertools.permutations(range(8))),dtype=np.int8)
    triple_index={s:i for i,s in enumerate(triples)}
    mapped=np.array([[triple_index[tuple(sorted(int(row[j]) for j in w))] for w in sorted(windows)]
                     for row in permutations],dtype=np.int16)
    frequencies=np.bincount(mapped.ravel(),minlength=56)
    assert np.array_equal(frequencies,np.full(56,5760))
    arrays.update(boundary_permutations=permutations,permuted_window_indices=mapped,
                  triple_frequencies=frequencies,triples=np.array(triples),four_subsets=np.array(fours))

    for model,tensors in pair.items():
        inv={k:check(np.linalg.norm(t.conj().T@t-np.eye(t.shape[1]))) for k,t in tensors.items()}
        v=graph_encoding(tensors)
        inv['encoding_isometry']=check(np.linalg.norm(v.conj().T@v-np.eye(2)))
        encodings={'graph':v,'network':backend.contract_network(tensors),'dense':backend.dense_encoding(tensors)}
        for path,encoding in encodings.items():
            inv[path+'_encoding_agreement']=check(np.linalg.norm(v-encoding))
        observables=[(f'{p}{i}',(i,),(p,)) for i in range(8) for p in 'XYZ']
        observables += [(f'Z{i}Z{(i+1)%8}',(i,(i+1)%8),('Z','Z')) for i in range(8)]
        observables += [('X7Y0',(7,0),('X','Y'))]
        for name,sites,paulis in observables:
            op=np.ones((1,1),complex)
            for symbol in paulis:op=np.kron(op,PAULI[symbol])
            labels=['B'+str(i) for i in sites]
            logical,steps=ascend(tensors,op,labels,tree=model=='tree')
            full=embed(op,labels,BOUNDARY)
            comparisons={}
            for path,encoding in encodings.items():
                dense=encoding.conj().T@full@encoding
                comparisons[path]=check(np.linalg.norm(logical-dense))
                arrays[model+'_'+name+'_'+path]=dense
            arrays[model+'_'+name+'_ascent']=logical
            cones.append(dict(encoder=model,observable=name,ordered_sites=list(sites),
                              frobenius_disagreements=comparisons,steps=steps))
        psi=v.T.reshape(-1)/np.sqrt(2)
        inv['state_norm']=check(abs(np.vdot(psi,psi)-1))
        def score(subset):
            key=model+'_pure_'+''.join(map(str,subset))
            qb=backend.reduced_tensor(psi,[0]+[i+1 for i in subset])
            b=backend.reduced_tensor(psi,[i+1 for i in subset])
            return (1+entropy(b,key+'_B')-entropy(qb,key+'_QB'))/2
        inv['reference_state']=check(np.linalg.norm(backend.reduced_tensor(psi,[0])-np.eye(2)/2))
        inv['empty_access']=check(abs(score(())))
        inv['full_access']=check(abs(score(tuple(range(8)))-1))
        r3=np.array([score(s) for s in triples]); r4=np.array([score(s) for s in fours])
        r5=np.array([score(tuple(i for i in range(8) if i not in s)) for s in triples])
        inv['three_five_complement']=check(np.max(abs(r3+r5-1)))
        arrays[model+'_three_scores']=r3; arrays[model+'_four_scores']=r4
        # Explicit depolarized logical input: boundary marginal unchanged, Q uncorrelated.
        boundary=v@v.conj().T/2
        erased=np.zeros((512,512),complex)
        for pauli in [np.eye(2),*PAULI.values()]:
            depolarized_branch=(v@pauli).T.reshape(-1)/np.sqrt(2)
            erased+=np.outer(depolarized_branch,depolarized_branch.conj())/4
        inv['depolarized_product_state']=check(np.linalg.norm(erased-np.kron(np.eye(2)/2,boundary)))
        inv['depolarized_boundary']=check(np.linalg.norm(mixed_reduce(erased,list(range(1,9)))-boundary))
        erased_scores=[]
        for mask in range(256):
            subset=[i for i in range(8) if mask & (1<<i)]
            b=mixed_reduce(erased,[i+1 for i in subset])
            qb=mixed_reduce(erased,[0]+[i+1 for i in subset])
            product_residual=check(np.linalg.norm(qb-np.kron(np.eye(2)/2,b)))
            key=f'{model}_erased_{mask}'
            erased_scores.append((1+entropy(b,key+'_B')-entropy(qb,key+'_QB'))/2)
        arrays[model+'_depolarized_scores']=np.array(erased_scores)
        comp=[fours.index(tuple(i for i in range(8) if i not in s)) for s in fours]
        pair_error=check(np.max(abs(r4+r4[comp]-1)))
        c4=float(np.mean([r4[i] for i,s in enumerate(fours) if s in four_windows]))
        u4=float(r4.mean()); check(abs(c4-.5)); check(abs(u4-.5))
        perm_means=r3[mapped].mean(axis=1)
        arrays[model+'_permutation_means']=perm_means
        perm_error=check(abs(math.fsum(perm_means)/len(permutations)-math.fsum(r3)/56))
        original_gap=float(np.mean([r3[i] for i,s in enumerate(triples) if s in windows])-r3.mean())
        nulls[model]=dict(depolarized_subset_count=256,
                         maximum_depolarized_information=check(np.max(np.abs(erased_scores))),
                         complement_pair_residual=pair_error,four_contiguous_mean=c4,
                         four_uniform_mean=u4,four_gap=check(abs(c4-u4)),
                         permutation_count=len(permutations),triple_frequency=5760,
                         permutation_gap_residual=perm_error,three_site_gap=original_gap)
        invariants[model]=inv
    # Original paired scores are numerical regression targets, not null inputs.
    rows=access.read_csv(root/'mera_ensemble_study/inputs/baseline_paired.csv')
    for model in pair:
        actual=np.array([arrays[model+'_three_scores'][triples.index(tuple(map(int,r['subset'].split())))] for r in rows])
        for path in ['network','dense']:
            expected=np.array([float(r['r_'+model+'_'+path]) for r in rows])
            invariants[model]['three_site_'+path+'_regression']=check(np.max(abs(actual-expected)))
    assert len(cones)==66
    summary=dict(passed=True,tolerance=TOL,input_sha256=hashes,causal_cone_comparisons=66,
                 maximum_operator_frobenius_residual=max(max(x['frobenius_disagreements'].values()) for x in cones),
                 invariants=invariants,exact_nulls=nulls,
                 maximum_invariant_residual=max(max(x.values()) for x in invariants.values()),
                 scope='Frozen exploratory MERA/tree, seed 20260918; unoptimized finite encoding.')
    np.savez_compressed(output/'operators_and_nulls.npz',**arrays)
    np.savez_compressed(output/'raw_eigenvalues.npz',**eigenvalues)
    for name,value in [('summary',summary),('causal_cones',cones),('density_validation',density_checks)]:
        (output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output directory.')
    print(json.dumps(execute(Path(__file__).resolve().parents[1],args.output),indent=2))
