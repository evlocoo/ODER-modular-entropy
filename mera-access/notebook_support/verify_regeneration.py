"""Regenerate the study and compare numerical values at its 1e-10 tolerance.

From the companion folder:
    python notebook_support/verify_regeneration.py --output outputs/reproduction
    python notebook_support/verify_regeneration.py --compare-only outputs/reproduction
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse, csv, hashlib, json, math, re, subprocess, sys, time
from pathlib import Path
import numpy as np

TOL = 1e-10
ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT/'mera_ensemble_study'
# Verification records are produced after the study. They are not study outputs.
POST_RUN_RECORDS = {'fresh_process_verification.json', 'statistical_verification.json', 'companion_verification.json'}
STRING_COLUMNS = {'encoder','subset','estimand','quantity','comparison','contrast','sampling','state'}
INT_COLUMNS = {'seed','n_encodings_or_pairs','positive','unresolved','negative','negative_eigenvalue_count'}
BOOL_COLUMNS = {'contiguous','passed','within_observed_range','within_sample_iqr'}
FLOAT_COLUMNS = {'r_network','r_dense','complement_r_network','complement_r_dense',
    'complement_residual_network','complement_residual_dense','mean','sample_sd','median','q1','q3','iqr',
    'minimum','maximum','percentile95_low','percentile95_high','exploratory_value','empirical_percentile_midrank',
    'contiguous_mean','uniform_mean','gap','maximum_validation_residual','complementary_gap_residual',
    'Delta_MERA','Delta_tree','Gamma','L_C','L_U','mera_contiguous','mera_uniform','tree_contiguous','tree_uniform',
    'estimate','trace_residual','hermiticity_fro','minimum_raw_eigenvalue'}

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def exact_tree(x, y):
    if type(x) is not type(y): return False
    if isinstance(x, dict): return x.keys()==y.keys() and all(exact_tree(x[k],y[k]) for k in x)
    if isinstance(x, list): return len(x)==len(y) and all(exact_tree(a,b) for a,b in zip(x,y))
    if isinstance(x, float): return math.isfinite(x) and math.isfinite(y) and x.hex()==y.hex()
    return x==y

def safe_metadata(value):
    if isinstance(value,float) and not math.isfinite(value):return '<nonfinite: '+repr(value)+'>'
    if isinstance(value,dict):return {k:safe_metadata(v) for k,v in value.items()}
    if isinstance(value,list):return [safe_metadata(v) for v in value]
    return value

class Comparator:
    def __init__(self):
        self.issues=[]
        self.maximum=0.; self.location=None
        self.categories={}
        self.arrays_bitwise_equal=True
        self.arrays=0
        self.diagnostic_count_differences=[]

    def diagnostic_count(self,x,y,location):
        # This method is called only for the two explicitly named run-local
        # eigenvalue diagnostics, never for scientific counts or identifiers.
        if type(x) is not int or type(y) is not int or x<0 or y<0:
            self.fail(location,'Expected nonnegative integer diagnostic counts');return
        if x!=y:self.diagnostic_count_differences.append({'location':location,'reference':x,'generated':y,'difference':y-x})

    def fail(self, location, message):
        self.issues.append({'location':location,'message':message})

    def finite_tree(self,value,location):
        if isinstance(value,float) and not math.isfinite(value):self.fail(location,'Nonfinite floating-point value')
        elif isinstance(value,dict):
            for k,v in value.items():self.finite_tree(v,location+'.'+k)
        elif isinstance(value,list):
            for i,v in enumerate(value):self.finite_tree(v,f'{location}[{i}]')

    def residual(self, value, location, category):
        value=float(value)
        if not math.isfinite(value):
            self.fail(location,'Nonfinite discrepancy');return
        record=self.categories.setdefault(category,{'comparisons':0,'maximum_discrepancy':0.,'location':None})
        record['comparisons']+=1
        if record['location'] is None or value>record['maximum_discrepancy']:
            record.update(maximum_discrepancy=value,location=location)
        if self.location is None or value>self.maximum:
            self.maximum=value; self.location=location
        if value>TOL:self.fail(location,f'Numerical discrepancy {value:.17g} exceeds {TOL}')

    def tree(self, x, y, location, category='json'):
        if type(x) is not type(y):
            self.fail(location,f'Type changed: {type(x).__name__} to {type(y).__name__}');return
        if isinstance(x, dict):
            if x.keys()!=y.keys():self.fail(location,'JSON keys differ');return
            for k in x:self.tree(x[k],y[k],location+'.'+k,category)
        elif isinstance(x, list):
            if len(x)!=len(y):self.fail(location,'List lengths differ');return
            for i,(a,b) in enumerate(zip(x,y)):self.tree(a,b,f'{location}[{i}]',category)
        elif isinstance(x,float):
            if not math.isfinite(x) or not math.isfinite(y):self.fail(location,'Nonfinite floating-point value');return
            self.residual(abs(x-y),location,category)
        elif isinstance(x,(str,bool,int)) or x is None:
            if x!=y:self.fail(location,f'Discrete value differs: {x!r} versus {y!r}')
        else:self.fail(location,'Unsupported JSON value type')

    def array(self,x,y,location):
        self.arrays+=1
        if x.shape!=y.shape:self.fail(location,'Array shape differs');self.arrays_bitwise_equal=False;return
        if x.dtype!=y.dtype:self.fail(location,f'Array dtype differs: {x.dtype} versus {y.dtype}');self.arrays_bitwise_equal=False;return
        self.arrays_bitwise_equal &= x.tobytes(order='C')==y.tobytes(order='C')
        if x.dtype.kind in 'fc':
            if not np.isfinite(x).all() or not np.isfinite(y).all():self.fail(location,'Nonfinite array value');return
            # Euclidean norm of state vectors; Frobenius norm of matrices.
            # General stored arrays use the same flattened Euclidean convention.
            with np.errstate(over='ignore',invalid='ignore'):
                error=float(np.linalg.norm((x-y).ravel()))
            self.residual(error,location,'arrays')
        elif x.dtype.kind in 'biuUS':
            if not np.array_equal(x,y):self.fail(location,'Discrete array values differ')
        else:self.fail(location,'Unsupported array dtype')

    def csv(self,xpath,ypath,location):
        with Path(xpath).open(newline='',encoding='utf-8') as f:xx=list(csv.reader(f))
        with Path(ypath).open(newline='',encoding='utf-8') as f:yy=list(csv.reader(f))
        if not xx or not yy:self.fail(location,'Empty CSV file');return
        if xx[0]!=yy[0] or len(set(xx[0]))!=len(xx[0]):self.fail(location,'CSV schema or column ordering differs');return
        if len(xx)!=len(yy):self.fail(location,'CSV record count differs');return
        for i,(rx,ry) in enumerate(zip(xx[1:],yy[1:]),2):
            if len(rx)!=len(xx[0]) or len(ry)!=len(xx[0]):self.fail(f'{location}:row{i}','CSV row width differs');continue
            for name,x,y in zip(xx[0],rx,ry):
                at=f'{location}:row{i}.{name}'
                try:
                    if name in STRING_COLUMNS:self.tree(x,y,at,'csv')
                    elif name in INT_COLUMNS:
                        if not re.fullmatch(r'-?(0|[1-9][0-9]*)',x) or not re.fullmatch(r'-?(0|[1-9][0-9]*)',y):
                            raise ValueError('Expected integer field')
                        if name=='negative_eigenvalue_count' and Path(location).name=='density_validation.csv':
                            self.diagnostic_count(int(x),int(y),at)
                        else:self.tree(int(x),int(y),at,'csv')
                    elif name in BOOL_COLUMNS:
                        if x not in ('True','False') or y not in ('True','False'):raise ValueError('Expected boolean field')
                        self.tree(x=='True',y=='True',at,'csv')
                    elif name in FLOAT_COLUMNS:self.tree(float(x),float(y),at,'csv')
                    else:self.fail(at,'Unknown column type; extend the explicit schema before comparing')
                except ValueError as e:self.fail(at,str(e))

    def report(self):
        return {'passed':not self.issues,'tolerance':TOL,'maximum_discrepancy':self.maximum,
                'maximum_discrepancy_location':self.location,'categories':self.categories,
                'issues':self.issues,'all_arrays_bitwise_equal':bool(self.arrays_bitwise_equal),'arrays_compared':self.arrays,
                'diagnostic_count_differences':self.diagnostic_count_differences,
                'diagnostic_count_difference_records':len(self.diagnostic_count_differences),
                'norms':'Frobenius for matrices, Euclidean for states and flattened general arrays; absolute scalar differences. No relative tolerance.'}

def check_density_records(root,comp,side):
    """Validate each run's raw eigenvalues and their sign-count diagnostics."""
    start=len(comp.issues)
    folders={p.parent for pattern in ('density_validation.csv','raw_eigenvalues.npz') for p in root.rglob(pattern)}
    for p in root.rglob('validation.json'):
        try:
            value=load_json(p)
            if not isinstance(value,dict):comp.fail(side+':'+p.relative_to(root).as_posix(),'Validation record must be an object')
            elif 'tolerated_negative_eigenvalues' in value:folders.add(p.parent)
        except (ValueError,TypeError,OSError) as e:comp.fail(side+':'+p.relative_to(root).as_posix(),str(e))
    states_checked=0;total_negative=0;minimum=None;groups=0
    headers=['state','trace_residual','hermiticity_fro','minimum_raw_eigenvalue','negative_eigenvalue_count']
    for folder in sorted(folders):
        at=side+':'+folder.relative_to(root).as_posix()
        paths=[folder/n for n in ('density_validation.csv','raw_eigenvalues.npz','validation.json')]
        if not all(p.is_file() for p in paths):
            comp.fail(at,'Density validation requires CSV, raw eigenvalue archive, and validation.json');continue
        try:
            with paths[0].open(newline='',encoding='utf-8') as f:
                reader=csv.DictReader(f);rows=list(reader)
                if reader.fieldnames!=headers:comp.fail(at,'Density CSV schema differs');continue
                if not rows:comp.fail(at,'Density-state records are missing')
            by_state={}
            for row in rows:
                key=row['state']
                if not isinstance(key,str) or not key:comp.fail(at,'Invalid density-state key');continue
                if key in by_state:comp.fail(at+':'+key,'Duplicate density-state key')
                else:by_state[key]=row
            run_total=0
            with np.load(paths[1],allow_pickle=False) as arrays:
                if not arrays.files:comp.fail(at,'Raw eigenvalue states are missing')
                if len(arrays.files)!=len(set(arrays.files)):comp.fail(at,'Duplicate raw eigenvalue state key')
                if set(by_state)!=set(arrays.files):
                    comp.fail(at,'Density/raw state keys differ; missing from CSV: '+str(sorted(set(arrays.files)-set(by_state)))+'; unexpected in CSV: '+str(sorted(set(by_state)-set(arrays.files))))
                for key in arrays.files:
                    eig=arrays[key]
                    loc=at+':'+key
                    if eig.ndim!=1 or not eig.size or eig.dtype.kind!='f':
                        comp.fail(loc,'Eigenvalues must be a nonempty real floating-point vector');continue
                    if not np.isfinite(eig).all():comp.fail(loc,'Nonfinite eigenvalue');continue
                    low=float(eig.min());minimum=low if minimum is None else min(minimum,low)
                    if low < -TOL:comp.fail(loc,f'Minimum eigenvalue {low:.17g} is below {-TOL}')
                    count=int(np.count_nonzero(eig<0));run_total+=count
                    if key not in by_state:continue
                    row=by_state[key];raw_count=row['negative_eigenvalue_count']
                    if not isinstance(raw_count,str) or not re.fullmatch(r'0|[1-9][0-9]*',raw_count):
                        comp.fail(loc,'Expected nonnegative integer negative_eigenvalue_count')
                    elif int(raw_count)!=count:comp.fail(loc,'Recorded negative count does not equal sum(saved eigenvalues < 0)')
                    recorded=float(row['minimum_raw_eigenvalue'])
                    if not math.isfinite(recorded):comp.fail(loc,'Nonfinite recorded minimum eigenvalue')
                    else:comp.residual(abs(recorded-low),loc+'.minimum_raw_eigenvalue','run_local_eigenvalues')
                    states_checked+=1
            validation=load_json(paths[2])
            aggregate=validation.get('tolerated_negative_eigenvalues')
            if type(aggregate) is not int or aggregate!=run_total:
                comp.fail(at+'.tolerated_negative_eigenvalues','Aggregate must equal exactly the sum of this run\'s per-state negative counts')
            total_negative+=run_total;groups+=1
        except (ValueError,KeyError,TypeError,OSError) as e:comp.fail(at,'Cannot validate density records: '+str(e))
    return {'passed':len(comp.issues)==start,'density_groups_checked':groups,'density_states_checked':states_checked,
            'total_negative_eigenvalues':total_negative,'minimum_eigenvalue':minimum}

def inventory(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob('*')
                  if p.is_file() and p.relative_to(root).as_posix() not in POST_RUN_RECORDS)

def check_hash_records(root, comp, side):
    for p in sorted(root.rglob('tensor_hashes.json')):
        record=load_json(p);npz=p.parent/'tensors.npz';at=side+':'+p.relative_to(root).as_posix()
        if sha(npz)!=record['archive_sha256']:comp.fail(at,'Tensor archive does not match its own hash record')
        with np.load(npz,allow_pickle=False) as z:
            if set(z.files)!=set(record['arrays']):comp.fail(at,'Tensor hash keys differ');continue
            for k in z.files:
                if hashlib.sha256(z[k].astype('<c16').tobytes(order='C')).hexdigest()!=record['arrays'][k]:
                    comp.fail(at+':'+k,'Tensor array does not match its own hash record')
    for p in sorted(root.rglob('checkpoint.json')):
        record=load_json(p)
        expected={f.relative_to(p.parent).as_posix() for f in p.parent.rglob('*') if f.is_file() and f.name!='checkpoint.json'}
        if set(record['output_sha256'])!=expected:comp.fail(side+':'+p.relative_to(root).as_posix(),'Checkpoint hash inventory incomplete')
        for name,h in record['output_sha256'].items():
            path=(p.parent/name).resolve()
            if not path.is_relative_to(p.parent.resolve()):comp.fail(side+':checkpoint','Unsafe checkpoint path');continue
            if not path.is_file() or sha(path)!=h:comp.fail(side+':'+p.relative_to(root).as_posix()+':'+name,'Checkpoint hash mismatch')

def compare_results(reference, target):
    reference,target=Path(reference),Path(target)
    comp=Comparator()
    names=inventory(reference);other=inventory(target)
    if names!=other:
        comp.fail('file_inventory',f'Missing: {sorted(set(names)-set(other))}; unexpected: {sorted(set(other)-set(names))}')
        comp.arrays_bitwise_equal=False
    counts={'numeric_archives_compared':0,'csv_tables_compared':0,'validation_records_compared':0,'summary_records_compared':0}
    csv_exact=True;validations_exact=True;summary_exact=None
    environments={};excluded=[]
    for name in sorted(set(names)&set(other)):
        x,y=reference/name,target/name
        try:
            if x.suffix=='.npz':
                counts['numeric_archives_compared']+=1
                with np.load(x,allow_pickle=False) as xx,np.load(y,allow_pickle=False) as yy:
                    if xx.files!=yy.files:comp.fail(name,'NPZ array keys or their order differ');comp.arrays_bitwise_equal=False;continue
                    for key in xx.files:comp.array(xx[key],yy[key],name+':'+key)
            elif x.suffix=='.csv':
                counts['csv_tables_compared']+=1
                csv_exact &= x.read_bytes()==y.read_bytes()
                comp.csv(x,y,name)
            elif x.suffix=='.json':
                xx,yy=load_json(x),load_json(y)
                comp.finite_tree(xx,'reference:'+name);comp.finite_tree(yy,'generated:'+name)
                if x.name=='tensor_hashes.json':
                    for record in (xx,yy):
                        record['archive_sha256']='<individually verified hash>'
                        record['arrays']={k:'<individually verified hash>' for k in record['arrays']}
                if x.name=='validation.json':
                    counts['validation_records_compared']+=1
                    validations_exact &= exact_tree(xx,yy)
                    field='tolerated_negative_eigenvalues'
                    if (field in xx)!=(field in yy):comp.fail(name+'.'+field,'Diagnostic field missing on one side')
                    elif field in xx:
                        comp.diagnostic_count(xx.pop(field),yy.pop(field),name+'.'+field)
                if x.name=='summary.json':counts['summary_records_compared']+=1
                # Execution observations are recorded separately. They are not
                # reproducible scientific outputs. All other fields remain checked.
                skip=[]
                if name=='summary.json':skip=['resources']
                elif name=='plan_before_execution.json':
                    skip=['software']
                    # Report wording changes the file hash without changing the
                    # sampling plan. Check both source identities individually.
                    if 'source_sha256' in xx or 'source_sha256' in yy:
                        expected_reference=load_json(STUDY/'results/plan_before_execution.json')['source_sha256']
                        if xx.get('source_sha256')!=expected_reference:
                            comp.fail(name+'.source_sha256','Reference source hash does not match the supplied plan')
                        if yy.get('source_sha256')!=sha(STUDY/'study.py'):
                            comp.fail(name+'.source_sha256','Generated source hash does not match the distributed study.py')
                        skip.append('source_sha256')
                elif x.name=='checkpoint.json':skip=['wall_seconds','output_sha256']
                for key in skip:
                    if (key in xx)!=(key in yy):comp.fail(name+'.'+key,'Metadata key differs');continue
                    if key in xx:
                        if key=='output_sha256' and xx[key].keys()!=yy[key].keys():comp.fail(name+'.'+key,'Hash record keys differ')
                        environments[name+'.'+key]={'reference':safe_metadata(xx.pop(key)),'generated':safe_metadata(yy.pop(key))}
                        excluded.append(name+'.'+key)
                if name=='summary.json':summary_exact=exact_tree(xx,yy)
                comp.tree(xx,yy,name,'validation' if x.name=='validation.json' else 'summary' if x.name=='summary.json' else 'json')
        except (ValueError,KeyError,OSError,TypeError) as e:comp.fail(name,f'Cannot compare record: {e}')
    # Hash strings cannot agree across machines when numerical bytes differ.
    # Instead validate every archive/checkpoint against its own hash record.
    for side,folder in [('reference',reference),('generated',target)]:
        try:check_hash_records(folder,comp,side)
        except (ValueError,KeyError,OSError) as e:comp.fail(side+':hash_records',str(e))
    density={side:check_density_records(folder,comp,side) for side,folder in [('reference',reference),('generated',target)]}
    result=comp.report()
    result.update(counts,all_csv_tables_byte_equal=bool(csv_exact),validation_records_exact=bool(validations_exact),
                  run_local_eigenvalue_diagnostics=density,
                  diagnostic_count_policy='Only density_validation.csv negative_eigenvalue_count and validation.json tolerated_negative_eigenvalues are run-local. Each must match saved raw eigenvalues exactly; cross-run count differences are diagnostic. All scientific counts remain exact.',
                  numerical_summary_exact=summary_exact,exact_summary_equality=summary_exact,
                  maximum_array_residual=comp.categories.get('arrays',{}).get('maximum_discrepancy',0.),
                  file_inventory_exact=names==other,
                  execution_metadata=environments,execution_metadata_excluded_from_agreement=excluded,
                  hash_policy='Each side must match its own tensor and checkpoint hashes. Numerical bytes need not match across sides.',
                  generated_presentation_files='PNG figures and REPORT.md require matching inventory; their render bytes and runtime prose are not numerical data.',
                  post_run_records_not_in_study_inventory=sorted(POST_RUN_RECORDS))
    return result

def verify_source():
    identity=load_json(ROOT/'verification/source_identity.json')
    for name,h in identity['files'].items():
        if sha(ROOT/name)!=h:raise ValueError('Package file integrity mismatch: '+name)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output',type=Path,help='New directory: generate once, then compare')
    group.add_argument('--compare-only',type=Path,help='Compare an existing complete result directory without generating')
    args=parser.parse_args();target=(args.output or args.compare_only).resolve()
    reference=STUDY/'results'
    verify_source()
    if target==reference.resolve():raise SystemExit('Choose a separate result directory, not the preserved archive.')
    started=time.perf_counter();process_seconds=None
    if args.output:
        if target.exists():raise SystemExit('Choose a new --output directory. Existing runs are preserved.')
        run=subprocess.run([sys.executable,'-u',str(STUDY/'study.py'),'--output',str(target)],cwd=STUDY)
        process_seconds=time.perf_counter()-started
        if run.returncode:raise SystemExit(f'Study exited with code {run.returncode}. Keep its checkpoints and failure record; do not substitute seeds.')
    if not (target/'summary.json').is_file():raise SystemExit('Missing summary.json. Supply a complete result directory or use --output to regenerate.')
    result=compare_results(reference,target)
    result.update(inputs_unchanged=True,fresh_process_exit_code=0 if args.output else None,
                  fresh_study_process_wall_seconds_including_imports=process_seconds,
                  verification_wall_seconds_including_study=time.perf_counter()-started,
                  fresh_study_resources=safe_metadata(load_json(target/'summary.json').get('resources')),
                  execution_method='Distributed study.py in a fresh subprocess' if args.output else 'Comparison only; no study executed')
    verify_source()
    (target/'companion_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('execution_metadata','execution_metadata_excluded_from_agreement','issues')},indent=2),flush=True)
    if not result['passed']:
        first=result['issues'][:3]
        raise SystemExit('Comparison failed. Keep this result directory and inspect companion_verification.json. First issues: '+json.dumps(first))

if __name__=='__main__':main()
