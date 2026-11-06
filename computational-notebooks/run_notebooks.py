"""Execute the current computational companion, one fresh kernel per notebook."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,platform,time

for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[variable]='1'
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parent
NOTEBOOKS=['Upstream_Regeneration.ipynb','Observer_Access_Reading_Guide.ipynb',
           'Schwarzschild_Propagation_Controls.ipynb','Modular_Realization.ipynb',
           'Finite_Receiver.ipynb','Receiver_and_Storage.ipynb','Causal_Phase_and_Delay.ipynb',
           'Gaussian_Noise_and_Recovery.ipynb','Observer_Trajectories_and_Physical_Scales.ipynb',
           'Joint_Record_Access.ipynb']


def execute(run_name,names,update_originals=False):
    if Path(run_name).name!=run_name or run_name in ('','.','..'):
        raise ValueError('Run name must be one directory name.')
    output=ROOT/'outputs/runs'/run_name
    if output.exists():
        raise FileExistsError('Choose a new run name; prior runs are preserved.')
    output.mkdir(parents=True)
    executed_directory=output/'notebooks'
    executed_directory.mkdir()
    os.environ['ODER_RUN_NAME']=run_name
    report=dict(python=platform.python_version(),system=platform.system(),
                run_name=run_name,BLAS_threads=1,update_originals=update_originals,
                notebooks=[],passed=False)
    for name in names:
        print('Executing '+name,flush=True)
        path=ROOT/name
        n=nbformat.read(path,as_version=4)
        for cell in n.cells:
            if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
        started=time.perf_counter()
        try:
            NotebookClient(n,timeout=1800,kernel_name='python3',
                           resources={'metadata':{'path':str(ROOT)}}).execute()
        except Exception:
            report['failed_notebook']=name
            (output/'execution.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            raise
        for cell in n.cells:
            cell.metadata.pop('execution',None)
        executed_path=executed_directory/name
        nbformat.write(n,executed_path)
        if update_originals:
            nbformat.write(n,path)
        item=dict(notebook=name,fresh_kernel=True,
                  code_cells=sum(c.cell_type=='code' for c in n.cells),
                  seconds=time.perf_counter()-started,passed=True,
                  executed_notebook=executed_path.relative_to(ROOT).as_posix())
        report['notebooks'].append(item)
        (output/'execution.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(item),flush=True)
    report['passed']=True
    (output/'execution.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--notebooks',nargs='+',choices=NOTEBOOKS,default=NOTEBOOKS)
    p.add_argument('--update-originals',action='store_true',
                   help='Also replace distributed notebooks with successfully executed copies.')
    args=p.parse_args()
    print(json.dumps(execute(args.run_name,args.notebooks,args.update_originals),indent=2))
