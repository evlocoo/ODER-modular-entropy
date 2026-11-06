"""Execute extension notebooks in fresh Python processes when kernel sockets are unavailable.

This is a transparent fallback for these plain-Python notebooks, not a Jupyter
kernel emulator. Each child runs code cells in order in one new namespace,
captures explicit display() tables and plt.show() figures, and saves outputs.
Use run_notebooks.py on a system supporting Jupyter kernel transport.
"""
from pathlib import Path
import argparse, base64, contextlib, io, json, os, platform, subprocess, sys, time
ROOT=Path(__file__).resolve().parent
NAMES=['Gaussian_Noise_and_Recovery.ipynb','Joint_Record_Access.ipynb','Observer_Trajectories_and_Physical_Scales.ipynb']

def child(name, destination, update):
    import nbformat
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import IPython.display
    from importlib.metadata import version
    notebook=nbformat.read(ROOT/name,as_version=4)
    namespace={'__name__':'__main__'}
    count=0; started=time.perf_counter()
    for index,cell in enumerate(notebook.cells):
        if cell.cell_type!='code': continue
        count+=1; cell.outputs=[]; cell.execution_count=count
        captured=io.StringIO()
        def flush():
            text=captured.getvalue()
            if text: cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=text))
            captured.seek(0);captured.truncate(0)
        def display(*objects,**kwargs):
            flush()
            for obj in objects:
                data={'text/plain':repr(obj)}
                if hasattr(obj,'_repr_html_'):
                    html=obj._repr_html_()
                    if html:data['text/html']=html
                cell.outputs.append(nbformat.v4.new_output('display_data',data=data,metadata={}))
        def show(*args,**kwargs):
            flush()
            for number in plt.get_fignums():
                fig=plt.figure(number);buf=io.BytesIO()
                fig.savefig(buf,format='png',dpi=150,bbox_inches='tight')
                cell.outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(buf.getvalue()).decode(),'text/plain':repr(fig)},metadata={}))
            plt.close('all')
        IPython.display.display=display; plt.show=show
        try:
            with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):
                exec(compile(cell.source,f'{name}:cell{index+1}','exec'),namespace)
            flush()
        except Exception:
            flush();raise
    nbformat.validate(notebook)
    nbformat.write(notebook,destination/name)
    if update:nbformat.write(notebook,ROOT/name)
    return dict(notebook=name,code_cells=count,passed=True,fresh_process=True,fresh_kernel=False,
                seconds=time.perf_counter()-started,
                environment={k:version(k) for k in ['numpy','scipy','matplotlib','pandas','nbformat','ipykernel']})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',required=True)
    p.add_argument('--notebooks',nargs='+',choices=NAMES,default=NAMES)
    p.add_argument('--update-originals',action='store_true')
    p.add_argument('--child',choices=NAMES)
    args=p.parse_args()
    if Path(args.run_name).name!=args.run_name:raise ValueError('Use one directory name')
    os.chdir(ROOT);os.environ['ODER_RUN_NAME']=args.run_name
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
    output=ROOT/'outputs/runs'/args.run_name
    if args.child:
        item=child(args.child,output/'notebooks',args.update_originals)
        (output/(args.child+'.execution.json')).write_text(json.dumps(item,indent=2)+'\n')
        print(json.dumps(item),flush=True);return
    if output.exists():raise FileExistsError('Choose a new run name')
    (output/'notebooks').mkdir(parents=True)
    report=dict(run_name=args.run_name,python=platform.python_version(),system=platform.system(),
                method='One fresh Python process per notebook; ordered plain-Python cell execution; explicit display/plot capture',
                reason='Explicit fresh-process mode; this command does not attempt Jupyter kernel startup',
                fresh_kernel=False,BLAS_threads=1,notebooks=[],passed=False)
    for name in args.notebooks:
        print('Executing in fresh process: '+name,flush=True)
        command=[sys.executable,str(Path(__file__).resolve()),'--run-name',args.run_name,'--child',name]
        if args.update_originals:command.append('--update-originals')
        completed=subprocess.run(command,cwd=ROOT)
        if completed.returncode:
            report['failed_notebook']=name
            (output/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
            raise SystemExit(completed.returncode)
        report['notebooks'].append(json.loads((output/(name+'.execution.json')).read_text()))
    report['passed']=True
    (output/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
