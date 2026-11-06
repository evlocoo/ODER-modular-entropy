"""Ensure reproduction does not mutate the distributed reading notebooks by default."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import nbformat

spec=importlib.util.spec_from_file_location('companion_runner',Path(__file__).resolve().parents[1]/'run_notebooks.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NotebookRunner(unittest.TestCase):
    def run_fixture(self,update_originals=False,fail=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/'Example.ipynb'
            nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('print(1)')]),path)
            original=path.read_bytes()
            class Client:
                def __init__(self,notebook,**kwargs):self.notebook=notebook
                def execute(self):
                    if fail:raise RuntimeError('Kernel failure fixture')
                    cell=self.notebook.cells[0];cell.execution_count=1
                    cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text='1\n')]
            with patch.object(runner,'ROOT',root),patch.object(runner,'NotebookClient',Client):
                if fail:
                    with self.assertRaises(RuntimeError):runner.execute('run',['Example.ipynb'],update_originals)
                    self.assertEqual(path.read_bytes(),original)
                    self.assertFalse((root/'outputs/runs/run/notebooks/Example.ipynb').exists())
                    return
                report=runner.execute('run',['Example.ipynb'],update_originals)
            copy=root/report['notebooks'][0]['executed_notebook']
            self.assertEqual(nbformat.read(copy,as_version=4).cells[0].execution_count,1)
            self.assertTrue(report['passed'])
            if update_originals:self.assertEqual(path.read_bytes(),copy.read_bytes())
            else:self.assertEqual(path.read_bytes(),original)

    def test_default_preserves_distributed_notebook(self):self.run_fixture()
    def test_explicit_option_updates_distributed_notebook(self):self.run_fixture(True)
    def test_failed_run_preserves_distributed_notebook(self):self.run_fixture(True,True)


if __name__=='__main__':unittest.main()
