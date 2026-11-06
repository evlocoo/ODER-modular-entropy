"""Focused tests for companion comparison, using validation fixtures, not new physical encodings."""
import csv, json, tempfile, unittest
from pathlib import Path
import numpy as np
from notebook_support.verify_regeneration import Comparator, compare_results

class ComparisonTests(unittest.TestCase):
    def density_fixture(self,root,small=1e-15):
        root.mkdir()
        arrays={'Q':np.array([small,1.-small]),'B':np.array([.5,.5])}
        np.savez(root/'raw_eigenvalues.npz',**arrays)
        with (root/'density_validation.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['state','trace_residual','hermiticity_fro','minimum_raw_eigenvalue','negative_eigenvalue_count'])
            for k,v in arrays.items():w.writerow([k,0.,0.,float(v.min()),int(np.count_nonzero(v<0))])
        (root/'validation.json').write_text(json.dumps({'passed':True,'tolerance':1e-10,
            'tolerated_negative_eigenvalues':int(sum(np.count_nonzero(v<0) for v in arrays.values()))}))
        (root/'summary.json').write_text(json.dumps({'seed':101001,'count':64}))

    def test_report_source_hashes_are_checked_without_relaxing_plan(self):
        from notebook_support.verify_regeneration import STUDY, load_json, sha
        with tempfile.TemporaryDirectory() as folder:
            ref,new=Path(folder)/'ref',Path(folder)/'new'
            self.density_fixture(ref);self.density_fixture(new)
            reference_hash=load_json(STUDY/'results/plan_before_execution.json')['source_sha256']
            current_hash=sha(STUDY/'study.py')
            def plans(left=reference_hash,right=current_hash,seed=101001):
                (ref/'plan_before_execution.json').write_text(json.dumps({'source_sha256':left,'seed':101001}))
                (new/'plan_before_execution.json').write_text(json.dumps({'source_sha256':right,'seed':seed}))
            plans()
            result=compare_results(ref,new)
            self.assertTrue(result['passed'],result['issues'])
            for arguments in ({'left':'0'*64},{'right':'0'*64},{'seed':101002}):
                plans(**arguments)
                self.assertFalse(compare_results(ref,new)['passed'])

    def change_density_csv(self,root,change):
        p=root/'density_validation.csv'
        with p.open(newline='',encoding='utf-8') as f:rows=list(csv.reader(f))
        change(rows)
        with p.open('w',newline='',encoding='utf-8') as f:csv.writer(f).writerows(rows)

    def test_consistent_near_zero_sign_change_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as folder:
            ref,new=Path(folder)/'ref',Path(folder)/'new'
            self.density_fixture(ref);self.density_fixture(new,-1e-15)
            r=compare_results(ref,new)
            self.assertTrue(r['passed'],r['issues'])
            self.assertEqual(r['diagnostic_count_difference_records'],2)
            self.assertFalse(r['validation_records_exact'])
            for side in ('reference','generated'):
                self.assertTrue(r['run_local_eigenvalue_diagnostics'][side]['passed'])
                self.assertEqual(r['run_local_eigenvalue_diagnostics'][side]['density_states_checked'],2)

    def test_incorrect_per_state_negative_count_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            ref,new=Path(folder)/'ref',Path(folder)/'new'
            self.density_fixture(ref);self.density_fixture(new)
            self.change_density_csv(new,lambda rows:rows[1].__setitem__(4,'1'))
            r=compare_results(ref,new)
            self.assertFalse(r['passed']);self.assertTrue(any('Recorded negative count' in x['message'] for x in r['issues']))

    def test_incorrect_aggregate_negative_count_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            ref,new=Path(folder)/'ref',Path(folder)/'new'
            self.density_fixture(ref);self.density_fixture(new,-1e-15)
            p=new/'validation.json';v=json.loads(p.read_text());v['tolerated_negative_eigenvalues']=0;p.write_text(json.dumps(v))
            r=compare_results(ref,new)
            self.assertFalse(r['passed']);self.assertTrue(any('Aggregate must equal' in x['message'] for x in r['issues']))

    def test_eigenvalue_below_bound_fails_even_when_arrays_agree(self):
        with tempfile.TemporaryDirectory() as folder:
            ref,new=Path(folder)/'ref',Path(folder)/'new'
            self.density_fixture(ref,-1.1e-10);self.density_fixture(new,-1.1e-10)
            r=compare_results(ref,new)
            self.assertFalse(r['passed']);self.assertEqual(r['maximum_array_residual'],0.)
            self.assertTrue(any('is below' in x['message'] for x in r['issues']))

    def test_state_keys_missing_duplicate_or_unexpected_fail(self):
        for change in ('missing','duplicate','unexpected','missing_array'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                ref,new=Path(folder)/'ref',Path(folder)/'new'
                self.density_fixture(ref);self.density_fixture(new)
                # Make the same invalid change on both sides so cross-platform
                # equality alone cannot catch the broken run-local state map.
                for run in (ref,new):
                    if change=='missing':self.change_density_csv(run,lambda rows:rows.pop())
                    elif change=='duplicate':self.change_density_csv(run,lambda rows:rows.append(rows[1].copy()))
                    elif change=='unexpected':self.change_density_csv(run,lambda rows:rows[1].__setitem__(0,'other'))
                    else:np.savez(run/'raw_eigenvalues.npz',Q=np.array([1e-15,1.-1e-15]))
                r=compare_results(ref,new);self.assertFalse(r['passed'])
                self.assertFalse(r['run_local_eigenvalue_diagnostics']['generated']['passed'])

    def test_run_local_minimum_and_finite_checks(self):
        for change in ('minimum','nonfinite'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                ref,new=Path(folder)/'ref',Path(folder)/'new'
                self.density_fixture(ref);self.density_fixture(new)
                if change=='minimum':
                    for run in (ref,new):self.change_density_csv(run,lambda rows:rows[1].__setitem__(3,'0.001'))
                else:np.savez(new/'raw_eigenvalues.npz',Q=np.array([float('nan'),1.]),B=np.array([.5,.5]))
                self.assertFalse(compare_results(ref,new)['passed'])

    def test_scientific_counts_and_seeds_remain_exact_with_density_records(self):
        for field in ('seed','count','negative_eigenvalue_count','tolerated_negative_eigenvalues'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as folder:
                ref,new=Path(folder)/'ref',Path(folder)/'new'
                self.density_fixture(ref);self.density_fixture(new,-1e-15)
                for run in (ref,new):
                    p=run/'summary.json';v=json.loads(p.read_text());v.setdefault(field,1);p.write_text(json.dumps(v))
                p=new/'summary.json';v=json.loads(p.read_text());v[field]+=1;p.write_text(json.dumps(v))
                r=compare_results(ref,new);self.assertFalse(r['passed'])
                self.assertTrue(any(x['location']=='summary.json.'+field for x in r['issues']))

    def test_small_float_differences_and_identity_are_separate(self):
        c=Comparator();c.tree({'seed':101001,'count':64,'mean':.3},{'seed':101001,'count':64,'mean':.3+4e-15},'summary')
        c.array(np.array([.3]),np.array([.3+4e-15]),'state')
        self.assertTrue(c.report()['passed']);self.assertFalse(c.report()['all_arrays_bitwise_equal'])
        self.assertGreater(c.report()['maximum_discrepancy'],0)
        self.assertIsNotNone(c.report()['maximum_discrepancy_location'])

    def test_beyond_tolerance_and_norm_accumulation(self):
        c=Comparator();c.tree(.3,.3+2e-10,'summary.mean');self.assertFalse(c.report()['passed'])
        c=Comparator();c.array(np.zeros((2,2)),np.full((2,2),6e-11),'matrix')
        self.assertFalse(c.report()['passed']) # Frobenius 1.2e-10, although each element is smaller.

    def test_discrete_values_types_and_ordering(self):
        for x,y in [(101001,101002),(64,63),('mera','tree'),(True,False),(True,1),(64,64.),([1,2],[2,1]),([1],[1,2]),({'a':1},{'b':1})]:
            with self.subTest(x=x,y=y):
                c=Comparator();c.tree(x,y,'record');self.assertFalse(c.report()['passed'])

    def test_shapes_dtypes_keys_and_integer_arrays(self):
        for x,y in [(np.zeros(2),np.zeros((1,2))),(np.array([1],dtype='int64'),np.array([1.],dtype='float64')),(np.array([101001]),np.array([101002]))]:
            c=Comparator();c.array(x,y,'array');self.assertFalse(c.report()['passed'])

    def test_nonfinite_values_are_rejected_on_either_side(self):
        for bad in [float('nan'),float('inf'),float('-inf')]:
            for x,y in [(bad,0.),(0.,bad)]:
                c=Comparator();c.tree(x,y,'float');self.assertFalse(c.report()['passed'])
                c=Comparator();c.array(np.array([x]),np.array([y]),'array');self.assertFalse(c.report()['passed'])

    def test_csv_typed_schema_and_missing_records(self):
        with tempfile.TemporaryDirectory() as folder:
            x,y=Path(folder)/'x.csv',Path(folder)/'y.csv'
            def write(p,rows):
                with p.open('w',newline='',encoding='utf-8') as f:csv.writer(f).writerows(rows)
            header=['seed','positive','encoder','passed','mean']
            row=['101001','64','mera','True','.3']
            write(x,[header,row])
            write(y,[header,row[:-1]+['.300000000000004']])
            c=Comparator();c.csv(x,y,'table');self.assertTrue(c.report()['passed'])
            for col,value in [(0,'101002'),(0,'101001.0'),(1,'63'),(2,'tree'),(3,'False'),(4,'nan'),(4,'.30000001')]:
                other=row.copy();other[col]=value;write(y,[header,other])
                c=Comparator();c.csv(x,y,'table');self.assertFalse(c.report()['passed'])
            write(y,[header]);c=Comparator();c.csv(x,y,'table');self.assertFalse(c.report()['passed'])
            write(y,[list(reversed(header)),row]);c=Comparator();c.csv(x,y,'table');self.assertFalse(c.report()['passed'])

    def test_complete_fixture_numeric_agreement_and_failure_records(self):
        with tempfile.TemporaryDirectory() as folder:
            ref,new=Path(folder)/'ref',Path(folder)/'new';ref.mkdir();new.mkdir()
            for root in [ref,new]:
                (root/'summary.json').write_text(json.dumps({'complete':True,'seed':101001,'count':64,'mean':.3,'resources':{'wall_seconds':1.}}))
                (root/'validation.json').write_text(json.dumps({'passed':True,'checks':{'residual':1e-15}}))
                (root/'table.csv').write_text('seed,mean\n101001,.3\n')
                np.savez(root/'values.npz',score=np.array([.3]))
            changed=json.loads((new/'summary.json').read_text());changed['mean']+=4e-15;changed['resources']['wall_seconds']=2.
            (new/'summary.json').write_text(json.dumps(changed))
            np.savez(new/'values.npz',score=np.array([.3+4e-15]))
            (new/'table.csv').write_text('seed,mean\n101001,.300000000000004\n')
            (new/'validation.json').write_text(json.dumps({'passed':True,'checks':{'residual':2e-15}}))
            r=compare_results(ref,new)
            self.assertTrue(r['passed']);self.assertFalse(r['exact_summary_equality']);self.assertFalse(r['all_arrays_bitwise_equal'])
            self.assertFalse(r['all_csv_tables_byte_equal']);self.assertFalse(r['validation_records_exact'])
            invalid=dict(changed);invalid['resources']={'wall_seconds':float('nan')}
            (new/'summary.json').write_text(json.dumps(invalid));r=compare_results(ref,new)
            self.assertFalse(r['passed']);json.dumps(r,allow_nan=False)
            (new/'summary.json').write_text(json.dumps(changed))
            altered=json.loads((new/'validation.json').read_text());altered['checks']['residual']=2e-10
            (new/'validation.json').write_text(json.dumps(altered));self.assertFalse(compare_results(ref,new)['passed'])
            (new/'validation.json').unlink();self.assertFalse(compare_results(ref,new)['passed'])
            (new/'validation.json').write_text(json.dumps({'passed':True,'checks':{'residual':1e-15}}))
            np.savez(new/'values.npz',other=np.array([.3]));r=compare_results(ref,new)
            self.assertFalse(r['passed']);self.assertTrue(any('keys' in e['message'] for e in r['issues']))

if __name__=='__main__':unittest.main()
