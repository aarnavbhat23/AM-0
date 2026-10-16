"""Verify stored data/source hashes and independently recompute reported metrics."""
import argparse
import json
import hashlib
from pathlib import Path
import subprocess
import numpy as np

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--development', type=Path, required=True)
parser.add_argument('--evaluation', type=Path, required=True)
args=parser.parse_args()
dev=args.development.resolve()
evaldir=args.evaluation.resolve()
assert json.loads((evaldir/'metadata.json').read_text())['status']=='complete'
checked=0
for folder, recordfile, predfile, split in [(dev,'development.json','validation_predictions.npz','validation'),(evaldir,'measurements.json','predictions.npz','test')]:
 meta=json.loads((folder/'metadata.json').read_text())
 for path,expected in meta['source_sha256'].items():
  actual=subprocess.check_output(['git','show',f"{meta['commit']}:{path}"],cwd=root)
  assert hashlib.sha256(actual).hexdigest()==expected,path
 contents=json.loads((folder/recordfile).read_text())
 preds=np.load(folder/predfile)
 for name,entry in contents['data'].items():
  data=np.load(folder/f'data_{name}.npz')
  for part,expected in entry['hashes'].items():
   digest=hashlib.sha256()
   for key in sorted(k[len(part)+1:] for k in data.files if k.startswith(part+'_')):
    array=np.ascontiguousarray(data[part+'_'+key])
    digest.update(key.encode()); digest.update(str(array.shape).encode()); digest.update(array.dtype.str.encode());digest.update(array.tobytes())
   assert digest.hexdigest()==expected,(name,part)
 for r in contents['records']:
  key=f"{r['family']}_{r['seed']}_{r['model']}_{r['eta']}" if split=='validation' else f"{r['family']}_{r['seed']}_{r['mode']}_{r['model']}"
  data=np.load(folder/f"data_{r['family']}_{r['seed']}.npz")
  y=data[split+'_y']; p=preds[key]; scale=max(float(data['train_y'].var()),0.01)
  assert abs(scale-r['scale'])<1e-14
  squared=(p-y)**2
  assert np.isfinite(p).all()
  assert np.isclose(squared.mean()/scale,r['nmse'],atol=1e-12)
  ids=data[split+'_sequence']
  assert np.allclose([squared[ids==i].mean()/scale for i in np.unique(ids)],r['sequence_nmse'])
  withheld=data[split+'_withheld']
  assert int(withheld.sum())==r['withheld_count']
  if withheld.any(): assert np.isclose(squared[withheld].mean()/scale,r['withheld_nmse'])
  checked+=1

content=json.loads((evaldir/'measurements.json').read_text()); rows=content['records']; audit=json.loads((evaldir/'audit.json').read_text())
summary={'verified_prediction_arrays':checked,'evaluation_records':len(rows),'audit':audit,'groups':{},'comparisons':{},'resources':{}}
methods=list(dict.fromkeys(r['model'] for r in rows))
for mode in ['one_pass','time_cap']:
 for fam in ['adaptive','resource','delayed']:
  group=[r for r in rows if r['family']==fam and r['mode']==mode]
  summary['groups'][f'{mode}/{fam}']={m:{'mean_nmse':float(np.mean([r['nmse'] for r in group if r['model']==m])), 'mean_withheld_nmse':float(np.mean([r['withheld_nmse'] for r in group if r['model']==m]))} for m in methods}
  a={r['seed']:r for r in group if r['model']=='assembly'}
  summary['comparisons'][f'{mode}/{fam}']={}
  for base in ['adaptive_lms','quadratic_lms','history_lms','assembly_fixed','assembly_no_pool','assembly_shuffled','assembly_untrained']:
   b={r['seed']:r for r in group if r['model']==base}
   rel=np.array([(b[s]['nmse']-a[s]['nmse'])/b[s]['nmse'] for s in a])
   summary['comparisons'][f'{mode}/{fam}'][base]={'median_relative_improvement':float(np.median(rel)), 'range':[float(rel.min()),float(rel.max())],'wins':int((rel>1e-10).sum()),'ties':int((abs(rel)<=1e-10).sum()),'losses':int((rel < -1e-10).sum())}
 for m in methods:
  group=[r for r in rows if r['model']==m and r['mode']==mode]
  summary['resources'][f'{mode}/{m}']={'state_bytes_range':[min(r['state_bytes'] for r in group),max(r['state_bytes'] for r in group)],'training_seconds_range':[min(r['training_seconds'] for r in group),max(r['training_seconds'] for r in group)],'median_train_seconds':float(np.median([r['training_seconds'] for r in group])),'updates_range':[min(r['updates'] for r in group),max(r['updates'] for r in group)],'incomplete_data_cases':sum(r['all_data_seen'] is False for r in group),'max_overrun_seconds':max(r['overrun_seconds'] for r in group),'median_inference_seconds':float(np.median([r['inference_seconds'] for r in group]))}
summary['predictive_gate_passed']=all(summary['comparisons'][f'{mode}/{fam}'][base]['median_relative_improvement']>=0.1 for mode in ['one_pass','time_cap'] for fam in ['adaptive','resource','delayed'] for base in ['adaptive_lms','quadratic_lms']) and not any(v['incomplete_data_cases'] for v in summary['resources'].values())
print(json.dumps(summary,indent=2))
