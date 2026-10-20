"""Diagnostic cross-validation of a latency floor; never modifies the predictor.

Fits offset on other points in each kernel category. This is exploratory
leave-one-out validation on previously inspected paper data, not blind accuracy.
"""
import csv, json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'results/reproduce/stages/fig08/model_points.csv').open()))
output=[]
for row in rows:
    peers=[x for x in rows if x['category']==row['category'] and x['point_index']!=row['point_index']]
    # Median residual minimizes absolute time error, not percentage error.
    offset=statistics.median(float(x['gpu_reference_ms'])-float(x['current_model_ms']) for x in peers)
    p=float(row['current_model_ms']); y=float(row['gpu_reference_ms'])
    output.append(dict(category=row['category'],point_index=int(row['point_index']),base_ms=p,reference_ms=y,training_offset_ms=offset,prediction_ms=max(0,p+offset),ape=abs(max(0,p+offset)-y)/y))
report=dict(scope='Exploratory within-category leave-one-out constant latency correction. Not a blind test, not the shipped model, no point excluded.',n=len(output),mape=sum(x['ape'] for x in output)/len(output),max_ape=max(x['ape'] for x in output),categories={k:statistics.mean(x['ape'] for x in output if x['category']==k) for k in sorted({x['category'] for x in output})},rows=output)
(ROOT/'validation/latency_floor_diagnostic.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
