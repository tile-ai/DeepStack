"""Evaluate archived paper predictions, without claiming they are rerun results."""
import csv
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = []
for path in sorted((ROOT / 'validation/reference/fig08').glob('*.csv')):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            measured = row.get('ref_time/ms(gpu)')
            predicted = row.get('overall_time/ms(modeling, overlap)')
            if measured and predicted and float(measured) > 0:
                y, p = float(measured), float(predicted)
                rows.append(dict(dataset=path.stem, row=index, prediction_ms=p, measurement_ms=y, ape=abs(p-y)/y))
report = dict(provenance='Archived paper CSV predictions; not rerun predictions or independent new hardware measurements', n=len(rows), mape=sum(r['ape'] for r in rows)/len(rows), max_ape=max(r['ape'] for r in rows), rows=rows)
(ROOT / 'validation/paper_fig08_baseline.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2))
