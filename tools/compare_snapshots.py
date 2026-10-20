"""Compare source integration output against a separate native-provider process."""
import json
import math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] / 'validation'
a = json.loads((ROOT / 'native_snapshots.json').read_text())
b = json.loads((ROOT / 'source_snapshots.json').read_text())
checks = 0
failures = []
def compare(x, y, path=''):
    global checks
    if isinstance(x, dict) and isinstance(y, dict):
        for key in sorted(x.keys() | y.keys()): compare(x.get(key), y.get(key), path+'/'+key)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y): failures.append(dict(path=path, native=x, source=y))
        for i, (xx, yy) in enumerate(zip(x, y)): compare(xx, yy, path+'/'+str(i))
    else:
        checks += 1
        equal = x == y
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            equal = equal or math.isclose(x, y, rel_tol=2e-14, abs_tol=1e-25) or (math.isnan(x) and math.isnan(y))
        if not equal: failures.append(dict(path=path, native=x, source=y))
compare(a, b)
report = dict(checks=checks, failures=failures, scope='11 q01/q02 profile states, SM/cluster/DRAM/design-point clones, 49 NoC IDs and capacity queries; not exhaustive API validation')
(ROOT/'integration_mismatches.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
raise SystemExit(bool(failures))
