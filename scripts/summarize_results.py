"""Recalculate source-level mean deltas from the archived extension CSV."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

path = Path(__file__).resolve().parents[1] / 'results/extension_30_sources.csv'
with path.open(newline='') as handle:
    rows = list(csv.DictReader(handle))
metrics = [name for name in rows[0] if name.endswith('_delta')]
source_ids = {row['source_id'] for row in rows}
print(f'{len(source_ids)} sources, {len(rows)} paired edits')
for metric in metrics:
    by_source = defaultdict(list)
    for row in rows:
        by_source[row['source_id']].append(float(row[metric]))
    result = mean(mean(values) for values in by_source.values())
    print(f'{metric}: {result:+.8f}')
