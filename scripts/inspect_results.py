"""Validate and summarize the archived final submission using the standard library."""
import argparse
import csv
import json
import math
from pathlib import Path


def inspect(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = ['image_id','pa_deg','fl_mm','mt_mm']
        if reader.fieldnames != required:
            raise ValueError(f'Expected columns {required}, got {reader.fieldnames}')
        rows = list(reader)
    ids = [r['image_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate image identifiers')
    for r in rows:
        for key in required[1:]:
            value = float(r[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid {key} for {r["image_id"]}')
    return {'rows':len(rows),'unique_image_ids':len(set(ids)),
            'ranges':{key:[min(float(r[key]) for r in rows),max(float(r[key]) for r in rows)] for key in required[1:]}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[1]/'results/final_submission.csv')
    args = parser.parse_args()
    print(json.dumps(inspect(args.input),indent=2))
