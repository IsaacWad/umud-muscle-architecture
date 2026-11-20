"""Replay saved final adjustments; does not run inference or recompute weighted PA."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rebuild(output):
    def load(name):
        with (ROOT/'results'/name).open(encoding='utf-8-sig',newline='') as handle:
            return list(csv.DictReader(handle))
    rows = load('verified_rulers_submission.csv')
    pa = {r['image_id']:r['weighted_pa_deg'] for r in load('weighted_pa_audit.csv')}
    patches = {r['image_id']:r for r in load('final_export_adjustments.csv')}
    if len(rows)!=309 or set(pa)!=set(r['image_id'] for r in rows):
        raise ValueError('Final export requires matching 309-row baseline and PA audit')
    for row in rows:
        row['pa_deg'] = pa[row['image_id']]
        if row['image_id'] in patches:
            row.update({k:patches[row['image_id']][k] for k in ['fl_mm','mt_mm']})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['image_id','pa_deg','fl_mm','mt_mm'])
        writer.writeheader()
        writer.writerows(rows)
    reference={r['image_id']:r for r in load('final_submission.csv')}
    error=max(abs(float(r[k])-float(reference[r['image_id']][k])) for r in rows for k in ['pa_deg','fl_mm','mt_mm'])
    if error>1e-9:
        raise ValueError(f'Replayed values differ from archived final submission: {error}')
    print(f'Rebuilt {len(rows)} rows from saved adjustments; maximum numeric difference {error:g}. No inference performed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/rebuilt_final_submission.csv')
    rebuild(parser.parse_args().output)
