"""Re-evaluate existing rankings once ALL 128 human relevance judgments are filled."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from search_core import retrieval_metrics
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('csv_path');args=parser.parse_args()
    rows=list(csv.DictReader(open(args.csv_path,newline='')))
    queries=json.loads((ROOT/'queries.json').read_text())
    clips=json.loads((ROOT/'data/manifest.json').read_text())
    expected={(q['id'],c['clip_id']) for q in queries for c in clips}
    pairs=[(r['query_id'],r['clip_id']) for r in rows]
    if len(pairs)!=len(set(pairs)) or set(pairs)!=expected:
        raise ValueError('Need exactly one judgment per query/clip pair (128 unique pairs)')
    if any(r['relevant_0_or_1'] not in ('0','1') for r in rows):
        raise ValueError('Human judgments are incomplete. Fill every row with 0 or 1; do not guess.')
    relevant={q['id']:{r['clip_id'] for r in rows if r['query_id']==q['id'] and r['relevant_0_or_1']=='1'} for q in queries}
    rankings=json.loads((ROOT/'reports/search_results.json').read_text())
    results=[]
    for ranking in rankings:
        positives=relevant[ranking['id']]
        metrics=retrieval_metrics([r['clip_id'] for r in ranking['results']],positives,3) if positives else None
        results.append(dict(query_id=ranking['id'],method=ranking['method'],relevant=sorted(positives),metrics=metrics,
            note=None if positives else 'No relevant clip judged: recall/MRR undefined, excluded from means'))
    means={}
    for method in ('max','mean'):
        valid=[r['metrics'] for r in results if r['method']==method and r['metrics']]
        means[method]={key:float(np.mean([r[key] for r in valid])) for key in valid[0]} if valid else None
    report=dict(basis='User-supplied complete relevance labels; local exploratory evaluation',
        annotation_file=str(Path(args.csv_path).resolve()),means=means,results=results)
    (ROOT/'reports/human_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
