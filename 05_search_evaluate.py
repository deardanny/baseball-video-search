"""Step 5/6: text search and an explicitly limited source-category retrieval check."""
from pathlib import Path
import argparse
import importlib
import json
import sqlite3
import numpy as np
import torch
from search_core import normalize, rank_clips, retrieval_metrics

ROOT=Path(__file__).resolve().parent
encoder=importlib.import_module('04_vectors')

def load_index():
    con=sqlite3.connect(ROOT/'data/baseball.sqlite')
    con.row_factory=sqlite3.Row
    rows=con.execute('''SELECT f.*,e.vector,e.dimensions,e.dtype,e.model_revision,
      c.raw_path,c.source_category FROM frames f JOIN embeddings e USING(frame_id)
      JOIN clips c USING(clip_id) ORDER BY f.clip_id,f.source_frame_index''').fetchall()
    expected=con.execute('SELECT COUNT(*) FROM frames').fetchone()[0]
    if len(rows)!=expected or not rows:
        raise ValueError('Embedding index incomplete; rerun 04_vectors.py')
    vectors,records=[],[]
    for row in rows:
        record=dict(row)
        if record['dimensions']!=512 or record['model_revision']!=encoder.MODEL_REVISION or record['dtype']!='<f4':
            raise ValueError('Unexpected model / dimension / dtype')
        vector=np.frombuffer(record.pop('vector'),dtype='<f4').copy()
        if vector.shape!=(512,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1,atol=1e-4):
            raise ValueError('Invalid stored embedding')
        vectors.append(vector);records.append(record)
    con.close()
    return np.stack(vectors),records

def text_vectors(texts):
    model,processor,device=encoder.load_encoder()
    inputs=processor(text=texts,padding=True,truncation=True,return_tensors='pt').to(device)
    with torch.inference_mode():
        return normalize(model.get_text_features(**inputs).float().cpu().numpy())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--query',help='English text for CLIP search; no API call is used')
    parser.add_argument('--method',choices=['max','mean'],default='max')
    args=parser.parse_args()
    vectors,records=load_index()
    if args.query:
        query=text_vectors([args.query])[0]
        results=rank_clips(vectors,records,query,args.method)[:5]
        print(json.dumps(dict(query=args.query,results=results),indent=2))
        return
    queries=json.loads((ROOT/'queries.json').read_text())
    queries_hash=__import__('hashlib').sha256((ROOT/'queries.json').read_bytes()).hexdigest()
    features=text_vectors([q['text'] for q in queries])
    all_results=[]
    methods={}
    for method in ('max','mean'):
        measures=[]
        for q,feature in zip(queries,features):
            ranked=rank_clips(vectors,records,feature,method)
            relevant=sorted({r['clip_id'] for r in records if r['source_category']==q['source_category']})
            metrics=retrieval_metrics([r['clip_id'] for r in ranked],relevant,3)
            measures.append(metrics)
            all_results.append(dict(**q,method=method,source_category_reference_clips=relevant,
                metrics=metrics,results=ranked))
        methods[method]={key:float(np.mean([m[key] for m in measures])) for key in measures[0]}
    # Exact expected random baseline for 16 ranked items, 2 relevant, k=3.
    from math import comb
    random_mrr=sum((comb(16-rank,1)/comb(16,2))/rank for rank in range(1,16))
    report=dict(evaluation_name='Weak source-category retrieval diagnostic, NOT semantic accuracy',
        n_clips=16,n_queries=8,k=3,reference_positives_per_query=2,query_file_sha256=queries_hash,
        methods=methods,random_expected=dict(precision_at_k=2/16,recall_at_k=3/16,reciprocal_rank=random_mrr),
        limitations=[
          'Filename labels are non-exhaustive: a hit/foul clip can also contain a swing.',
          'Ball/strike/foul outcomes may be impossible to determine from sampled still images.',
          '16 author-selected examples; no held-out test set; cannot claim production/generalization accuracy.',
          'No temporal encoder or ball tracking; mean/max aggregation discards frame order.',
          'Game identity is unavailable; cross-game independence is unverified.',
          'Both pooling methods are exploratory comparisons, not validated improvement.',
          'Human query relevance judgments are pending and must not be filled from model scores.'
        ])
    (ROOT/'reports/search_results.json').write_text(json.dumps(all_results,indent=2))
    (ROOT/'reports/evaluation.json').write_text(json.dumps(report,indent=2))
    # Blank 128-pair annotation sheet: user judges full clips without scores.
    import csv
    template=ROOT/'reports/human_relevance_template.csv'
    if template.exists():
        print('Preserved existing human annotation template; never overwrite user labels.')
        return
    with template.open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['query_id','query_text','clip_id','relevant_0_or_1','notes'])
        for q in queries:
            for cid in sorted({r['clip_id'] for r in records}):
                writer.writerow([q['id'],q['text'],cid,'',''])
    print(json.dumps(report,indent=2))
    for result in all_results[:8]:
        print(result['id'],[(r['clip_id'],round(r['score'],3)) for r in result['results'][:3]])

if __name__=='__main__':
    main()
