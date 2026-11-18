"""Controlled development-set ablations; never overwrite baseline or human labels.

All choices are recorded before evaluation. These 16 clips have already been
inspected; improved scores are development results, not held-out performance.
"""
from pathlib import Path
import importlib
import json
import hashlib
import csv
import datetime
import numpy as np
import torch
from PIL import Image, ImageOps
from search_core import normalize, retrieval_metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports/spatial_ablations'
POSITIVE = [
    'A baseball batter bunting the ball.',
    'A baseball batter attempting a bunt, holding the bat horizontally in front of the body to meet the pitch.',
    'A baseball hitter squares around to bunt, extending the bat in front of home plate.',
    'A baseball player gently taps a pitch with the bat held out, attempting a bunt.',
]
NEGATIVE = [
    'A baseball batter waiting for a pitch, holding the bat up near the shoulder.',
    'A baseball batter taking a full swing at a pitch.',
    'A baseball pitcher delivering a pitch while the batter watches.',
]
PLAN = dict(
    scope='Exploratory development set: one manually judged query, 16 clips, 2 positives; no held-out test.',
    positives=POSITIVE, negatives=NEGATIVE,
    views=['plate_padded', 'tight_padded'],
    samples_per_clip=[8, 16, 32],
    text_methods=['baseline', 'posture', 'ensemble', 'contrastive'],
    pooling=['max', 'top2'],
    crop_rule='Fixed normalized boxes: plate_padded=(.25,.08,.80,.64), tight_padded=(.32,.10,.74,.56); aspect-preserving resize and gray letterbox to224square. Same rule for every clip; no object detection.',
    contrastive_rule='mean positive-prompt cosine minus max negative-prompt cosine, then clip pooling',
    selection='highest Recall@3, then RR, then Precision@3; ties prefer simpler variant by run order',
    no_finetuning=True,
)

def pool_scores(scores, records, method):
    if len(scores) != len(records) or not np.isfinite(scores).all():
        raise ValueError('Invalid scores / metadata')
    if method not in ('max', 'top2'):
        raise ValueError('Unknown pool')
    groups = {}
    for i, rec in enumerate(records):
        groups.setdefault(rec['clip_id'], []).append(i)
    ranked = []
    for cid, ids in groups.items():
        ordered = sorted(ids, key=lambda i: (-float(scores[i]), records[i]['timestamp_ms']))
        used = ordered[:1 if method == 'max' else 2]
        ranked.append(dict(clip_id=cid, score=float(np.mean(scores[used])),
                           evidence=[dict(frame_id=records[i]['frame_id'], timestamp_ms=records[i]['timestamp_ms'], score=float(scores[i])) for i in used]))
    return sorted(ranked, key=lambda r: (-r['score'], r['clip_id']))

def crop_image(image, view):
    boxes = dict(plate_padded=(.25,.08,.80,.64),tight_padded=(.32,.10,.74,.56))
    if view not in boxes: raise ValueError('Unknown view')
    w,h=image.size;x0,y0,x1,y1=boxes[view]
    crop=image.crop((round(w*x0),round(h*y0),round(w*x1),round(h*y1)))
    return ImageOps.pad(crop,(224,224),method=Image.Resampling.BICUBIC,color=(122,116,104))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path=OUT/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())['plan'] != PLAN:
            raise ValueError('Plan changed: use a new experiment directory')
    else:
        plan_path.write_text(json.dumps(dict(recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=PLAN),indent=2))
    protected=[ROOT/'reports/search_results.json',ROOT/'reports/human_relevance_template.csv',ROOT/'data/baseball.sqlite']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    labels=list(csv.DictReader((ROOT/'reports/human_relevance_template.csv').open()))
    judged={r['clip_id']:int(r['relevant_0_or_1']) for r in labels if r['query_id']=='bunt' and r['relevant_0_or_1'] in ('0','1')}
    if len(judged)!=16: raise ValueError('Need all 16 explicit bunt judgments')
    relevant={cid for cid,label in judged.items() if label}
    search=importlib.import_module('05_search_evaluate')
    encoder=importlib.import_module('04_vectors')
    sampler=importlib.import_module('02_frames')
    original, original_records=search.load_index()
    model,processor,device=encoder.load_encoder()
    with torch.inference_mode():
        t=processor(text=POSITIVE+NEGATIVE,padding=True,return_tensors='pt').to(device)
        text=normalize(model.get_text_features(**t).float().cpu().numpy())
    results=[]
    for count in PLAN['samples_per_clip']:
        if count==8:
            records=original_records
            images=[Image.open(ROOT/r['relative_path']).convert('RGB') for r in records]
        else:
            images=[];records=[]
            for path in sorted((ROOT/'data/raw').glob('*.gif')):
                frames,times,durations=sampler.decode_gif(path)
                for i in sampler.sample_indices(times,durations,count):
                    images.append(frames[i].copy())
                    records.append(dict(clip_id=path.stem,frame_id=f'{path.stem}_{int(i):04d}',timestamp_ms=int(times[i])))
                for im in frames: im.close()
        for view in PLAN['views']:
            if count==8 and view=='baseline':
                vectors=original
            else:
                batches=[]
                with torch.inference_mode():
                    for start in range(0,len(images),32):
                        batch=[crop_image(im,view) for im in images[start:start+32]]
                        x=processor(images=batch,return_tensors='pt').to(device)
                        batches.append(normalize(model.get_image_features(**x).float().cpu().numpy()))
                vectors=np.concatenate(batches)
            similarities=vectors@text.T
            variants=dict(baseline=similarities[:,0],posture=similarities[:,1],
                          ensemble=similarities[:,:4].mean(axis=1),
                          contrastive=similarities[:,:4].mean(axis=1)-similarities[:,4:].max(axis=1))
            for name,scores in variants.items():
                for pooling in PLAN['pooling']:
                    ranking=pool_scores(scores,records,pooling)
                    metrics=retrieval_metrics([r['clip_id'] for r in ranking],relevant,3)
                    run=dict(id=f'{count}_{view}_{name}_{pooling}',frames_per_clip=count,view=view,text_method=name,pooling=pooling,
                             metrics=metrics,ranking=ranking,positive_ranks=[i+1 for i,r in enumerate(ranking) if r['clip_id'] in relevant])
                    results.append(run)
                    print(run['id'],run['positive_ranks'],metrics,flush=True)
            np.save(OUT/f'vectors_{count}_{view}.npy',vectors,allow_pickle=False)
            (OUT/f'frames_{count}_{view}.json').write_text(json.dumps(records,indent=2))
        for im in images: im.close()
    for p in protected:
        assert hashes[str(p.relative_to(ROOT))]==hashlib.sha256(p.read_bytes()).hexdigest()
    selected=max(results,key=lambda r:(r['metrics']['recall_at_k'],r['metrics']['reciprocal_rank'],r['metrics']['precision_at_k']))
    report=dict(plan=PLAN,baseline_hashes=hashes,human_positive_clips=sorted(relevant),
                n_variants=len(results),results=results,selected_variant=selected['id'],
                selection_is_development_only=True,completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['variant','precision_at_3','recall_at_3','rr','positive_ranks'])
        for r in results:w.writerow([r['id'],*r['metrics'].values(),';'.join(map(str,r['positive_ranks']))])
    print('Selected development variant:',selected['id'],selected['metrics'],flush=True)

if __name__=='__main__':main()
