"""Small, declared follow-up grid; development-only rank fusion."""
import json,datetime,csv
from pathlib import Path
from search_core import retrieval_metrics
R=Path(__file__).resolve().parent
O=R/'reports/fusion';O.mkdir(parents=True,exist_ok=True)
PLAN={'scope':'Same 16-clip bunt development set; no held-out evaluation','rule':'weighted reciprocal rank fusion: (1-w)/(k+rank_a)+w/(k+rank_b)','k':[0,10,60],'weight_b':[0.25,0.5,0.75],'pairs':[['8_baseline_posture_max','16_home_plate_baseline_max'],['8_baseline_baseline_max','16_home_plate_baseline_max']],'selection':'Recall@3, then reciprocal rank, then first declared variant; report every trial'}
p=O/'plan.json'
if p.exists(): assert json.loads(p.read_text())['plan']==PLAN
else:p.write_text(json.dumps({'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan':PLAN},indent=2))
d=json.loads((R/'reports/ablations/results.json').read_text()); runs={r['id']:r for r in d['results']};relevant=set(d['human_positive_clips']);results=[]
for a,b in PLAN['pairs']:
 ar={r['clip_id']:i+1 for i,r in enumerate(runs[a]['ranking'])};br={r['clip_id']:i+1 for i,r in enumerate(runs[b]['ranking'])}
 for k in PLAN['k']:
  for w in PLAN['weight_b']:
   ranked=sorted([{'clip_id':c,'score':(1-w)/(k+ar[c])+w/(k+br[c]),'rank_a':ar[c],'rank_b':br[c]} for c in ar],key=lambda r:(-r['score'],r['clip_id']))
   results.append({'id':f'{a}+{b}_k{k}_w{w}', 'a':a,'b':b,'k':k,'weight_b':w,'ranking':ranked,'metrics':retrieval_metrics([r['clip_id'] for r in ranked],relevant,3),'positive_ranks':[i+1 for i,r in enumerate(ranked) if r['clip_id'] in relevant]})
best=max(results,key=lambda r:(r['metrics']['recall_at_k'],r['metrics']['reciprocal_rank']))
(O/'results.json').write_text(json.dumps({'plan':PLAN,'results':results,'selected_variant':best['id'],'n_variants':len(results),'selection_is_development_only':True},indent=2))
with (O/'summary.csv').open('w') as f:
 wr=csv.writer(f);wr.writerow(['variant','P@3','R@3','RR','positive_ranks'])
 for r in results:wr.writerow([r['id'],*r['metrics'].values(),';'.join(map(str,r['positive_ranks']))])
print(json.dumps({'selected':best['id'],'ranks':best['positive_ranks'],'metrics':best['metrics'],'n':len(results)}))
