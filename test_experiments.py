import unittest, json, csv, importlib
from pathlib import Path
from search_core import retrieval_metrics
class EvidenceTests(unittest.TestCase):
 def test_saved_rankings_match_human_labels(self):
  root=Path(__file__).resolve().parent
  with (root/'reports/human_relevance_template.csv').open() as handle: rows=list(csv.DictReader(handle))
  labels={r['clip_id']:r['relevant_0_or_1'] for r in rows if r['query_id']=='bunt'}
  self.assertEqual(len(labels),16);self.assertEqual(sum(v=='1' for v in labels.values()),2)
  self.assertEqual(sum(r['relevant_0_or_1'] in ('0','1') for r in rows),16)
  relevant={c for c,v in labels.items() if v=='1'}
  total=0
  for directory in ('ablations','spatial_ablations','fusion'):
   d=json.loads((root/f'reports/{directory}/results.json').read_text())
   total+=len(d['results'])
   for r in d['results']:
    actual=retrieval_metrics([x['clip_id'] for x in r['ranking']],relevant,3)
    self.assertEqual(actual,r['metrics'])
  self.assertEqual(total,98)
 def test_pooling_rejects_bad_metadata(self):
  import numpy as np
  pool=importlib.import_module('08_ablation').pool_scores
  with self.assertRaises(ValueError):pool(np.array([1.0]),[],'max')
  with self.assertRaises(ValueError):pool(np.array([float('nan')]),[{}],'max')
if __name__=='__main__':unittest.main()
