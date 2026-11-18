"""Small failure-focused tests, plus integrity checks on the real generated index."""
import importlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import numpy as np
from PIL import Image
from search_core import normalize, rank_clips, retrieval_metrics
frames_module=importlib.import_module('02_frames')
ROOT=Path(__file__).resolve().parent

class RetrievalTests(unittest.TestCase):
    def test_duplicate_frames_do_not_fill_all_top_results(self):
        vectors=np.array([[1,0],[.99,.01],[0,1]],dtype=np.float32)
        records=[dict(clip_id='A',frame_id='A1'),dict(clip_id='A',frame_id='A2'),dict(clip_id='B',frame_id='B1')]
        result=rank_clips(vectors,records,[1,0])
        self.assertEqual([r['clip_id'] for r in result],['A','B'])
        self.assertEqual(result[0]['frame_id'],'A1')
        self.assertEqual([r['clip_id'] for r in rank_clips(vectors,records,[1,0],exclude_clip='A')],['B'])

    def test_corrupt_or_misaligned_vectors_fail_loudly(self):
        for vectors in ([[0,0]],[[float('nan'),1]],[[float('inf'),1]]):
            with self.assertRaises(ValueError):normalize(vectors)
        with self.assertRaises(ValueError):rank_clips([[1,0]],[],[1,0])
        with self.assertRaises(ValueError):rank_clips([[1,0]],[dict(clip_id='A')],[1,0,0])

    def test_metrics_hand_calculated_and_deduplicated(self):
        m=retrieval_metrics(['B','A','C','D'],['A','D'],3)
        self.assertAlmostEqual(m['precision_at_k'],1/3)
        self.assertAlmostEqual(m['recall_at_k'],1/2)
        self.assertAlmostEqual(m['reciprocal_rank'],1/2)
        with self.assertRaises(ValueError):retrieval_metrics(['A','A','B'],['A'],3)

    def test_time_sampling_uses_duration_not_frame_count(self):
        # Unequal-duration toy frames; time midpoint belongs to long second frame.
        indices=frames_module.sample_indices(np.array([0,100,900]),np.array([100,800,100]),1)
        self.assertEqual(indices.tolist(),[1])

    def test_invalid_media_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'broken.gif';path.write_bytes(b'not an image')
            with self.assertRaises(Exception):frames_module.decode_gif(path)
            image=Image.new('RGB',(8,8),'red');image.save(path,format='GIF',duration=100)
            with self.assertRaises(ValueError):frames_module.decode_gif(path)

    def test_real_database_provenance_and_vector_integrity(self):
        con=sqlite3.connect(ROOT/'data/baseball.sqlite')
        self.assertEqual(con.execute('PRAGMA integrity_check').fetchone()[0],'ok')
        self.assertFalse(con.execute('PRAGMA foreign_key_check').fetchall())
        self.assertEqual(con.execute('SELECT COUNT(*) FROM clips').fetchone()[0],16)
        self.assertEqual(con.execute('SELECT COUNT(*) FROM frames').fetchone()[0],128)
        self.assertEqual(con.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0],128)
        self.assertEqual(con.execute('SELECT COUNT(*) FROM frames f JOIN clips c USING(clip_id) WHERE f.timestamp_ms>=c.duration_ms').fetchone()[0],0)
        for blob,dimension,norm in con.execute('SELECT vector,dimensions,l2_norm FROM embeddings'):
            self.assertEqual(len(blob),dimension*4)
            self.assertTrue(np.isfinite(np.frombuffer(blob,dtype='<f4')).all())
            self.assertAlmostEqual(norm,1,places=5)
        for path, in con.execute('SELECT relative_path FROM frames'):
            self.assertTrue((ROOT/path).is_file())
        vectors=np.load(ROOT/'data/frame_vectors.npy',allow_pickle=False)
        ids=json.loads((ROOT/'data/vector_frame_ids.json').read_text())
        for fid,vector in zip(ids,vectors):
            stored=np.frombuffer(con.execute('SELECT vector FROM embeddings WHERE frame_id=?',(fid,)).fetchone()[0],dtype='<f4')
            np.testing.assert_array_equal(stored,vector)
        con.close()

if __name__=='__main__':unittest.main(verbosity=2)
