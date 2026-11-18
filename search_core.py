"""Pure retrieval functions: numeric features only, never filename labels."""
import numpy as np

def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or (norms < 1e-10).any():
        raise ValueError('Nonfinite or zero-length vector')
    return vectors / norms

def rank_clips(vectors, records, query, method='max', exclude_clip=None):
    if len(vectors) != len(records):
        raise ValueError('Embedding / metadata row count mismatch')
    vectors, query = normalize(vectors), normalize(query)
    if query.ndim != 1 or vectors.ndim != 2 or vectors.shape[1] != query.shape[0]:
        raise ValueError('Embedding dimension mismatch')
    if method not in ('max','mean'):
        raise ValueError('Choose max or mean')
    scores = vectors @ query
    groups = {}
    for index, record in enumerate(records):
        if record['clip_id'] != exclude_clip:
            groups.setdefault(record['clip_id'], []).append(index)
    ranked = []
    for cid, indices in groups.items():
        best_index = max(indices, key=lambda i: float(scores[i]))
        score = float(scores[best_index]) if method == 'max' else float(normalize(vectors[indices].mean(axis=0)) @ query)
        ranked.append(dict(records[best_index], score=score, clip_id=cid, aggregation=method))
    return sorted(ranked, key=lambda row: (-row['score'],row['clip_id']))

def retrieval_metrics(ranked_ids, relevant_ids, k=3):
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError('Duplicate clip IDs would inflate retrieval metrics')
    relevant = set(relevant_ids)
    if not relevant or k <= 0 or len(ranked_ids)<k:
        raise ValueError('Need relevant clips and at least k ranked clips')
    hits = sum(cid in relevant for cid in ranked_ids[:k])
    reciprocal_rank = next((1/(i+1) for i,cid in enumerate(ranked_ids) if cid in relevant),0.0)
    return dict(precision_at_k=hits/k, recall_at_k=hits/len(relevant), reciprocal_rank=reciprocal_rank)
