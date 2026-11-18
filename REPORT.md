# Retrieval development report

## Question and protocol

Can text retrieve an attempted bunt from a small baseball video collection, and can we identify which preprocessing or scoring change helps? The baseline uses eight equal-time-bin frames, CLIP ViT-B/32, the prompt “A baseball batter bunting the ball.”, and maximum frame cosine per clip. Human review marked `bunt1` and `bunt2` relevant and the other 14 clips irrelevant. The clip filenames provide provenance only and are not model inputs.

## What failed

The two leading false positives were `foul1` (review clip 08, 1.80 s, cosine 0.32465) and `ball2` (clip 11, 0.52 s, 0.32209). The highest-scoring frame in `bunt1` was preparation at 1.28 s (0.31731), above a visible bunt posture at 1.84 s (0.30148). Therefore, simply missing the action frame does not fully explain the error. A still-image encoder can confuse similar baseball poses, and max pooling ignores temporal order. Background bias is a plausible explanation, not a measured causal finding.

## Experiments

Each experiment group recorded its plan before its execution, **after** earlier results had already been seen. This is iterative development, not preregistered independent validation.

| Group | Settings | Configurations | Best recall@3 |
|---|---|---:|---:|
| Main ablation | 8/16 frames × whole/fixed plate crop × 4 text scores × max/top-2 pooling | 32 | 0.50 |
| Spatial follow-up | 8/16/32 frames × 2 padded crops × 4 text scores × 2 pooling rules | 48 | 0.50 |
| Rank fusion | 2 fixed pairs × 3 reciprocal-rank constants × 3 weights | 18 | 0.50 |

All 98 rows are available in the three `summary.csv` files. Ranking records include frame timestamps and individual scores. The baseline 8-frame index used saved JPEGs; denser variants decode the GIFs directly, so frame count comparisons also have a small image-encoding difference. Fixed crop boxes are heuristics applied identically across clips, not learned batter detection.

## Selected result

Choose `8_baseline_posture_max`: retain the same 128 frames, image vectors, model, database, and pooling; change only the text to describe a bat held horizontally in front of the body. Baseline top four: foul1, ball2, bunt2, bunt1. Selected top four: bunt2, foul1, ball2, neg1. Relevant ranks move from [3, 4] to [1, 7].

| Metric | Baseline | Selected | Interpretation |
|---|---:|---:|---|
| Precision@3 | 1/3 | 1/3 | One of the three results is relevant |
| Recall@3 | 1/2 | 1/2 | One of two relevant clips is retrieved |
| Reciprocal rank | 1/3 | 1 | First relevant clip appears sooner |

With a single query, mean reciprocal rank equals reciprocal rank. Reporting a general “MRR improvement” across unseen queries would be misleading. No held-out experiment has been completed. The simple prompt is preferred over larger/fused variants because they do not improve coverage.

## Next meaningful experiment

Freeze this configuration before collecting new clips. Build a separate test set from different games, with many visually similar non-bunt hard negatives; obtain labels without rankings. Compare whole-frame CLIP against a batter-region detector and a temporal video encoder, keeping test labels sealed while selecting settings. Report query-level metrics and failures. No such future results are claimed here.

## Engineering checks

Tests verify duplicate frames cannot occupy multiple clip-level results, malformed/zero vectors fail loudly, metric values match hand calculations, and source/frame/embedding joins are intact. Additional checks preserve the human labels and recompute saved experiment metrics. Original baseline files and annotations were not overwritten by optimization.
