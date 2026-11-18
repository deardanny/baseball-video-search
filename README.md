# Baseball video search and retrieval evaluation

**Daniel (Heedong) Jang · CMU Computer Vision course project · November 2026**

A small, reproducible pipeline that turns baseball video into searchable data: source clips → timestamped frames → CLIP image vectors → SQLite → text-to-frame similarity → clip-level rankings → human relevance evaluation.

The useful result is a working data pipeline and an auditable error-analysis process. On one **16-clip development query**, a posture-specific prompt moved the first relevant bunt clip from rank **3 to 1**. Precision@3 stayed **33.3%** and Recall@3 stayed **50%**. This is not a held-out accuracy claim.

## What is implemented

- Pin and hash-verify 16 public MLB-YouTube example GIFs, retaining source URLs and revision metadata.
- Extract 8 frames at evenly spaced time-bin midpoints per clip: **128 frames**.
- Encode each image with pretrained CLIP ViT-B/32 into a normalized **512-dimensional float32 vector**.
- Store vectors with frame IDs, timestamps, clip IDs, source provenance, and model version in SQLite. Search joins by IDs rather than assuming array order.
- Encode English text in the same vector space, calculate cosine similarity, then return one result per clip with the highest-scoring frame as evidence.
- Preserve explicit human judgments separately from filename-derived diagnostic categories.
- Compare **98 development configurations** spanning prompt specificity, frame count, fixed crops, pooling, contrastive scoring, and rank fusion. All trials, including failures, are retained.

## Result at a glance

| Bunt query, two relevant clips | Baseline | Selected prompt |
|---|---:|---:|
| First relevant clip rank | 3 | 1 |
| Reciprocal rank | 0.333 | 1.000 |
| Precision@3 | 0.333 | 0.333 |
| Recall@3 | 0.500 | 0.500 |
| Ranks of both relevant clips | 3, 4 | 1, 7 |

The selected variant trades better first-hit ranking for a worse rank for the other relevant clip. It does **not** improve overall coverage. [Read the full experiment report](REPORT.md).

## Reproduce

Use Python 3.12. The model download is approximately 605 MB and the example GIFs total about 40 MB. The pipeline runs locally on CPU or Apple MPS; it does not call a paid inference API.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python 01_download.py
.venv/bin/python 02_frames.py
.venv/bin/python 03_model_download.py
.venv/bin/python 04_vectors.py
.venv/bin/python 05_search_evaluate.py
.venv/bin/python test_pipeline.py
.venv/bin/python test_experiments.py
.venv/bin/python 08_ablation.py
.venv/bin/python 09_spatial_ablation.py
.venv/bin/python 10_rank_fusion.py
```

Run the selected query against the same index:

```bash
.venv/bin/python 05_search_evaluate.py --query "A baseball batter attempting a bunt, holding the bat horizontally in front of the body to meet the pitch."
```

`02_frames.py` rebuilds only this project's local database; rerun steps 04–05 afterward. Experiment plans refuse to change silently. Reproduction can vary at floating-point precision across hardware. Recomputed evidence may differ near ties.

## Data and evaluation boundaries

Only the bunt query has complete human judgments: 16 clips, 2 relevant. An attempted bunt counts even without contact or a successful play. The other seven query sets remain unjudged, with blank annotations intentionally preserved. Do not treat filename labels or model scores as human ground truth. `06_evaluate_human.py` is for the complete 128-pair multiquery review and intentionally rejects incomplete annotations.

The same 16 clips were used to diagnose errors and select variants. Results are exploratory development results, not an independent test. One person's judgments, a tiny sample, and many comparisons limit generalization. No model fine-tuning, learned action detector, full-video temporal model, or production deployment is claimed. AI assisted implementation and experiment execution; Daniel defined relevance and manually reviewed all 16 bunt clips.

## Sources and media

- [MLB-YouTube research repository](https://github.com/piergiaj/mlb-youtube), examples pinned to `7187fbc1801daab9141d5b6a9ad4b679a96b15bf`.
- [Fine-grained Activity Recognition in Baseball Videos, CVsports 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w34/Piergiovanni_Fine-Grained_Activity_Recognition_CVPR_2018_paper.pdf).
- [CLIP](https://github.com/openai/CLIP) and [model card](https://github.com/openai/CLIP/blob/main/model-card.md).
- [Pinned ViT-B/32 weights](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268).

Raw broadcast GIFs, extracted frames, model weights, and the generated database are not redistributed in this repository. Source links and download/verification scripts are provided. Public access to source examples does not transfer broadcast-media rights.
