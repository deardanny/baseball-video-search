"""Step 3/4: CLIP image embeddings, normalized and stored beside frame metadata."""
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import time
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from search_core import normalize

ROOT=Path(__file__).resolve().parent
MODEL_ID='openai/clip-vit-base-patch32'
MODEL_REVISION='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'

def load_encoder():
    torch.set_num_threads(4)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model=CLIPModel.from_pretrained(ROOT/'models/clip',local_files_only=True,weights_only=True).eval().to(device)
    processor=CLIPProcessor.from_pretrained(ROOT/'models/clip',local_files_only=True,use_fast=False)
    return model,processor,device

def main():
    started=time.perf_counter()
    con=sqlite3.connect(ROOT/'data/baseball.sqlite')
    con.row_factory=sqlite3.Row
    records=[dict(r) for r in con.execute('SELECT * FROM frames ORDER BY clip_id,source_frame_index')]
    model,processor,device=load_encoder()
    print(f'Device={device}; pretrained CLIP; {len(records)} frames; no training',flush=True)
    batches=[]
    with torch.inference_mode():
        for start in range(0,len(records),16):
            images=[Image.open(ROOT/r['relative_path']).convert('RGB') for r in records[start:start+16]]
            inputs=processor(images=images,return_tensors='pt').to(device)
            encoded=model.get_image_features(**inputs).float().cpu().numpy()
            batches.append(normalize(encoded))
            for im in images: im.close()
            print(f'Encoded {min(start+16,len(records))}/{len(records)}',flush=True)
    vectors=np.concatenate(batches).astype('<f4')
    assert vectors.shape==(len(records),512)
    with con:
        con.execute('DELETE FROM embeddings')
        for record,vector in zip(records,vectors):
            con.execute('INSERT INTO embeddings VALUES(?,?,?,?,?,?,?)',
                (record['frame_id'],MODEL_ID,MODEL_REVISION,512,'<f4',float(np.linalg.norm(vector)),vector.tobytes()))
        meta=dict(model_id=MODEL_ID,model_revision=MODEL_REVISION,device=device,
            dimensions=512,dtype='<f4',normalized=True,frame_count=len(records),
            elapsed_seconds=round(time.perf_counter()-started,2),training=False,
            model_weights_sha256=hashlib.file_digest(open(ROOT/'models/clip/pytorch_model.bin','rb'),'sha256').hexdigest())
        con.execute('INSERT OR REPLACE INTO runs VALUES(?,?)',('embedding',json.dumps(meta)))
    # Educational export; SQLite is canonical for retrieval (IDs prevent row-order mistakes).
    np.save(ROOT/'data/frame_vectors.npy',vectors,allow_pickle=False)
    (ROOT/'data/vector_frame_ids.json').write_text(json.dumps([r['frame_id'] for r in records],indent=2))
    (ROOT/'reports/embedding_run.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
    con.close()

if __name__=='__main__':
    main()
