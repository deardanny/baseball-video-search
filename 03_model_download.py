"""Download only the pinned CLIP model, tokenizer and preprocessing files."""
import os
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parent
MODEL_ID='openai/clip-vit-base-patch32'
MODEL_REVISION='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
if __name__=='__main__':
    snapshot_download(MODEL_ID,revision=MODEL_REVISION,local_dir=ROOT/'models/clip',
        allow_patterns=['*.json','merges.txt','vocab.json','pytorch_model.bin','README.md'],max_workers=3)
    print('Pinned CLIP files downloaded. No training performed.')
