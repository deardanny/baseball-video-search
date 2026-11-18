"""Step 1: download 16 small public research examples, pin and verify their content."""
from pathlib import Path
import hashlib
import json
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
REV = '7187fbc1801daab9141d5b6a9ad4b679a96b15bf'
REPO = 'piergiaj/mlb-youtube'
CATEGORIES = ['ball', 'bunt', 'foul', 'hbp', 'hit', 'neg', 'strike', 'swing']

def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'baseball-learning-project'}), timeout=45) as response:
        return response.read()

def main():
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    listing_url = f'https://api.github.com/repos/{REPO}/contents/examples?ref={REV}'
    listing = json.loads(fetch(listing_url))
    lookup = {item['name']: item for item in listing}
    manifest = []
    for category in CATEGORIES:
        for number in [1, 2]:
            name = f'{category}{number}.gif'
            item = lookup[name]
            url = f'https://raw.githubusercontent.com/{REPO}/{REV}/examples/{name}'
            path = raw / name
            payload = path.read_bytes() if path.exists() else fetch(url)
            git_sha = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
            if git_sha != item['sha']:
                raise ValueError(f'Git blob hash mismatch for {name}')
            if not path.exists():
                path.write_bytes(payload)
            manifest.append(dict(clip_id=path.stem, source_category=category,
                source_url=url, source_page=f'https://github.com/{REPO}/blob/{REV}/examples/{name}',
                repository_revision=REV, git_blob_sha1=git_sha,
                sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
                relative_path=str(path.relative_to(ROOT)),
                label_basis='Repository example filename/category; NOT exhaustive action labels.',
                retrieved_at_utc=datetime.now(timezone.utc).isoformat()))
            print(f'{name}: {len(payload):,} bytes, hash verified', flush=True)
    (ROOT / 'data/manifest.json').write_text(json.dumps(manifest, indent=2))
    (ROOT / 'data/source_listing.json').write_text(json.dumps(listing, indent=2))
    print(f'Downloaded/verified {len(manifest)} clips, {sum(x["bytes"] for x in manifest)/1e6:.2f} MB')

if __name__ == '__main__':
    main()
