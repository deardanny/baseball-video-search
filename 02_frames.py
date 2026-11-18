"""Step 2: decode GIF timelines, sample 8 frames, preserve provenance in SQLite."""
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parent
DB = ROOT / 'data/baseball.sqlite'

def decode_gif(path):
    frames, timestamps, durations = [], [], []
    with Image.open(path) as image:
        elapsed = 0
        for index in range(image.n_frames):
            image.seek(index)
            duration = int(image.info.get('duration', 0))
            if duration <= 0:
                raise ValueError(f'Missing/nonpositive frame duration: {path}, frame {index}')
            frames.append(image.convert('RGB').copy())
            timestamps.append(elapsed)
            durations.append(duration)
            elapsed += duration
    if len(frames) < 2:
        raise ValueError('Expected an animated clip with at least 2 frames')
    return frames, np.array(timestamps), np.array(durations)

def sample_indices(timestamps, durations, count=8):
    total = timestamps[-1] + durations[-1]
    targets = (np.arange(count) + 0.5) * total / count
    return np.unique(np.searchsorted(timestamps, targets, side='right') - 1)

def main():
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    (ROOT/'data/frames').mkdir(exist_ok=True)
    (ROOT/'reports').mkdir(exist_ok=True)
    tmp = DB.with_suffix('.building.sqlite')
    if tmp.exists():
        tmp.unlink()  # Only an interrupted derived database from this project.
    con = sqlite3.connect(tmp)
    con.executescript('''
    PRAGMA foreign_keys=ON;
    CREATE TABLE clips(clip_id TEXT PRIMARY KEY, source_category TEXT, raw_path TEXT,
      source_url TEXT, sha256 TEXT UNIQUE, duration_ms INTEGER CHECK(duration_ms>0),
      source_frame_count INTEGER, width INTEGER, height INTEGER, provenance_json TEXT);
    CREATE TABLE frames(frame_id TEXT PRIMARY KEY, clip_id TEXT REFERENCES clips,
      source_frame_index INTEGER, timestamp_ms INTEGER CHECK(timestamp_ms>=0),
      relative_path TEXT UNIQUE, pixel_sha256 TEXT, mean_brightness REAL,
      UNIQUE(clip_id,source_frame_index));
    CREATE TABLE embeddings(frame_id TEXT PRIMARY KEY REFERENCES frames,
      model_id TEXT, model_revision TEXT, dimensions INTEGER,
      dtype TEXT, l2_norm REAL, vector BLOB);
    CREATE TABLE runs(name TEXT PRIMARY KEY, metadata_json TEXT);
    ''')
    quality, rows, seen_hashes = [], [], {}
    for clip in manifest:
        path = ROOT/clip['relative_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != clip['sha256']:
            raise ValueError(f'Raw file changed: {path}')
        frames, timestamps, durations = decode_gif(path)
        duration = int(timestamps[-1]+durations[-1])
        indices = sample_indices(timestamps, durations)
        con.execute('INSERT INTO clips VALUES(?,?,?,?,?,?,?,?,?,?)',
            (clip['clip_id'],clip['source_category'],clip['relative_path'],clip['source_url'],
             clip['sha256'],duration,len(frames),*frames[0].size,json.dumps(clip)))
        selected = []
        for index in indices:
            frame = frames[index]
            fid = f'{clip["clip_id"]}_{int(index):04d}'
            out = ROOT/'data/frames'/f'{fid}.jpg'
            frame.save(out, quality=95)
            pixel_hash = hashlib.sha256(frame.tobytes()).hexdigest()
            brightness = float(np.asarray(frame).mean())
            if pixel_hash in seen_hashes:
                quality.append(dict(type='exact_duplicate_sampled_frame',frame=fid,other=seen_hashes[pixel_hash]))
            seen_hashes[pixel_hash] = fid
            if brightness < 5:
                quality.append(dict(type='near_black_candidate',frame=fid,mean_brightness=brightness))
            con.execute('INSERT INTO frames VALUES(?,?,?,?,?,?,?)',
                (fid,clip['clip_id'],int(index),int(timestamps[index]),str(out.relative_to(ROOT)),pixel_hash,brightness))
            selected.append((frame, int(timestamps[index])))
        rows.append((clip['clip_id'], duration, selected))
        print(f'{clip["clip_id"]}: {len(frames)} source frames / {duration/1000:.2f}s -> {len(indices)} samples',flush=True)
    con.execute('INSERT INTO runs VALUES(?,?)',('frame_extraction',json.dumps(dict(samples_per_clip=8,sampling='8 equal-duration bin midpoints; select containing GIF frame'))))
    assert not con.execute('PRAGMA foreign_key_check').fetchall()
    assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    summary = dict(clips=len(rows),frames=con.execute('SELECT COUNT(*) FROM frames').fetchone()[0],
        total_source_duration_seconds=sum(r[1] for r in rows)/1000,
        min_duration_seconds=min(r[1] for r in rows)/1000,max_duration_seconds=max(r[1] for r in rows)/1000,
        source_bytes=sum(c['bytes'] for c in manifest),integrity='ok',findings=quality,
        caveat='No exhaustive action labels or original-game IDs/timestamps. GIF playback time is not guaranteed original realtime.')
    con.commit()
    con.close()
    tmp.replace(DB)
    (ROOT/'reports/data_quality.json').write_text(json.dumps(summary,indent=2))
    # Two contact sheets, 8 clips each; all 8 samples stay visible for review.
    for page in range(2):
        sheet = Image.new('RGB',(1280,8*145),(19,27,37))
        draw = ImageDraw.Draw(sheet)
        for row,(cid,duration,selected) in enumerate(rows[page*8:(page+1)*8]):
            y=row*145
            draw.text((8,y+3),f'{cid} | {duration/1000:.2f}s | left to right = time',fill='white')
            for column,(frame,timestamp) in enumerate(selected):
                thumb = ImageOps.contain(frame,(156,104))
                x=column*160
                sheet.paste(thumb,(x+(156-thumb.width)//2,y+23))
                draw.text((x+3,y+128),f'{timestamp/1000:.2f}s',fill='#9fb5ca')
        sheet.save(ROOT/f'reports/contact_sheet_{page+1}.jpg')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
