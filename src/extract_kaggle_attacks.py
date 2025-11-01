#!/usr/bin/env python3
import os, subprocess, pathlib

SRC = "data/raw_kaggle/monitors-replay/attacks"
DST = "data/attack"
pathlib.Path(DST).mkdir(parents=True, exist_ok=True)

for fname in os.listdir(SRC):
    if not fname.lower().endswith(('.mp4','.mov','.MOV')):
        continue
    video_path = os.path.join(SRC, fname)
    base = os.path.splitext(fname)[0]
    out_dir = os.path.join(DST, f"kaggle_{base}")
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    # extraer 1 frame por segundo (ajustable)
    cmd = [
        "ffmpeg", "-i", video_path, "-vf", "fps=3,scale=224:224",
        os.path.join(out_dir, "frame_%05d.jpg")
    ]
    print("Extracting:", video_path)
    subprocess.run(cmd, check=False)
print("Done extracting frames.")
