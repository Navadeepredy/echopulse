import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW = Path("data/echonet/raw/EchoNet-Dynamic")
OUT = Path("data/echonet/processed")
OUT.mkdir(parents=True, exist_ok=True)

def load_video(path, h=112, w=112):
    cap = cv2.VideoCapture(str(path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (w, h))
        frames.append(resized)

    cap.release()

    if len(frames) == 0:
        return None

    return np.array(frames, dtype=np.float32)

def zscore(video):
    return (video - video.mean()) / (video.std() + 1e-8)

df = pd.read_csv(RAW / "FileList.csv")
records = []
failed = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    video_id = row["FileName"]
    video_path = RAW / "Videos" / f"{video_id}.avi"
    out_path = OUT / f"{video_id}.npy"

    if out_path.exists() and out_path.stat().st_size > 0:
        records.append({
            "patient_id": video_id,
            "ef": row["EF"],
            "esv": row["ESV"],
            "edv": row["EDV"],
            "split": row["Split"],
            "n_frames": row["NumberOfFrames"],
            "out_path": str(out_path)
        })
        continue

    try:
        if not video_path.exists():
            failed.append(video_id)
            continue

        video = load_video(video_path)

        if video is None:
            failed.append(video_id)
            continue

        video = zscore(video).astype(np.float16)
        np.save(out_path, video)

        records.append({
            "patient_id": video_id,
            "ef": row["EF"],
            "esv": row["ESV"],
            "edv": row["EDV"],
            "split": row["Split"],
            "n_frames": video.shape[0],
            "out_path": str(out_path)
        })

    except Exception as e:
        print(f"Failed: {video_id} | Error: {e}")
        failed.append(video_id)

pd.DataFrame(records).to_csv(OUT / "labels.csv", index=False)
pd.DataFrame({"failed_video": failed}).to_csv(OUT / "failed_files.csv", index=False)

print("Preprocessing completed.")
print(f"Processed records: {len(records)}")
print(f"Failed files: {len(failed)}")
print(f"Labels saved to: {OUT / 'labels.csv'}")