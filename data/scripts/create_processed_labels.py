import pandas as pd
from pathlib import Path

RAW_CSV = Path("data/echonet/raw/EchoNet-Dynamic/FileList.csv")
PROCESSED = Path("data/echonet/processed")

df = pd.read_csv(RAW_CSV)
records = []

for _, row in df.iterrows():
    video_id = row["FileName"]
    npy_path = PROCESSED / f"{video_id}.npy"

    if npy_path.exists() and npy_path.stat().st_size > 0:
        records.append({
            "patient_id": video_id,
            "ef": row["EF"],
            "esv": row["ESV"],
            "edv": row["EDV"],
            "split": row["Split"],
            "n_frames": row["NumberOfFrames"],
            "out_path": str(npy_path)
        })

out_csv = PROCESSED / "labels.csv"
pd.DataFrame(records).to_csv(out_csv, index=False)

print(f"Available processed files: {len(records)}")
print(f"Labels saved to: {out_csv}")