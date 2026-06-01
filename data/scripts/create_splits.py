import os
import pandas as pd

ROOT = "data/echonet/raw/EchoNet-Dynamic"
OUT_DIR = "data/echonet/splits"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(ROOT, "FileList.csv"))

for split_name in ["TRAIN", "VAL", "TEST"]:
    split_df = df[df["Split"] == split_name]
    out_path = os.path.join(OUT_DIR, split_name.lower() + ".csv")
    split_df.to_csv(out_path, index=False)
    print(f"{split_name}: {len(split_df)} records saved to {out_path}")
