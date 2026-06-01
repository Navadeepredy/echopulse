import os
import pandas as pd

ROOT = "data/echonet/raw/EchoNet-Dynamic"

filelist_path = os.path.join(ROOT, "FileList.csv")
tracing_path = os.path.join(ROOT, "VolumeTracings.csv")

filelist = pd.read_csv(filelist_path)
tracings = pd.read_csv(tracing_path)

print("=" * 50)
print("ECHONET DATASET SUMMARY")
print("=" * 50)

print(f"Total videos: {len(filelist)}")
print(f"Total tracing rows: {len(tracings)}")

print("\nSplit distribution:")
print(filelist["Split"].value_counts())

print("\nColumns in FileList.csv:")
print(filelist.columns.tolist())

print("\nColumns in VolumeTracings.csv:")
print(tracings.columns.tolist())