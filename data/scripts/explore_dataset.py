import pandas as pd

df = pd.read_csv("data/echonet/processed/labels.csv")

print("="*50)
print("ECHONET PROCESSED DATA SUMMARY")
print("="*50)

print("\nTotal processed videos:")
print(len(df))

print("\nSplit distribution:")
print(df["split"].value_counts())

print("\nEF statistics:")
print(df["ef"].describe())

print("\nFrame statistics:")
print(df["n_frames"].describe())