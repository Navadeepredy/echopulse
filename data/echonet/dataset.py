import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class EchoNetDataset(Dataset):
    def __init__(self, root, split="TRAIN", transform=None):
        self.root = root
        self.transform = transform

        csv_path = os.path.join(root, "FileList.csv")
        self.df = pd.read_csv(csv_path)

        self.df = self.df[self.df["Split"] == split].reset_index(drop=True)
        self.video_dir = os.path.join(root, "Videos")

    def __len__(self):
        return len(self.df)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)

        cap.release()

        frames = np.stack(frames)
        frames = torch.from_numpy(frames).float() / 255.0
        return frames

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_name = row["FileName"]
        label = torch.tensor(row["EF"], dtype=torch.float32)

        video_path = os.path.join(self.video_dir, video_name + ".avi")
        video = self.load_video(video_path)

        if self.transform:
            video = self.transform(video)

        return video, label