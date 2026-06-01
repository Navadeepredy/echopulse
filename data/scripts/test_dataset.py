import sys
import os

sys.path.append(os.getcwd())

from data.echonet.dataset import EchoNetDataset

dataset = EchoNetDataset(
    root="data/echonet/raw/EchoNet-Dynamic",
    split="TRAIN"
)

print("Dataset size:", len(dataset))

video, label = dataset[0]

print("Video shape:", video.shape)
print("EF label:", label)