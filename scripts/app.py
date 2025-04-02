import os
from glob import glob
import random


class AudioDataset:
    def __init__(self, base_path="data/train"):
        self.base_path = base_path
        self.bonafide = self._load_files("bonafide")
        self.spoof = self._load_files("spoof")

    def _load_files(self, folder):
        return glob(os.path.join(self.base_path, folder, "*.wav")) + \
            glob(os.path.join(self.base_path, folder, "*.flac"))

    def get_random_pair(self):
        return {
            "real": random.choice(self.bonafide),
            "fake": random.choice(self.spoof)
        }

    def stats(self):
        return {
            "total_real": len(self.bonafide),
            "total_fake": len(self.spoof),
            "ratio": f"{len(self.spoof) / len(self.bonafide):.2f}:1"
        }


if __name__ == "__main__":
    dataset = AudioDataset()
    print(dataset.stats())
    print("Sample pair:", dataset.get_random_pair())