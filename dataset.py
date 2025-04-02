import os
import torchaudio
import torch
from torch.utils.data import Dataset


class ASVSpoofDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = os.path.normpath(root_dir)  # Normalize path
        self.transform = transform
        self.file_list = []
        self.labels = []

        print(f"\nSearching in: {self.root_dir}")  # Debug

        # Check both possible directory structures
        bonafide_paths = [
            os.path.join(self.root_dir, 'bonafide'),
            os.path.join(self.root_dir, 'LA', 'bonafide')  # ASVspoof default
        ]

        for path in bonafide_paths:
            if os.path.exists(path):
                print(f"Found bonafide directory: {path}")
                self._load_files(path, 0)

        for path in [p.replace('bonafide', 'spoof') for p in bonafide_paths]:
            if os.path.exists(path):
                print(f"Found spoof directory: {path}")
                self._load_files(path, 1)

        if not self.file_list:
            available = [f for f in os.listdir(self.root_dir) if f.endswith('.flac')]
            raise RuntimeError(
                f"No valid audio files found in {self.root_dir}\n"
                f"Available files: {available[:5]}... (total: {len(available)})"
            )

        print(f"Successfully loaded {len(self.file_list)} samples")

    def _load_files(self, path, label):
        for file in os.listdir(path):
            if file.lower().endswith(('.flac', '.wav', '.mp3')):
                self.file_list.append(os.path.join(path, file))
                self.labels.append(label)