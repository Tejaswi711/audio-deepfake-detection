import os
import shutil


def organize_dataset():
    base_path = os.path.normpath("data")

    # For both train and dev sets
    for dataset in ['train', 'dev']:
        # Create target directories
        os.makedirs(os.path.join(base_path, dataset, "bonafide"), exist_ok=True)
        os.makedirs(os.path.join(base_path, dataset, "spoof"), exist_ok=True)

        # Get protocol file
        protocol_file = os.path.join(base_path, f"ASVspoof2019.LA.cm.{dataset}.trn.txt")

        if not os.path.exists(protocol_file):
            print(f"Missing protocol file: {protocol_file}")
            continue

        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:  # Skip malformed lines
                    continue

                file_id = parts[0]
                label = parts[-1]  # Last column is 'bonafide' or 'spoof'

                # Original file location (ASVspoof default structure)
                src = os.path.join(base_path, f"ASVspoof2019_LA_{dataset}", "flac", f"{file_id}.flac")

                # New location
                dst = os.path.join(base_path, dataset, label, f"{file_id}.flac")

                if os.path.exists(src):
                    shutil.move(src, dst)
                else:
                    print(f"Missing: {src}")


if __name__ == "__main__":
    organize_dataset()
    print("Organization complete. Verify files in data/train/ and data/dev/")