#!/usr/bin/env python3
"""
Download GloVe embeddings from ann-benchmarks and convert to NumPy .npy format.

Usage:
    python download_glove.py [--output-dir ./data]
"""

import argparse
import urllib.request
import os
from pathlib import Path

import h5py
import numpy as np


DATASETS = {
    "glove-100": {
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "metric": "cosine",
        "dim": 100,
    },
    "glove-25": {
        "url": "http://ann-benchmarks.com/glove-25-angular.hdf5",
        "metric": "cosine",
        "dim": 25,
    },
    "sift-128": {
        "url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "metric": "l2",
        "dim": 128,
    },
}


def download_file(url: str, output_path: Path) -> None:
    """Download a file with progress indication."""
    print(f"Downloading {url}...")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
    print()


def convert_hdf5_to_npy(hdf5_path: Path, output_dir: Path, dataset_name: str) -> tuple:
    """Convert HDF5 dataset to NumPy .npy files."""
    print(f"Converting {hdf5_path} to NumPy format...")

    with h5py.File(hdf5_path, "r") as f:
        # Load train data (the main vectors)
        train_data = np.array(f["train"])
        test_data = np.array(f["test"])

        # Some datasets have ground truth neighbors
        if "neighbors" in f:
            neighbors = np.array(f["neighbors"])
        else:
            neighbors = None

        if "distances" in f:
            distances = np.array(f["distances"])
        else:
            distances = None

    print(f"  Train vectors: {train_data.shape}")
    print(f"  Test vectors: {test_data.shape}")

    # Save as .npy files
    train_npy_path = output_dir / f"{dataset_name}_train.npy"
    test_npy_path = output_dir / f"{dataset_name}_test.npy"

    np.save(train_npy_path, train_data.astype(np.float32))
    np.save(test_npy_path, test_data.astype(np.float32))

    print(f"  Saved: {train_npy_path}")
    print(f"  Saved: {test_npy_path}")

    # Generate IDs file
    ids_path = output_dir / f"{dataset_name}_ids.txt"
    with open(ids_path, "w") as f:
        for i in range(train_data.shape[0]):
            f.write(f"vec_{i}\n")
    print(f"  Saved: {ids_path}")

    return train_npy_path, test_npy_path, ids_path


def main():
    parser = argparse.ArgumentParser(description="Download ANN-benchmarks datasets")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="glove-100",
                       help="Dataset to download (default: glove-100)")
    parser.add_argument("--output-dir", type=Path, default=Path("./data"),
                       help="Output directory (default: ./data)")
    parser.add_argument("--keep-hdf5", action="store_true",
                       help="Keep the HDF5 file after conversion")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = DATASETS[args.dataset]
    hdf5_path = args.output_dir / f"{args.dataset}.hdf5"

    # Download if not exists
    if not hdf5_path.exists():
        download_file(dataset_info["url"], hdf5_path)
    else:
        print(f"Using existing file: {hdf5_path}")

    # Convert to NumPy format
    train_path, test_path, ids_path = convert_hdf5_to_npy(
        hdf5_path, args.output_dir, args.dataset
    )

    # Remove HDF5 file unless --keep-hdf5
    if not args.keep_hdf5:
        os.remove(hdf5_path)
        print(f"  Removed: {hdf5_path}")

    # Print usage instructions
    print("\n" + "=" * 60)
    print("Dataset ready!")
    print("=" * 60)
    print(f"\nTo run the real-recall-bench:")
    print(f"""
./target/release/real-recall-bench \\
    --embeddings-file {train_path} \\
    --ids-file {ids_path} \\
    --collection-name {args.dataset.replace('-', '_')} \\
    --metric {dataset_info['metric']} \\
    --top-k 10 \\
    --num-queries 500 \\
    --nprobe 4,8,16,32,64 \\
    --output recall_{args.dataset}.csv
""")


if __name__ == "__main__":
    main()
