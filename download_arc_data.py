"""
Utility script to download ARC (Abstraction and Reasoning Corpus) dataset
Downloads the official dataset from GitHub repository
"""

import os
import json
import urllib.request
import urllib.error
import time
from pathlib import Path
from typing import List, Dict


def download_file(url: str, destination: str, max_retries: int = 3) -> bool:
    """Download a file with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"  Downloading: {os.path.basename(destination)}...", end=" ")
            urllib.request.urlretrieve(url, destination)
            print("✓")
            return True
        except urllib.error.URLError as e:
            print(f"✗ (Attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"  Failed to download: {e}")
                return False
    return False


def get_file_list_from_github(owner: str, repo: str, path: str, branch: str = "master") -> List[str]:
    """Get list of files from a GitHub directory using the API."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"

    try:
        with urllib.request.urlopen(api_url) as response:
            data = json.loads(response.read().decode())

        files = []
        for item in data:
            if item['type'] == 'file' and item['name'].endswith('.json'):
                files.append(item['download_url'])

        return files

    except Exception as e:
        print(f"Error accessing GitHub API: {e}")
        return []


def download_arc_dataset(output_dir: str = "./arc_data") -> bool:
    """
    Download the complete ARC dataset from the official GitHub repository.

    Args:
        output_dir: Directory where the dataset will be saved

    Returns:
        True if successful, False otherwise
    """

    print("=" * 80)
    print("ARC Dataset Downloader")
    print("=" * 80)
    print(f"\nDownloading to: {output_dir}")

    # GitHub repository details
    OWNER = "fchollet"
    REPO = "ARC-AGI"
    BRANCH = "master"

    datasets = {
        "training": "data/training",
        "evaluation": "data/evaluation",
        "test": "data/test"
    }

    # Create directories
    for dataset_name in datasets.keys():
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

    total_downloaded = 0
    total_failed = 0

    # Download each dataset
    for dataset_name, github_path in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"Downloading {dataset_name} dataset...")
        print('=' * 80)

        dataset_dir = os.path.join(output_dir, dataset_name)

        # Get list of files from GitHub
        print("Fetching file list from GitHub API...")
        file_urls = get_file_list_from_github(OWNER, REPO, github_path, BRANCH)

        if not file_urls:
            print(f"Warning: Could not fetch file list for {dataset_name}")
            print("This might be due to GitHub API rate limiting.")
            print(f"\nAlternative: Please manually download from:")
            print(f"https://github.com/{OWNER}/{REPO}/tree/{BRANCH}/{github_path}")
            continue

        print(f"Found {len(file_urls)} files")

        # Download each file
        for file_url in file_urls:
            filename = os.path.basename(file_url.split('?')[0])
            destination = os.path.join(dataset_dir, filename)

            if os.path.exists(destination):
                print(f"  Skipping: {filename} (already exists)")
                continue

            if download_file(file_url, destination):
                total_downloaded += 1
            else:
                total_failed += 1

            # Be nice to GitHub's servers
            time.sleep(0.1)

    print(f"\n{'=' * 80}")
    print("Download Summary")
    print('=' * 80)
    print(f"Successfully downloaded: {total_downloaded} files")
    print(f"Failed: {total_failed} files")

    if total_failed > 0:
        print("\nSome files failed to download. You can:")
        print("1. Run this script again to retry")
        print("2. Manually download from: https://github.com/fchollet/ARC-AGI")

    print(f"\nDataset location: {os.path.abspath(output_dir)}")
    print("=" * 80)

    return total_failed == 0


def verify_dataset(data_dir: str = "./arc_data") -> Dict[str, int]:
    """
    Verify the downloaded dataset.

    Args:
        data_dir: Directory containing the dataset

    Returns:
        Dictionary with counts of files in each split
    """
    print("\nVerifying dataset...")

    splits = ["training", "evaluation", "test"]
    counts = {}

    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            json_files = list(Path(split_dir).glob("*.json"))
            counts[split] = len(json_files)
            print(f"  {split}: {len(json_files)} files")
        else:
            counts[split] = 0
            print(f"  {split}: directory not found")

    return counts


def download_with_alternative_method(output_dir: str = "./arc_data") -> bool:
    """
    Alternative download method using direct file URLs.
    Uses a curated list of known file names.
    """
    print("\nUsing alternative download method...")

    # Base URL for raw GitHub content
    base_url = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data"

    # Known file IDs (partial list - would need complete list for production)
    # For demonstration, we'll try a few known task IDs
    example_tasks = [
        "007bbfb7", "00d62c1b", "025d127b", "0520fde7", "05f2a901",
        "06df4c85", "08ed6ac7", "09629e4f", "0a938d79", "0b148d64"
    ]

    datasets = ["training", "evaluation"]

    for dataset in datasets:
        print(f"\nDownloading {dataset} examples...")
        dataset_dir = os.path.join(output_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        for task_id in example_tasks:
            filename = f"{task_id}.json"
            url = f"{base_url}/{dataset}/{filename}"
            destination = os.path.join(dataset_dir, filename)

            if not os.path.exists(destination):
                download_file(url, destination)

    print("\nNote: This method only downloads a subset of tasks.")
    print("For the complete dataset, please visit:")
    print("https://github.com/fchollet/ARC-AGI")

    return True


def create_sample_tasks(output_dir: str = "./arc_data") -> None:
    """Create sample ARC tasks for testing."""

    print("\nCreating sample tasks for testing...")

    sample_dir = os.path.join(output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # Sample 1: Simple color replacement
    sample1 = {
        "train": [
            {
                "input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                "output": [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
            },
            {
                "input": [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
                "output": [[2, 2, 0], [2, 0, 0], [0, 0, 0]]
            }
        ],
        "test": [
            {
                "input": [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
            }
        ]
    }

    # Sample 2: Rotation
    sample2 = {
        "train": [
            {
                "input": [[1, 0], [0, 0]],
                "output": [[0, 1], [0, 0]]
            },
            {
                "input": [[0, 1], [0, 0]],
                "output": [[0, 0], [0, 1]]
            }
        ],
        "test": [
            {
                "input": [[1, 1], [0, 0]]
            }
        ]
    }

    # Sample 3: Pattern extension
    sample3 = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]
            }
        ],
        "test": [
            {
                "input": [[5, 6], [7, 8]]
            }
        ]
    }

    samples = [
        ("sample_color_replacement.json", sample1),
        ("sample_rotation.json", sample2),
        ("sample_pattern_extension.json", sample3)
    ]

    for filename, task in samples:
        filepath = os.path.join(sample_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(task, f, indent=2)
        print(f"  Created: {filename}")

    print(f"\nSample tasks saved to: {sample_dir}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Download ARC dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./arc_data",
        help="Output directory for dataset (default: ./arc_data)"
    )
    parser.add_argument(
        "--samples-only",
        action="store_true",
        help="Only create sample tasks (for testing)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing dataset"
    )

    args = parser.parse_args()

    if args.verify_only:
        verify_dataset(args.output_dir)
        return

    if args.samples_only:
        create_sample_tasks(args.output_dir)
        return

    # Try main download method
    success = download_arc_dataset(args.output_dir)

    # Verify what was downloaded
    counts = verify_dataset(args.output_dir)

    # If main method didn't work well, suggest alternatives
    if sum(counts.values()) < 10:
        print("\n" + "=" * 80)
        print("Manual Download Instructions")
        print("=" * 80)
        print("\nTo get the complete ARC dataset:")
        print("1. Visit: https://github.com/fchollet/ARC-AGI")
        print("2. Clone the repository:")
        print("   git clone https://github.com/fchollet/ARC-AGI.git")
        print("3. Copy the data directory:")
        print(f"   cp -r ARC-AGI/data/* {args.output_dir}/")
        print("\nAlternatively, download the ZIP file:")
        print("   https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip")

    # Create sample tasks regardless
    create_sample_tasks(args.output_dir)

    print("\n✓ Setup complete!")


if __name__ == "__main__":
    main()
