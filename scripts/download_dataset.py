"""
Download and prepare HUGO-Bench dataset from HuggingFace.

This script downloads the dataset from https://huggingface.co/datasets/AI-EcoNet/HUGO-Bench
and organizes it into the expected folder structure for the pipeline.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import argparse


# Species to exclude from download (too few samples)
EXCLUDED_SPECIES = {
    "quail",  # Only 44 images
}


def check_huggingface_auth() -> bool:
    """Check if user is authenticated with HuggingFace."""
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return True
    except Exception:
        pass
    
    # Check environment variable
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    
    return False


def authenticate_huggingface() -> bool:
    """Prompt user to authenticate with HuggingFace if needed."""
    if check_huggingface_auth():
        print("[OK] HuggingFace authentication found")
        return True
    
    print("\n" + "=" * 70)
    print("  HUGGINGFACE AUTHENTICATION")
    print("=" * 70)
    print()
    print("To download the dataset, you need a HuggingFace account.")
    print()
    print("Options:")
    print("  1. Enter your HuggingFace token now")
    print("  2. Exit and set up authentication later")
    print()
    print("To get a token:")
    print("  1. Create account at: https://huggingface.co/join")
    print("  2. Go to: https://huggingface.co/settings/tokens")
    print("  3. Create a new token (read access is sufficient)")
    print("=" * 70)
    print()
    
    while True:
        choice = input("Enter choice (1/2): ").strip()
        
        if choice == "1":
            token = input("\nEnter your HuggingFace token: ").strip()
            if token:
                try:
                    from huggingface_hub import login
                    login(token=token)
                    print("[OK] Successfully authenticated with HuggingFace!")
                    return True
                except Exception as e:
                    print(f"[X] Authentication failed: {e}")
                    continue
            else:
                print("[!] No token entered")
                continue
                
        elif choice == "2":
            print("\n[*] Exiting. To authenticate later, run:")
            print("    python -c \"from huggingface_hub import login; login()\"")
            print("  Or set environment variable: HF_TOKEN=your_token_here")
            return False
            
        else:
            print("[!] Invalid choice. Please enter 1 or 2")


def get_dataset_size_estimate() -> Tuple[int, str]:
    """Return estimated dataset size for user information."""
    return 139111, "~20GB"


def prompt_download_confirmation(split: Optional[str] = None) -> bool:
    """Ask user to confirm download of large dataset."""
    total_images, size_estimate = get_dataset_size_estimate()
    
    print("\n" + "=" * 70)
    print("  HUGO-BENCH DATASET DOWNLOAD")
    print("=" * 70)
    print()
    
    if split:
        if split == "aves":
            print(f"  Split: Aves (Birds)")
            print(f"  Images: ~73,528")
            print(f"  Size: ~10GB")
        elif split == "mammals":
            print(f"  Split: Mammals")
            print(f"  Images: ~65,583")
            print(f"  Size: ~10GB")
    else:
        print(f"  Full dataset: {total_images:,} images")
        print(f"  Estimated size: {size_estimate}")
    
    print()
    print("  Source: https://huggingface.co/datasets/AI-EcoNet/HUGO-Bench")
    print("=" * 70)
    print()
    
    while True:
        choice = input("Do you want to proceed with download? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            print("[*] Download cancelled.")
            return False
        else:
            print("[!] Please enter 'y' or 'n'")


def download_dataset(
    output_dir: Path,
    split: Optional[str] = None,
    streaming: bool = False,
    max_samples: Optional[int] = None,
    skip_confirmation: bool = False
) -> bool:
    """
    Download HUGO-Bench dataset and organize into folder structure.
    
    Args:
        output_dir: Directory to save images (e.g., test_data/)
        split: 'aves', 'mammals', or None for both
        streaming: Use streaming mode (saves memory but slower)
        max_samples: Limit number of samples (for testing)
        skip_confirmation: Skip download confirmation prompt
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from datasets import load_dataset
        from PIL import Image
        from tqdm import tqdm
    except ImportError as e:
        print(f"[X] Missing required package: {e}")
        print("[*] Install with: pip install datasets pillow tqdm")
        return False
    
    # Confirm download
    if not skip_confirmation:
        if not prompt_download_confirmation(split):
            return False
    
    # Note: HUGO-Bench is a public dataset - no authentication required
    print("[OK] HUGO-Bench is a public dataset - no authentication required")
    
    print(f"\n[>] Loading dataset from HuggingFace...")
    
    try:
        # Determine splits to download
        splits_to_download = []
        if split is None or split == "all":
            splits_to_download = ["aves", "mammals"]
        elif split.lower() in ["aves", "birds"]:
            splits_to_download = ["aves"]
        elif split.lower() in ["mammals", "mammalia"]:
            splits_to_download = ["mammals"]
        else:
            print(f"[X] Unknown split: {split}")
            print("[*] Use 'aves', 'mammals', or 'all'")
            return False
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_saved = 0
        
        for current_split in splits_to_download:
            print(f"\n[>] Downloading {current_split} split...")
            
            if streaming:
                dataset = load_dataset(
                    "AI-EcoNet/HUGO-Bench",
                    split=current_split,
                    streaming=True
                )
            else:
                dataset = load_dataset(
                    "AI-EcoNet/HUGO-Bench",
                    split=current_split
                )
            
            # Create split directory
            split_dir = output_dir / current_split.capitalize()
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Track species folders created
            species_folders = set()
            
            # Process samples
            if streaming:
                iterator = iter(dataset)
                pbar = tqdm(desc=f"Saving {current_split}", unit=" images")
            else:
                iterator = iter(dataset)
                pbar = tqdm(dataset, desc=f"Saving {current_split}", unit=" images")
            
            count = 0
            for sample in pbar if not streaming else iterator:
                if streaming:
                    pbar.update(1)
                
                if max_samples and count >= max_samples:
                    break
                
                # Handle both 'image' and 'image_bytes' keys (dataset format may vary)
                if 'image' in sample:
                    image = sample['image']
                elif 'image_bytes' in sample:
                    import io
                    image = Image.open(io.BytesIO(sample['image_bytes']))
                else:
                    print(f"[!] Unknown image format. Keys: {sample.keys()}")
                    continue
                
                # Get filename - handle both 'filename' and alternative keys
                if 'filename' in sample:
                    filename = sample['filename']
                elif 'species_name' in sample:
                    # Generate filename from species name
                    filename = f"{sample['species_name']}_{count:05d}.jpg"
                else:
                    filename = f"image_{count:05d}.jpg"
                
                # Extract species from filename (e.g., "american-crow_0001.jpg" -> "american-crow")
                # Handle uncertain prefix: "uncertain_american-crow_0001.jpg"
                if filename.startswith("uncertain_"):
                    # Remove uncertain prefix for species extraction
                    name_part = filename[10:]  # Skip "uncertain_"
                else:
                    name_part = filename
                
                # Split by underscore and take all parts except the last (which is the number)
                parts = name_part.rsplit('_', 1)
                if len(parts) >= 2:
                    species = parts[0]
                else:
                    species = "unknown"
                
                # Skip excluded species
                if species.lower() in EXCLUDED_SPECIES:
                    continue
                
                # Create species folder
                species_dir = split_dir / species
                if species not in species_folders:
                    species_dir.mkdir(parents=True, exist_ok=True)
                    species_folders.add(species)
                
                # Save image
                img_path = species_dir / filename
                if isinstance(image, Image.Image):
                    image.save(img_path)
                else:
                    # Handle case where image might be bytes
                    Image.open(image).save(img_path)
                
                count += 1
                total_saved += 1
            
            pbar.close()
            print(f"[OK] Saved {count} images to {split_dir}")
            print(f"    Species folders created: {len(species_folders)}")
        
        print(f"\n[OK] Dataset download complete!")
        print(f"    Total images saved: {total_saved:,}")
        print(f"    Location: {output_dir}")
        
        # Create metadata file
        metadata = {
            "source": "https://huggingface.co/datasets/AI-EcoNet/HUGO-Bench",
            "total_images": total_saved,
            "splits": splits_to_download
        }
        
        import json
        with open(output_dir / "dataset_info.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"\n[X] Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dataset_exists(data_dir: Path) -> Tuple[bool, List[str]]:
    """
    Check if dataset already exists and return available splits.
    
    Returns:
        (exists, list_of_splits)
    """
    if not data_dir.exists():
        return False, []
    
    splits = []
    for split_name in ["Aves", "Mammals"]:
        split_dir = data_dir / split_name
        if split_dir.exists() and any(split_dir.iterdir()):
            # Count species folders
            species_count = sum(1 for d in split_dir.iterdir() if d.is_dir())
            if species_count > 0:
                splits.append(split_name.lower())
    
    return len(splits) > 0, splits


def main():
    parser = argparse.ArgumentParser(
        description="Download HUGO-Bench dataset for zero-shot clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_dataset.py                    # Download full dataset to test_data/
  python download_dataset.py --split aves      # Download only birds
  python download_dataset.py --split mammals   # Download only mammals
  python download_dataset.py --max-samples 100 # Download 100 samples per split (for testing)
  python download_dataset.py --streaming       # Use streaming mode (lower memory)
        """
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="test_data",
        help="Output directory for downloaded images (default: test_data/)"
    )
    
    parser.add_argument(
        "--split", "-s",
        type=str,
        choices=["aves", "mammals", "all"],
        default="all",
        help="Which split to download (default: all)"
    )
    
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (lower memory, slower)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples per split (for testing)"
    )
    
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompts"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if data exists"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Check if data already exists
    exists, existing_splits = check_dataset_exists(output_dir)
    
    if exists and not args.force:
        print(f"\n[*] Dataset already exists at: {output_dir}")
        print(f"    Existing splits: {', '.join(existing_splits)}")
        print(f"\n[*] Use --force to re-download")
        return
    
    # Download
    success = download_dataset(
        output_dir=output_dir,
        split=args.split if args.split != "all" else None,
        streaming=args.streaming,
        max_samples=args.max_samples,
        skip_confirmation=args.yes
    )
    
    if success:
        print("\n[OK] Dataset ready for use!")
        print(f"[*] Run the pipeline with: python main.py")
    else:
        print("\n[X] Dataset download failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
