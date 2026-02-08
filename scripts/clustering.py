"""
Clustering methods for zero-shot species grouping.

Supported methods:
- HDBSCAN: Unsupervised, density-based, auto cluster count (RECOMMENDED)
- DBSCAN: Unsupervised, density-based, can over-cluster
- Hierarchical: Supervised, requires K clusters
- GMM: Supervised, Gaussian Mixture Model, requires K clusters
"""

from __future__ import annotations

import time
import pickle
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


# Predefined HDBSCAN configurations
HDBSCAN_CONFIGS = {
    "small": {
        "min_cluster_size": 15,
        "min_samples": 5,
        "description": "For even data, <300 samples/class"
    },
    "medium": {
        "min_cluster_size": 100,
        "min_samples": 30,
        "description": "For moderate data, mixed representation"
    },
    "large": {
        "min_cluster_size": 150,
        "min_samples": 50,
        "description": "For large/uneven data, >150 samples/class expected"
    }
}

# Predefined DBSCAN configurations
DBSCAN_CONFIGS = {
    "auto": {
        "eps_multiplier": 1.0,
        "min_samples": 5,
        "description": "Auto epsilon, may create many sub-clusters"
    },
    "relaxed": {
        "eps_multiplier": 1.5,
        "min_samples": 20,
        "description": "Larger epsilon, fewer clusters"
    },
    "very_relaxed": {
        "eps_multiplier": 2.0,
        "min_samples": 30,
        "description": "Very large epsilon, fewer clusters"
    }
}


@dataclass(frozen=True)
class ClusteringResult:
    """Holds clustering results and metrics."""
    labels: np.ndarray
    method: str
    n_clusters: int
    n_outliers: int
    metrics: Dict[str, float]
    params: Dict[str, float]
    time_seconds: float


def evaluate_clustering(
    labels_true: List[str],
    labels_pred: np.ndarray,
) -> Dict[str, float]:
    """Calculate clustering evaluation metrics."""
    encoder = LabelEncoder()
    encoded_true = encoder.fit_transform(labels_true)
    
    # Handle outliers (label = -1)
    valid_mask = labels_pred != -1
    n_outliers = np.sum(~valid_mask)
    outlier_ratio = n_outliers / len(labels_pred)
    
    if not np.any(valid_mask):
        return {
            "n_clusters": 0,
            "n_outliers": int(n_outliers),
            "outlier_ratio": outlier_ratio,
        }
    
    unique_clusters = np.unique(labels_pred[valid_mask])
    n_clusters = len(unique_clusters)
    
    metrics = {
        "n_clusters": n_clusters,
        "n_outliers": int(n_outliers),
        "outlier_ratio": outlier_ratio,
    }
    
    if n_clusters <= 1:
        return metrics
    
    # Compute metrics on valid points only
    metrics["v_measure"] = v_measure_score(encoded_true[valid_mask], labels_pred[valid_mask])
    metrics["homogeneity"] = homogeneity_score(encoded_true[valid_mask], labels_pred[valid_mask])
    metrics["completeness"] = completeness_score(encoded_true[valid_mask], labels_pred[valid_mask])
    metrics["ami"] = adjusted_mutual_info_score(encoded_true[valid_mask], labels_pred[valid_mask])
    metrics["ari"] = adjusted_rand_score(encoded_true[valid_mask], labels_pred[valid_mask])
    
    return metrics


def estimate_eps(data: np.ndarray, min_samples: int, multiplier: float = 1.0) -> float:
    """Estimate DBSCAN epsilon using k-nearest neighbors."""
    if len(data) <= min_samples:
        return 0.5
    
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(data)
    distances, _ = neighbors.kneighbors(data)
    mean_distance = float(np.mean(distances[:, -1]))
    
    return max(mean_distance * multiplier, np.finfo(float).eps)


def run_hdbscan(
    data: np.ndarray,
    labels: List[str],
    min_cluster_size: int = 15,
    min_samples: int = 5
) -> ClusteringResult:
    """Run HDBSCAN clustering."""
    if not HAS_HDBSCAN:
        raise RuntimeError("HDBSCAN not installed. Install with: pip install hdbscan")
    
    start = time.perf_counter()
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    pred_labels = clusterer.fit_predict(data)
    
    duration = time.perf_counter() - start
    
    metrics = evaluate_clustering(labels, pred_labels)
    
    return ClusteringResult(
        labels=pred_labels,
        method="hdbscan",
        n_clusters=metrics["n_clusters"],
        n_outliers=metrics["n_outliers"],
        metrics=metrics,
        params={"min_cluster_size": min_cluster_size, "min_samples": min_samples},
        time_seconds=duration
    )


def run_dbscan(
    data: np.ndarray,
    labels: List[str],
    eps_multiplier: float = 1.0,
    min_samples: int = 5
) -> ClusteringResult:
    """Run DBSCAN clustering."""
    start = time.perf_counter()
    
    eps = estimate_eps(data, min_samples, eps_multiplier)
    
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    pred_labels = clusterer.fit_predict(data)
    
    duration = time.perf_counter() - start
    
    metrics = evaluate_clustering(labels, pred_labels)
    
    return ClusteringResult(
        labels=pred_labels,
        method="dbscan",
        n_clusters=metrics["n_clusters"],
        n_outliers=metrics["n_outliers"],
        metrics=metrics,
        params={"eps": eps, "eps_multiplier": eps_multiplier, "min_samples": min_samples},
        time_seconds=duration
    )


def run_hierarchical(
    data: np.ndarray,
    labels: List[str],
    n_clusters: int
) -> ClusteringResult:
    """Run Hierarchical Agglomerative Clustering."""
    start = time.perf_counter()
    
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    pred_labels = clusterer.fit_predict(data)
    
    duration = time.perf_counter() - start
    
    metrics = evaluate_clustering(labels, pred_labels)
    
    return ClusteringResult(
        labels=pred_labels,
        method="hierarchical",
        n_clusters=n_clusters,
        n_outliers=0,
        metrics=metrics,
        params={"n_clusters": n_clusters, "linkage": "ward"},
        time_seconds=duration
    )


def run_gmm(
    data: np.ndarray,
    labels: List[str],
    n_clusters: int,
    random_state: int = 42
) -> ClusteringResult:
    """Run Gaussian Mixture Model clustering."""
    start = time.perf_counter()
    
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=random_state,
        n_init=5
    )
    pred_labels = gmm.fit_predict(data)
    
    duration = time.perf_counter() - start
    
    metrics = evaluate_clustering(labels, pred_labels)
    metrics["bic"] = gmm.bic(data)
    metrics["aic"] = gmm.aic(data)
    
    return ClusteringResult(
        labels=pred_labels,
        method="gmm",
        n_clusters=n_clusters,
        n_outliers=0,
        metrics=metrics,
        params={"n_components": n_clusters},
        time_seconds=duration
    )


def run_clustering(
    data: np.ndarray,
    labels: List[str],
    method: str,
    **kwargs
) -> ClusteringResult:
    """Run specified clustering method."""
    method = method.lower()
    
    if method == "hdbscan":
        return run_hdbscan(data, labels, **kwargs)
    elif method == "dbscan":
        return run_dbscan(data, labels, **kwargs)
    elif method == "hierarchical":
        return run_hierarchical(data, labels, **kwargs)
    elif method == "gmm":
        return run_gmm(data, labels, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def prompt_clustering_config() -> Tuple[str, Dict]:
    """Interactive prompt for clustering configuration."""
    print("\n" + "=" * 70)
    print("  CLUSTERING METHOD SELECTION")
    print("=" * 70)
    print()
    print("  UNSUPERVISED METHODS (auto-detect cluster count):")
    print("    1. HDBSCAN - Recommended for most cases")
    print("    2. DBSCAN  - May create many sub-clusters with biological meaning")
    print()
    print("  SUPERVISED METHODS (requires specifying K clusters):")
    print("    3. Hierarchical - Agglomerative with Ward linkage")
    print("    4. GMM - Gaussian Mixture Model")
    print("=" * 70)
    print()
    
    while True:
        choice = input("Select method (1-4): ").strip()
        
        if choice == "1":
            return prompt_hdbscan_config()
        elif choice == "2":
            return prompt_dbscan_config()
        elif choice == "3":
            return prompt_supervised_config("hierarchical")
        elif choice == "4":
            return prompt_supervised_config("gmm")
        else:
            print("[!] Invalid choice. Enter 1-4.")


def prompt_hdbscan_config() -> Tuple[str, Dict]:
    """Prompt for HDBSCAN configuration."""
    print("\n" + "-" * 50)
    print("  HDBSCAN CONFIGURATION")
    print("-" * 50)
    print()
    print("  Presets:")
    for key, config in HDBSCAN_CONFIGS.items():
        print(f"    {key}: min_cluster_size={config['min_cluster_size']}, "
              f"min_samples={config['min_samples']}")
        print(f"           {config['description']}")
    print("    custom: Specify your own values")
    print()
    
    while True:
        preset = input("Select preset (small/medium/large/custom): ").strip().lower()
        
        if preset in HDBSCAN_CONFIGS:
            config = HDBSCAN_CONFIGS[preset]
            return "hdbscan", {
                "min_cluster_size": config["min_cluster_size"],
                "min_samples": config["min_samples"]
            }
        elif preset == "custom":
            try:
                mcs = int(input("  min_cluster_size (default 15): ").strip() or "15")
                ms = int(input("  min_samples (default 5): ").strip() or "5")
                return "hdbscan", {"min_cluster_size": mcs, "min_samples": ms}
            except ValueError:
                print("[!] Invalid number. Try again.")
        else:
            print("[!] Invalid preset. Try again.")


def prompt_dbscan_config() -> Tuple[str, Dict]:
    """Prompt for DBSCAN configuration."""
    print("\n" + "-" * 50)
    print("  DBSCAN CONFIGURATION")
    print("-" * 50)
    print()
    print("  Note: DBSCAN often creates many small clusters that may")
    print("  contain biologically meaningful sub-groups (age, sex, etc.)")
    print()
    print("  Presets:")
    for key, config in DBSCAN_CONFIGS.items():
        print(f"    {key}: eps_multiplier={config['eps_multiplier']}, "
              f"min_samples={config['min_samples']}")
        print(f"           {config['description']}")
    print("    custom: Specify your own values")
    print()
    
    while True:
        preset = input("Select preset (auto/relaxed/very_relaxed/custom): ").strip().lower()
        
        if preset in DBSCAN_CONFIGS:
            config = DBSCAN_CONFIGS[preset]
            return "dbscan", {
                "eps_multiplier": config["eps_multiplier"],
                "min_samples": config["min_samples"]
            }
        elif preset == "custom":
            try:
                eps_mult = float(input("  eps_multiplier (default 1.0): ").strip() or "1.0")
                ms = int(input("  min_samples (default 5): ").strip() or "5")
                return "dbscan", {"eps_multiplier": eps_mult, "min_samples": ms}
            except ValueError:
                print("[!] Invalid number. Try again.")
        else:
            print("[!] Invalid preset. Try again.")


def prompt_supervised_config(method: str) -> Tuple[str, Dict]:
    """Prompt for supervised clustering configuration."""
    print("\n" + "-" * 50)
    print(f"  {method.upper()} CONFIGURATION")
    print("-" * 50)
    print()
    print("  K = number of clusters (should match expected species count)")
    print()
    
    while True:
        try:
            k = int(input("  Enter K (number of clusters): ").strip())
            if k < 2:
                print("[!] K must be at least 2")
                continue
            return method, {"n_clusters": k}
        except ValueError:
            print("[!] Invalid number. Try again.")


def save_clustering(
    result: ClusteringResult,
    labels_true: List[str],
    paths: List[str],
    embeddings_2d: np.ndarray,
    output_path: Path
) -> None:
    """Save clustering result to pickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        'cluster_labels': result.labels,
        'method': result.method,
        'n_clusters': result.n_clusters,
        'n_outliers': result.n_outliers,
        'metrics': result.metrics,
        'params': result.params,
        'time_seconds': result.time_seconds,
        'labels_true': labels_true,
        'paths': paths,
        'embeddings_2d': embeddings_2d
    }
    
    with output_path.open('wb') as f:
        pickle.dump(payload, f)


def save_cluster_images(
    cluster_labels: np.ndarray,
    image_paths: List[str],
    output_dir: Path,
    copy: bool = True
) -> Dict[int, List[Path]]:
    """
    Organize images into cluster folders.
    
    Creates a folder structure:
        output_dir/
            clusters/
                cluster_1/
                    image1.jpg
                    image2.jpg
                cluster_2/
                    ...
                rejected/  (outliers, label = -1)
                    ...
    
    Args:
        cluster_labels: Cluster assignments for each image
        image_paths: Paths to original images
        output_dir: Directory to save organized images
        copy: If True, copy images. If False, create symlinks (Unix) or shortcuts (Windows)
    
    Returns:
        Dictionary mapping cluster_id to list of copied image paths
    """
    import shutil
    
    clusters_dir = output_dir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)
    
    # Group images by cluster
    cluster_images: Dict[int, List[Path]] = {}
    
    for idx, (label, img_path) in enumerate(zip(cluster_labels, image_paths)):
        label = int(label)
        if label not in cluster_images:
            cluster_images[label] = []
        cluster_images[label].append(Path(img_path))
    
    saved_paths: Dict[int, List[Path]] = {}
    
    # Create folders and copy/link images
    for cluster_id, paths in sorted(cluster_images.items()):
        if cluster_id == -1:
            folder_name = "rejected"
        else:
            folder_name = f"cluster_{cluster_id + 1}"  # 1-indexed for user friendliness
        
        cluster_folder = clusters_dir / folder_name
        cluster_folder.mkdir(parents=True, exist_ok=True)
        
        saved_paths[cluster_id] = []
        
        for src_path in paths:
            if not src_path.exists():
                continue
            
            dst_path = cluster_folder / src_path.name
            
            # Handle duplicate filenames
            if dst_path.exists():
                stem = src_path.stem
                suffix = src_path.suffix
                counter = 1
                while dst_path.exists():
                    dst_path = cluster_folder / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            try:
                if copy:
                    shutil.copy2(src_path, dst_path)
                else:
                    # Try symlink, fall back to copy
                    try:
                        dst_path.symlink_to(src_path)
                    except (OSError, NotImplementedError):
                        shutil.copy2(src_path, dst_path)
                
                saved_paths[cluster_id].append(dst_path)
            except Exception as e:
                print(f"    Warning: Could not copy {src_path.name}: {e}")
    
    return saved_paths


def load_clustering(path: Path) -> Dict:
    """Load clustering result from pickle file."""
    with path.open('rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Apply clustering to reduced embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clustering.py --input outputs/reductions/dinov3_tsne.pkl
  python clustering.py --input outputs/reductions/dinov3_tsne.pkl --method hdbscan --preset medium
  python clustering.py --input outputs/reductions/dinov3_tsne.pkl --method hierarchical --k 30
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to reduction result pickle file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs/clustering",
        help="Output directory for clustering results"
    )
    
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["hdbscan", "dbscan", "hierarchical", "gmm"],
        help="Clustering method (interactive if not specified)"
    )
    
    parser.add_argument(
        "--preset",
        type=str,
        help="Preset configuration (for HDBSCAN: small/medium/large, for DBSCAN: auto/relaxed/very_relaxed)"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        help="Number of clusters (for hierarchical/GMM)"
    )
    
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        help="HDBSCAN min_cluster_size"
    )
    
    parser.add_argument(
        "--min-samples",
        type=int,
        help="HDBSCAN/DBSCAN min_samples"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    if not input_path.exists():
        print(f"[X] Input file not found: {input_path}")
        return
    
    # Load reduction data
    print(f"[>] Loading: {input_path}")
    with input_path.open('rb') as f:
        data = pickle.load(f)
    
    embeddings_2d = data['embeddings_2d']
    labels = data['labels']
    paths = data['paths']
    reduction_method = data['method']
    
    print(f"    {len(embeddings_2d)} samples, {len(set(labels))} classes")
    
    # Get model name from filename
    model_name = input_path.stem.split('_')[0]
    
    # Determine clustering method and config
    if args.method:
        method = args.method
        kwargs = {}
        
        if method == "hdbscan":
            if args.preset and args.preset in HDBSCAN_CONFIGS:
                kwargs = {
                    "min_cluster_size": HDBSCAN_CONFIGS[args.preset]["min_cluster_size"],
                    "min_samples": HDBSCAN_CONFIGS[args.preset]["min_samples"]
                }
            else:
                kwargs = {
                    "min_cluster_size": args.min_cluster_size or 15,
                    "min_samples": args.min_samples or 5
                }
        elif method == "dbscan":
            if args.preset and args.preset in DBSCAN_CONFIGS:
                kwargs = {
                    "eps_multiplier": DBSCAN_CONFIGS[args.preset]["eps_multiplier"],
                    "min_samples": DBSCAN_CONFIGS[args.preset]["min_samples"]
                }
            else:
                kwargs = {"min_samples": args.min_samples or 5}
        elif method in ["hierarchical", "gmm"]:
            if not args.k:
                print("[X] --k is required for hierarchical/GMM")
                return
            kwargs = {"n_clusters": args.k}
    else:
        # Interactive mode
        method, kwargs = prompt_clustering_config()
    
    # Run clustering
    print(f"\n[>] Running {method.upper()} clustering...")
    result = run_clustering(embeddings_2d, labels, method, **kwargs)
    
    # Print results
    print(f"\n" + "=" * 50)
    print(f"  CLUSTERING RESULTS")
    print("=" * 50)
    print(f"  Method: {result.method.upper()}")
    print(f"  Clusters found: {result.n_clusters}")
    print(f"  Outliers: {result.n_outliers} ({result.metrics.get('outlier_ratio', 0):.1%})")
    print(f"  Time: {result.time_seconds:.1f}s")
    print()
    if "v_measure" in result.metrics:
        print(f"  V-Measure: {result.metrics['v_measure']:.3f}")
        print(f"  Homogeneity: {result.metrics['homogeneity']:.3f}")
        print(f"  Completeness: {result.metrics['completeness']:.3f}")
        print(f"  AMI: {result.metrics['ami']:.3f}")
    print("=" * 50)
    
    # Save result
    config_str = "_".join(f"{k}{v}" for k, v in sorted(kwargs.items()))
    output_path = output_dir / f"{model_name}_{reduction_method}_{method}_{config_str}.pkl"
    
    save_clustering(result, labels, paths, embeddings_2d, output_path)
    print(f"\n[OK] Saved: {output_path}")


if __name__ == "__main__":
    main()
