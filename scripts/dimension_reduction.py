"""
Dimension reduction methods for embedding visualization and clustering.

Supported methods:
- t-SNE: Best for visualization, emphasizes local structure
- UMAP: Fast, preserves both local and global structure
- PCA: Linear baseline, fast
- Isomap: Non-linear, preserves geodesic distances
- Kernel PCA: Non-linear with RBF kernel
"""

from __future__ import annotations

import time
import pickle
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# Available methods
AVAILABLE_METHODS = {
    "tsne": {
        "name": "t-SNE",
        "description": "t-Distributed Stochastic Neighbor Embedding - Best for visualization",
        "deterministic": False
    },
    "umap": {
        "name": "UMAP",
        "description": "Uniform Manifold Approximation and Projection - Fast, good structure preservation",
        "deterministic": False
    },
    "pca": {
        "name": "PCA",
        "description": "Principal Component Analysis - Linear baseline",
        "deterministic": True
    },
    "isomap": {
        "name": "Isomap",
        "description": "Isometric Feature Mapping - Preserves geodesic distances",
        "deterministic": True
    },
    "kernel_pca": {
        "name": "Kernel PCA",
        "description": "Kernel PCA with RBF kernel - Non-linear",
        "deterministic": True
    }
}


@dataclass(frozen=True)
class ReductionResult:
    """Holds reduced coordinates plus metadata."""
    embeddings: np.ndarray
    method: str
    time_seconds: float
    params: Dict[str, float]


class DimensionReducer:
    """Runs supported reduction methods on embeddings."""
    
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._scaler = StandardScaler()
    
    def _preprocess(self, embeddings: np.ndarray) -> np.ndarray:
        return self._scaler.fit_transform(embeddings)
    
    def tsne(self, embeddings: np.ndarray) -> ReductionResult:
        """t-SNE - emphasizes local neighborhood structure."""
        start = time.perf_counter()
        
        perplexity = min(30, max(5, len(embeddings) // 50))
        
        reducer = TSNE(
            n_components=self.n_components,
            random_state=self.random_state,
            learning_rate="auto",
            init="pca",
            perplexity=perplexity,
        )
        projected = reducer.fit_transform(self._preprocess(embeddings))
        duration = time.perf_counter() - start
        
        return ReductionResult(
            embeddings=projected,
            method="tsne",
            time_seconds=duration,
            params={"perplexity": perplexity}
        )
    
    def umap(self, embeddings: np.ndarray) -> ReductionResult:
        """UMAP - fast, preserves local and global structure."""
        if not HAS_UMAP:
            raise RuntimeError("UMAP is not installed. Install with: pip install umap-learn")
        
        start = time.perf_counter()
        
        n_neighbors = min(15, max(5, len(embeddings) // 100))
        
        reducer = UMAP(
            n_components=self.n_components,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="euclidean",
            random_state=self.random_state,
        )
        projected = reducer.fit_transform(self._preprocess(embeddings))
        duration = time.perf_counter() - start
        
        return ReductionResult(
            embeddings=projected,
            method="umap",
            time_seconds=duration,
            params={"n_neighbors": n_neighbors, "min_dist": 0.1}
        )
    
    def pca(self, embeddings: np.ndarray) -> ReductionResult:
        """PCA - linear dimensionality reduction."""
        start = time.perf_counter()
        
        reducer = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        projected = reducer.fit_transform(self._preprocess(embeddings))
        duration = time.perf_counter() - start
        
        return ReductionResult(
            embeddings=projected,
            method="pca",
            time_seconds=duration,
            params={"explained_variance": float(reducer.explained_variance_ratio_.sum())}
        )
    
    def isomap(self, embeddings: np.ndarray) -> ReductionResult:
        """Isomap - non-linear, preserves geodesic distances."""
        start = time.perf_counter()
        
        n_neighbors = min(15, max(5, len(embeddings) // 100))
        
        reducer = Isomap(
            n_components=self.n_components,
            n_neighbors=n_neighbors,
            metric="euclidean",
        )
        projected = reducer.fit_transform(self._preprocess(embeddings))
        duration = time.perf_counter() - start
        
        return ReductionResult(
            embeddings=projected,
            method="isomap",
            time_seconds=duration,
            params={"n_neighbors": n_neighbors}
        )
    
    def kernel_pca(self, embeddings: np.ndarray) -> ReductionResult:
        """Kernel PCA with RBF kernel."""
        start = time.perf_counter()
        
        reducer = KernelPCA(
            n_components=self.n_components,
            kernel="rbf",
            random_state=self.random_state,
            n_jobs=-1,
        )
        projected = reducer.fit_transform(self._preprocess(embeddings))
        duration = time.perf_counter() - start
        
        return ReductionResult(
            embeddings=projected,
            method="kernel_pca",
            time_seconds=duration,
            params={"kernel": "rbf"}
        )
    
    def reduce(self, embeddings: np.ndarray, method: str) -> ReductionResult:
        """Run a specific reduction method."""
        method = method.lower()
        
        if method == "tsne":
            return self.tsne(embeddings)
        elif method == "umap":
            return self.umap(embeddings)
        elif method == "pca":
            return self.pca(embeddings)
        elif method == "isomap":
            return self.isomap(embeddings)
        elif method in ["kernel_pca", "kpca"]:
            return self.kernel_pca(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}. Available: {list(AVAILABLE_METHODS.keys())}")


def run_reduction(
    embeddings: np.ndarray,
    methods: List[str],
    n_components: int = 2,
    random_state: int = 42
) -> Dict[str, ReductionResult]:
    """
    Run dimension reduction with specified methods.
    
    Args:
        embeddings: High-dimensional embeddings (N x D)
        methods: List of method names to run
        n_components: Target dimensionality (default: 2)
        random_state: Random seed
    
    Returns:
        Dict mapping method names to ReductionResult
    """
    reducer = DimensionReducer(n_components=n_components, random_state=random_state)
    results = {}
    
    for method in methods:
        print(f"    [>] {method.upper()}...", end=" ", flush=True)
        try:
            result = reducer.reduce(embeddings, method)
            results[method] = result
            print(f"[OK] ({result.time_seconds:.1f}s)")
        except Exception as e:
            print(f"[X] {e}")
    
    return results


def save_reduction(
    result: ReductionResult,
    labels: List[str],
    paths: List[str],
    output_path: Path
) -> None:
    """Save reduction result to pickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        'embeddings_2d': result.embeddings,
        'method': result.method,
        'time_seconds': result.time_seconds,
        'params': result.params,
        'labels': labels,
        'paths': paths
    }
    
    with output_path.open('wb') as f:
        pickle.dump(payload, f)


def load_reduction(path: Path) -> Dict:
    """Load reduction result from pickle file."""
    with path.open('rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Apply dimension reduction to embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dimension_reduction.py --embeddings outputs/embeddings/dinov3_embeddings.pkl
  python dimension_reduction.py --embeddings outputs/embeddings/dinov3_embeddings.pkl --methods tsne umap
  python dimension_reduction.py --embeddings outputs/embeddings/*.pkl --methods tsne
        """
    )
    
    parser.add_argument(
        "--embeddings", "-e",
        type=str,
        required=True,
        help="Path to embeddings pickle file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs/reductions",
        help="Output directory for reduction results (default: outputs/reductions)"
    )
    
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        type=str,
        default=["tsne", "umap"],
        choices=list(AVAILABLE_METHODS.keys()),
        help="Reduction methods to use (default: tsne umap)"
    )
    
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Target dimensionality (default: 2)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-computation even if results exist"
    )
    
    args = parser.parse_args()
    
    embeddings_path = Path(args.embeddings)
    output_dir = Path(args.output_dir)
    
    if not embeddings_path.exists():
        print(f"[X] Embeddings file not found: {embeddings_path}")
        return
    
    # Load embeddings
    print(f"[>] Loading embeddings from: {embeddings_path}")
    with embeddings_path.open('rb') as f:
        data = pickle.load(f)
    
    embeddings = np.asarray(data['embeddings'])
    labels = data['labels']
    paths = data['paths']
    
    print(f"    {len(embeddings)} samples, {embeddings.shape[1]}D")
    
    # Get model name from filename
    model_name = embeddings_path.stem.replace('_embeddings', '')
    
    print(f"\n[>] Running dimension reduction...")
    results = run_reduction(
        embeddings=embeddings,
        methods=args.methods,
        n_components=args.n_components,
        random_state=args.seed
    )
    
    # Save results
    for method, result in results.items():
        output_path = output_dir / f"{model_name}_{method}.pkl"
        
        if output_path.exists() and not args.force:
            print(f"[*] Result already exists: {output_path}")
            continue
        
        save_reduction(result, labels, paths, output_path)
        print(f"[OK] Saved: {output_path}")
    
    print(f"\n[OK] Dimension reduction complete!")


if __name__ == "__main__":
    main()
