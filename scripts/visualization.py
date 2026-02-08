"""
Visualization utilities for clustering results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import pickle


def create_cluster_plot(
    embeddings_2d: np.ndarray,
    cluster_labels: np.ndarray,
    labels_true: Optional[List[str]] = None,
    title: str = "Cluster Visualization",
    figsize: tuple = (12, 10),
    show_legend: bool = True,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a scatter plot of clustered embeddings.
    
    Args:
        embeddings_2d: 2D embedding coordinates (N, 2)
        cluster_labels: Cluster assignments
        labels_true: True species labels (optional, for coloring)
        title: Plot title
        figsize: Figure size
        show_legend: Whether to show legend
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters[unique_clusters != -1])
    
    # Color palette
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_clusters)))
    
    # Plot outliers first (if any)
    outlier_mask = cluster_labels == -1
    if np.any(outlier_mask):
        ax.scatter(
            embeddings_2d[outlier_mask, 0],
            embeddings_2d[outlier_mask, 1],
            c='gray',
            alpha=0.3,
            s=10,
            label=f'Outliers ({np.sum(outlier_mask)})'
        )
    
    # Plot each cluster
    for idx, cluster_id in enumerate(sorted(unique_clusters[unique_clusters != -1])):
        mask = cluster_labels == cluster_id
        
        # Get dominant species if available
        if labels_true is not None:
            cluster_species = [labels_true[i] for i in np.where(mask)[0]]
            from collections import Counter
            most_common = Counter(cluster_species).most_common(1)[0][0]
            label = f"C{cluster_id}: {most_common} ({np.sum(mask)})"
        else:
            label = f"Cluster {cluster_id} ({np.sum(mask)})"
        
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[idx % len(colors)]],
            alpha=0.7,
            s=20,
            label=label if show_legend and idx < 20 else None
        )
    
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title)
    
    if show_legend and n_clusters <= 30:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved plot: {save_path}")
    
    return fig


def create_species_plot(
    embeddings_2d: np.ndarray,
    labels_true: List[str],
    title: str = "Species Distribution",
    figsize: tuple = (12, 10),
    show_legend: bool = True,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a scatter plot colored by true species labels.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_species = sorted(set(labels_true))
    n_species = len(unique_species)
    
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_species)))
    
    for idx, species in enumerate(unique_species):
        mask = np.array([l == species for l in labels_true])
        
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[idx % len(colors)]],
            alpha=0.7,
            s=20,
            label=f"{species} ({np.sum(mask)})" if show_legend and idx < 20 else None
        )
    
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title)
    
    if show_legend and n_species <= 30:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved plot: {save_path}")
    
    return fig


def print_cluster_summary(
    cluster_labels: np.ndarray,
    labels_true: List[str],
    top_n: int = 5
) -> None:
    """Print summary of cluster compositions."""
    from collections import Counter
    
    unique_clusters = sorted([c for c in np.unique(cluster_labels) if c != -1])
    
    print("\n" + "=" * 60)
    print("  CLUSTER SUMMARY")
    print("=" * 60)
    
    for cluster_id in unique_clusters[:20]:  # Limit to first 20
        mask = cluster_labels == cluster_id
        cluster_species = [labels_true[i] for i in np.where(mask)[0]]
        counts = Counter(cluster_species)
        
        total = sum(counts.values())
        dominant = counts.most_common(1)[0]
        purity = dominant[1] / total
        
        print(f"\n  Cluster {cluster_id} ({total} samples, {purity:.1%} purity)")
        for species, count in counts.most_common(top_n):
            print(f"    - {species}: {count} ({count/total:.1%})")
    
    if len(unique_clusters) > 20:
        print(f"\n  ... and {len(unique_clusters) - 20} more clusters")
    
    # Outlier summary
    outliers = cluster_labels == -1
    if np.any(outliers):
        print(f"\n  Outliers: {np.sum(outliers)} samples")
        outlier_species = [labels_true[i] for i in np.where(outliers)[0]]
        for species, count in Counter(outlier_species).most_common(5):
            print(f"    - {species}: {count}")
    
    print("=" * 60)


def show_cluster_visualization(clustering_result_path: Path) -> None:
    """Load and display clustering visualization."""
    with clustering_result_path.open('rb') as f:
        data = pickle.load(f)
    
    embeddings_2d = data['embeddings_2d']
    cluster_labels = data['cluster_labels']
    labels_true = data['labels_true']
    method = data['method']
    metrics = data['metrics']
    
    # Create visualization
    title = f"{method.upper()} Clustering (V-measure: {metrics.get('v_measure', 0):.3f})"
    
    fig = create_cluster_plot(
        embeddings_2d=embeddings_2d,
        cluster_labels=cluster_labels,
        labels_true=labels_true,
        title=title,
        show_legend=True
    )
    
    print_cluster_summary(cluster_labels, labels_true)
    
    plt.show()
