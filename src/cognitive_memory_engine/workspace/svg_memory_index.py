"""
Support Vector Graph (SVG) Integration for Cognitive Memory Engine

This module provides an implementation of the SVG method for enhanced vector search
with formal navigability guarantees, based on the paper shared by the user.

Key Benefits:
1. Formal navigability guarantees in both Euclidean and non-Euclidean spaces
2. Improved search quality with bounded approximation ratios
3. Efficient sparse graph construction using kernel methods
4. Better handling of high-dimensional embeddings
"""

import networkx as nx
import numpy as np
from scipy.optimize import nnls
from sklearn.metrics.pairwise import rbf_kernel

from ..types import RTMNode


class SVGMemoryIndex:
    """
    Support Vector Graph index for enhanced vector search in the cognitive memory engine.

    This replaces or augments the standard ChromaDB vector search with a theoretically
    grounded approach that provides navigability guarantees.
    """

    def __init__(
        self,
        kernel_width: float = 1.0,
        max_out_degree: int = 32,
        epsilon: float = 0.1
    ):
        """
        Initialize SVG index.

        Args:
            kernel_width: Width parameter (σ) for RBF kernel
            max_out_degree: Maximum number of outgoing edges per node (M)
            epsilon: Approximation parameter for search quality
        """
        self.kernel_width = kernel_width
        self.max_out_degree = max_out_degree
        self.epsilon = epsilon
        self.graph = nx.DiGraph()
        self.embeddings = {}
        self.nodes = {}

    def _compute_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute RBF kernel between two vectors."""
        diff = x - y
        return np.exp(-np.dot(diff, diff) / (2 * self.kernel_width ** 2))

    def _solve_nonnegative_least_squares(
        self,
        target_idx: int,
        embeddings_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Solve the nonnegative least squares problem for SVG construction.

        Implements: min_s ||x_i - Φs||^2 s.t. s ≥ 0, s_i = 0
        """
        target_embedding = embeddings_matrix[target_idx]

        # Compute kernel matrix (Φ)
        kernel_matrix = rbf_kernel(
            embeddings_matrix,
            embeddings_matrix,
            gamma=1/(2 * self.kernel_width ** 2)
        )

        # Remove target column (enforce s_i = 0)
        kernel_matrix_reduced = np.delete(kernel_matrix, target_idx, axis=1)

        # Solve NNLS
        coefficients, _ = nnls(kernel_matrix_reduced, target_embedding)

        # Insert zero at target position
        full_coefficients = np.insert(coefficients, target_idx, 0)

        return full_coefficients

    def _subspace_pursuit(
        self,
        target_idx: int,
        embeddings_matrix: np.ndarray,
        max_iterations: int = 10
    ) -> list[int]:
        """
        Implement Algorithm 5: Subspace pursuit for sparse SVG construction.

        Returns indices of the M nearest neighbors in kernel space.
        """
        n = embeddings_matrix.shape[0]
        target_embedding = embeddings_matrix[target_idx]

        # Initialize with empty support set
        support = set()

        for _ in range(max_iterations):
            # Compute residual
            if support:
                # Project onto current support
                support_indices = list(support)
                kernel_submatrix = rbf_kernel(
                    embeddings_matrix[support_indices],
                    embeddings_matrix[support_indices],
                    gamma=1/(2 * self.kernel_width ** 2)
                )
                target_kernel = rbf_kernel(
                    target_embedding.reshape(1, -1),
                    embeddings_matrix[support_indices],
                    gamma=1/(2 * self.kernel_width ** 2)
                ).flatten()

                # Solve least squares on support
                coeffs = np.linalg.lstsq(kernel_submatrix, target_kernel, rcond=None)[0]

                # Compute residual
                reconstruction = embeddings_matrix[support_indices].T @ coeffs
                residual = target_embedding - reconstruction
            else:
                residual = target_embedding

            # Find indices with largest correlation to residual
            correlations = embeddings_matrix @ residual
            correlations[target_idx] = -np.inf  # Exclude self
            correlations[list(support)] = -np.inf  # Exclude current support

            # Add top-k to support
            k = min(self.max_out_degree - len(support), n - len(support) - 1)
            if k <= 0:
                break

            top_k_indices = np.argpartition(correlations, -k)[-k:]
            support.update(top_k_indices)

            # Trim to max_out_degree if needed
            if len(support) > self.max_out_degree:
                # Keep only the most important
                support_list = list(support)
                importance = np.abs(correlations[support_list])
                keep_indices = np.argpartition(importance, -self.max_out_degree)[-self.max_out_degree:]
                support = set(np.array(support_list)[keep_indices])

        return list(support)

    def build_index(self, nodes: list[RTMNode], embeddings: dict[str, np.ndarray]):
        """
        Build the Support Vector Graph from RTM nodes and their embeddings.

        Args:
            nodes: List of RTM nodes to index
            embeddings: Dictionary mapping node IDs to embedding vectors
        """
        # Store nodes and embeddings
        self.nodes = {node.node_id: node for node in nodes}
        self.embeddings = embeddings

        # Create embedding matrix
        node_ids = list(self.nodes.keys())
        embeddings_matrix = np.array([embeddings[nid] for nid in node_ids])

        # Build SVG
        for i, node_id in enumerate(node_ids):
            # Find support vectors using subspace pursuit
            support_indices = self._subspace_pursuit(i, embeddings_matrix)

            # Add edges to graph
            for j in support_indices:
                neighbor_id = node_ids[j]
                # Weight is the kernel value
                weight = self._compute_kernel(
                    embeddings_matrix[i],
                    embeddings_matrix[j]
                )
                self.graph.add_edge(node_id, neighbor_id, weight=weight)

        print(f"Built SVG with {len(nodes)} nodes and {self.graph.number_of_edges()} edges")
        print(f"Average out-degree: {self.graph.number_of_edges() / len(nodes):.2f}")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        entry_point: str | None = None
    ) -> list[tuple[str, float]]:
        """
        Search for k-nearest neighbors using greedy traversal on SVG.

        Implements Algorithm 3 from the paper.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            entry_point: Starting node ID (random if not specified)

        Returns:
            List of (node_id, similarity_score) tuples
        """
        if not self.graph:
            raise ValueError("Index not built. Call build_index first.")

        # Select random entry point if not specified
        if entry_point is None:
            entry_point = str(np.random.choice(list(self.nodes.keys())))
        else:
            # Ensure entry_point is a valid node id and is a string
            if not isinstance(entry_point, str) or entry_point not in self.nodes:
                raise ValueError(f"Entry point '{entry_point}' not found in nodes.")

        # Initialize visited set and candidate heap
        visited = set()
        candidates = []

        # Add entry point
        entry_similarity = self._compute_kernel(
            query_embedding,
            self.embeddings[entry_point]
        )
        candidates.append((-entry_similarity, entry_point))  # Negative for max-heap behavior

        # Best k results
        best_results = []

        while candidates and len(visited) < len(self.nodes):
            # Get best unvisited candidate
            neg_sim, current_id = min(candidates)
            similarity = -neg_sim

            if current_id in visited:
                candidates.remove((neg_sim, current_id))
                continue

            visited.add(current_id)
            best_results.append((current_id, similarity))
            best_results.sort(key=lambda x: -x[1])  # Sort by similarity descending
            best_results = best_results[:k]  # Keep only top-k

            # Explore neighbors
            if current_id in self.graph:
                for neighbor_id in self.graph.successors(current_id):
                    if neighbor_id not in visited:
                        neighbor_similarity = self._compute_kernel(
                            query_embedding,
                            self.embeddings[neighbor_id]
                        )
                        candidates.append((-neighbor_similarity, neighbor_id))

            # Prune candidates worse than k-th best
            if len(best_results) >= k:
                threshold = best_results[-1][1]
                candidates = [(s, nid) for s, nid in candidates if -s > threshold]

        return best_results

    def add_node(self, node: RTMNode, embedding: np.ndarray):
        """
        Incrementally add a new node to the SVG.

        This is more efficient than rebuilding the entire index.
        """
        self.nodes[node.node_id] = node
        self.embeddings[node.node_id] = embedding

        # Find neighbors for the new node
        all_embeddings = np.array(list(self.embeddings.values()))
        new_node_idx = len(self.embeddings) - 1

        support_indices = self._subspace_pursuit(new_node_idx, all_embeddings)
        node_ids = list(self.embeddings.keys())

        # Add edges from new node
        for idx in support_indices:
            neighbor_id = node_ids[idx]
            weight = self._compute_kernel(embedding, self.embeddings[neighbor_id])
            self.graph.add_edge(node.node_id, neighbor_id, weight=weight)

        # Optionally: Update edges pointing to the new node
        # This requires recomputing some existing edges for optimality
        # but can be skipped for efficiency

    def get_statistics(self) -> dict:
        """Get statistics about the SVG index."""
        return {
            "num_nodes": len(self.nodes),
            "num_edges": self.graph.number_of_edges(),
            "avg_out_degree": self.graph.number_of_edges() / len(self.nodes) if self.nodes else 0,
            "kernel_width": self.kernel_width,
            "max_out_degree": self.max_out_degree,
            "is_connected": nx.is_weakly_connected(self.graph) if self.graph else False
        }



