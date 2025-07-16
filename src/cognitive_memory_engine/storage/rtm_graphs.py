"""
RTM Graph Store

Handles persistence and retrieval of Random Tree Model narrative structures
using NetworkX for graph operations and efficient serialization.

Provides tree traversal, compression analytics, and export capabilities.
"""

import json
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

from ..types import NodeType, RTMNode, RTMTree, TemporalScale


class RTMGraphStore:
    """
    Manages storage and retrieval of RTM narrative trees.

    Uses NetworkX for efficient graph operations and provides:
    - Persistent storage of tree structures
    - Tree traversal algorithms
    - Compression and statistics analysis
    - Export capabilities for visualization
    """

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Separate directories for different storage formats
        self.trees_dir = self.storage_path / "trees"
        self.graphs_dir = self.storage_path / "graphs"
        self.exports_dir = self.storage_path / "exports"

        for dir_path in [self.trees_dir, self.graphs_dir, self.exports_dir]:
            dir_path.mkdir(exist_ok=True)

        # In-memory cache for frequently accessed trees
        self._tree_cache: dict[str, RTMTree] = {}
        self._graph_cache: dict[str, nx.DiGraph] = {}

        # Track statistics
        self._stats = {
            "total_trees": 0,
            "total_nodes": 0,
            "avg_compression_ratio": 0.0,
            "storage_size_bytes": 0
        }

    async def store_tree(self, tree: RTMTree) -> bool:
        """
        Store an RTM tree with both structured data and NetworkX graph.

        Args:
            tree: RTMTree to store

        Returns:
            bool: Success status
        """
        try:
            # Store structured tree data
            tree_path = self.trees_dir / f"{tree.tree_id}.json"
            tree_data = self._serialize_tree(tree)

            with open(tree_path, 'w') as f:
                json.dump(tree_data, f, indent=2)

            # Convert to NetworkX graph and store
            graph = self._tree_to_networkx(tree)
            graph_path = self.graphs_dir / f"{tree.tree_id}.pickle"

            with open(graph_path, 'wb') as f:
                pickle.dump(graph, f)

            # Update cache
            self._tree_cache[tree.tree_id] = tree
            self._graph_cache[tree.tree_id] = graph

            # Update statistics
            await self._update_statistics()

            return True

        except Exception as e:
            print(f"Error storing tree {tree.tree_id}: {e}")
            return False

    async def load_tree(self, tree_id: str) -> RTMTree | None:
        """Load an RTM tree by ID"""
        # Check cache first
        if tree_id in self._tree_cache:
            return self._tree_cache[tree_id]

        tree_path = self.trees_dir / f"{tree_id}.json"
        if not tree_path.exists():
            return None

        try:
            with open(tree_path) as f:
                tree_data = json.load(f)

            tree = self._deserialize_tree(tree_data)

            # Cache for future access
            self._tree_cache[tree_id] = tree

            return tree

        except Exception as e:
            print(f"Error loading tree {tree_id}: {e}")
            return None

    async def load_graph(self, tree_id: str) -> nx.DiGraph | None:
        """Load NetworkX graph representation of tree"""
        # Check cache first
        if tree_id in self._graph_cache:
            return self._graph_cache[tree_id]

        graph_path = self.graphs_dir / f"{tree_id}.pickle"
        if not graph_path.exists():
            # Try to regenerate from tree
            tree = await self.load_tree(tree_id)
            if tree:
                graph = self._tree_to_networkx(tree)
                self._graph_cache[tree_id] = graph
                return graph
            return None

        try:
            with open(graph_path, 'rb') as f:
                graph = pickle.load(f)

            # Cache for future access
            self._graph_cache[tree_id] = graph

            return graph

        except Exception as e:
            print(f"Error loading graph {tree_id}: {e}")
            return None

    async def traverse_tree(
        self,
        tree_id: str,
        max_depth: int | None = None,
        start_node_id: str | None = None
    ) -> list[RTMNode]:
        """
        Traverse RTM tree using breadth-first search.

        Args:
            tree_id: Tree to traverse
            max_depth: Maximum depth to traverse (None = full tree)
            start_node_id: Node to start from (None = root)

        Returns:
            list of nodes in traversal order
        """
        tree = await self.load_tree(tree_id)
        if not tree:
            return []

        start_id = start_node_id or tree.root_node_id
        if start_id not in tree.nodes:
            return []

        visited = []
        queue = [(start_id, 0)]  # (node_id, depth)
        seen = set()

        while queue:
            node_id, depth = queue.pop(0)

            # Check depth limit
            if max_depth is not None and depth > max_depth:
                continue

            # Avoid cycles
            if node_id in seen:
                continue
            seen.add(node_id)

            # Add node to results
            if node_id in tree.nodes:
                visited.append(tree.nodes[node_id])

                # Add children to queue
                node = tree.nodes[node_id]
                for child_id in node.children_ids:
                    if child_id not in seen:
                        queue.append((child_id, depth + 1))

        return visited

    async def find_nodes_by_content(
        self,
        tree_id: str,
        search_term: str,
        node_types: list[NodeType] | None = None
    ) -> list[RTMNode]:
        """Find nodes containing specific content"""
        tree = await self.load_tree(tree_id)
        if not tree:
            return []

        matches = []
        search_term_lower = search_term.lower()

        for node in tree.nodes.values():
            # Check node type filter
            if node_types and node.node_type not in node_types:
                continue

            # Check content match
            content_text = f"{node.content} {node.summary}".lower()
            if search_term_lower in content_text:
                matches.append(node)

        return matches

    async def get_tree_statistics(self, tree_id: str) -> dict[str, Any]:
        """Get detailed statistics for a specific tree"""
        tree = await self.load_tree(tree_id)
        if not tree:
            return {}

        graph = await self.load_graph(tree_id)

        stats = {
            "tree_id": tree_id,
            "total_nodes": len(tree.nodes),
            "compression_ratio": tree.compression_ratio,
            "max_depth": max(node.depth for node in tree.nodes.values()) if tree.nodes else 0,
            "node_types": {},
            "temporal_scale": tree.temporal_scale.value,
            "created": tree.created.isoformat(),
            "session_id": tree.session_id
        }

        # Count node types
        for node in tree.nodes.values():
            node_type = node.node_type.value
            stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1

        # Graph-based statistics
        if graph:
            stats.update({
                "graph_density": nx.density(graph),
                "average_clustering": nx.average_clustering(graph.to_undirected()),
                "diameter": nx.diameter(graph.to_undirected()) if nx.is_connected(graph.to_undirected()) else "disconnected"
            })

        return stats

    async def get_statistics(self) -> dict[str, Any]:
        """Get overall storage statistics"""
        await self._update_statistics()
        return self._stats.copy()

    async def list_trees(
        self,
        session_id: str | None = None,
        temporal_scale: TemporalScale | None = None,
        limit: int | None = None
    ) -> list[dict[str, Any]]:
        """list stored trees with optional filtering"""
        tree_files = list(self.trees_dir.glob("*.json"))
        trees_info = []

        for tree_file in tree_files:
            try:
                with open(tree_file) as f:
                    tree_data = json.load(f)

                # Apply filters
                if session_id and tree_data.get("session_id") != session_id:
                    continue

                if temporal_scale and tree_data.get("temporal_scale") != temporal_scale.value:
                    continue

                trees_info.append({
                    "tree_id": tree_data["tree_id"],
                    "title": tree_data["title"],
                    "session_id": tree_data["session_id"],
                    "created": tree_data["created"],
                    "node_count": len(tree_data["nodes"]),
                    "compression_ratio": tree_data["compression_ratio"]
                })

                if limit and len(trees_info) >= limit:
                    break

            except Exception as e:
                print(f"Error reading tree file {tree_file}: {e}")
                continue

        # Sort by creation time (newest first)
        trees_info.sort(key=lambda x: x["created"], reverse=True)

        return trees_info

    async def export_graph(
        self,
        tree_id: str,
        format: str = "gml",
        include_embeddings: bool = False
    ) -> str:
        """
        Export tree as graph file for visualization.

        Args:
            tree_id: Tree to export
            format: Export format ('gml', 'graphml', 'json', 'dot')
            include_embeddings: Whether to include vector embeddings

        Returns:
            Path to exported file
        """
        graph = await self.load_graph(tree_id)
        if not graph:
            raise ValueError(f"Tree {tree_id} not found")

        # Prepare graph for export
        export_graph = graph.copy()

        # Optionally remove embeddings to reduce file size
        if not include_embeddings:
            for node_id in export_graph.nodes():
                node_data = export_graph.nodes[node_id]
                node_data.pop("base_embedding", None)
                node_data.pop("weighted_embedding", None)

        # Export based on format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{tree_id}_{timestamp}.{format}"
        export_path = self.exports_dir / filename

        if format == "gml":
            nx.write_gml(export_graph, export_path)
        elif format == "graphml":
            nx.write_graphml(export_graph, export_path)
        elif format == "json":
            graph_data = nx.node_link_data(export_graph)
            with open(export_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
        elif format == "dot":
            nx.drawing.nx_pydot.write_dot(export_graph, export_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        return str(export_path)

    async def delete_tree(self, tree_id: str) -> bool:
        """Delete a tree and its associated files"""
        try:
            # Remove files
            tree_path = self.trees_dir / f"{tree_id}.json"
            graph_path = self.graphs_dir / f"{tree_id}.pickle"

            if tree_path.exists():
                tree_path.unlink()
            if graph_path.exists():
                graph_path.unlink()

            # Remove from cache
            self._tree_cache.pop(tree_id, None)
            self._graph_cache.pop(tree_id, None)

            # Update statistics
            await self._update_statistics()

            return True

        except Exception as e:
            print(f"Error deleting tree {tree_id}: {e}")
            return False

    def _serialize_tree(self, tree: RTMTree) -> dict[str, Any]:
        """Convert RTMTree to JSON-serializable format"""
        tree_dict = asdict(tree)

        # Convert datetime objects to ISO strings
        tree_dict["created"] = tree.created.isoformat()
        tree_dict["temporal_scale"] = tree.temporal_scale.value

        # Convert nodes dict
        nodes_dict = {}
        for node_id, node in tree.nodes.items():
            node_dict = asdict(node)
            node_dict["timestamp"] = node.timestamp.isoformat()
            node_dict["node_type"] = node.node_type.value
            node_dict["temporal_scale"] = node.temporal_scale.value
            nodes_dict[node_id] = node_dict

        tree_dict["nodes"] = nodes_dict

        return tree_dict

    def _deserialize_tree(self, tree_data: dict[str, Any]) -> RTMTree:
        """Convert JSON data back to RTMTree"""
        # Parse temporal scale
        temporal_scale = TemporalScale(tree_data["temporal_scale"])

        # Parse creation time
        created = datetime.fromisoformat(tree_data["created"])

        # Create tree
        tree = RTMTree(
            tree_id=tree_data["tree_id"],
            title=tree_data["title"],
            description=tree_data["description"],
            root_node_id=tree_data["root_node_id"],
            max_branching_factor=tree_data["max_branching_factor"],
            max_recall_depth=tree_data["max_recall_depth"],
            total_clauses=tree_data["total_clauses"],
            compression_ratio=tree_data["compression_ratio"],
            node_count=tree_data["node_count"],
            created=created,
            temporal_scale=temporal_scale,
            session_id=tree_data["session_id"]
        )

        # Parse nodes
        for node_id, node_data in tree_data["nodes"].items():
            node = RTMNode(
                node_id=node_data["node_id"],
                content=node_data["content"],
                summary=node_data["summary"],
                node_type=NodeType(node_data["node_type"]),
                parent_id=node_data.get("parent_id"),
                children_ids=node_data["children_ids"],
                depth=node_data["depth"],
                base_embedding=node_data.get("base_embedding"),
                salience_score=node_data.get("salience_score", 1.0),
                weighted_embedding=node_data.get("weighted_embedding"),
                timestamp=datetime.fromisoformat(node_data["timestamp"]),
                temporal_scale=TemporalScale(node_data["temporal_scale"]),
                metadata=node_data.get("metadata", {})
            )
            tree.nodes[node_id] = node

        return tree

    def _tree_to_networkx(self, tree: RTMTree) -> nx.DiGraph:
        """Convert RTMTree to NetworkX directed graph"""
        graph = nx.DiGraph()

        # Add nodes with attributes
        for node_id, node in tree.nodes.items():
            graph.add_node(
                node_id,
                content=node.content,
                summary=node.summary,
                node_type=node.node_type.value,
                depth=node.depth,
                salience_score=node.salience_score,
                timestamp=node.timestamp.isoformat(),
                base_embedding=node.base_embedding,
                weighted_embedding=node.weighted_embedding
            )

        # Add edges (parent-child relationships)
        for node_id, node in tree.nodes.items():
            for child_id in node.children_ids:
                if child_id in tree.nodes:
                    graph.add_edge(node_id, child_id, relationship="parent_child")

        # Add graph-level metadata
        graph.graph.update({
            "tree_id": tree.tree_id,
            "title": tree.title,
            "compression_ratio": tree.compression_ratio,
            "temporal_scale": tree.temporal_scale.value,
            "session_id": tree.session_id
        })

        return graph

    async def _update_statistics(self):
        """Update storage statistics"""
        tree_files = list(self.trees_dir.glob("*.json"))

        total_trees = len(tree_files)
        total_nodes = 0
        total_compression = 0.0
        storage_size = 0

        for tree_file in tree_files:
            try:
                with open(tree_file) as f:
                    tree_data = json.load(f)

                total_nodes += len(tree_data.get("nodes", {}))
                total_compression += tree_data.get("compression_ratio", 1.0)
                storage_size += tree_file.stat().st_size

            except Exception:
                continue

        # Add graph storage size
        for graph_file in self.graphs_dir.glob("*.pickle"):
            storage_size += graph_file.stat().st_size

        self._stats.update({
            "total_trees": total_trees,
            "total_nodes": total_nodes,
            "avg_compression_ratio": total_compression / max(total_trees, 1),
            "storage_size_bytes": storage_size
        })

    async def close(self):
        """Clean shutdown - mainly clears caches"""
        self._tree_cache.clear()
        self._graph_cache.clear()
