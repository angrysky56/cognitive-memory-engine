"""
Semantic Similarity Calculator

Provides semantic similarity scoring using embeddings for better
concept matching between conversations and documents.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from ..types import KnowledgeConcept

logger = logging.getLogger(__name__)


class SemanticSimilarityCalculator:
    """
    Calculates semantic similarity between text segments using embeddings.

    Uses cosine similarity between embedding vectors to determine
    how semantically related two pieces of text are.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic similarity calculator.

        Args:
            embedding_model: Name of sentence transformer model to use
        """
        self.embedding_model_name = embedding_model
        self._model = None

    @property
    def model(self) -> SentenceTransformer | None:
        """Lazy load the embedding model."""
        if self._model is None and SentenceTransformer is not None:
            try:
                self._model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self._model = None
        return self._model

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two text strings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Check if model is available
            if self.model is None:
                logger.debug("Embedding model not available, using word overlap")
                return self._word_overlap_similarity(text1, text2)

            # Get embeddings
            embeddings = self.model.encode([text1, text2])

            # Calculate cosine similarity
            similarity = self._cosine_similarity(embeddings[0], embeddings[1])

            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            # Fall back to simple word overlap
            return self._word_overlap_similarity(text1, text2)

    def calculate_concept_similarity(self,
                                   node_content: str,
                                   concept: KnowledgeConcept) -> float:
        """
        Calculate similarity between conversation node and knowledge concept.

        Considers multiple aspects of the concept for better matching.

        Args:
            node_content: Content from conversation node
            concept: Knowledge concept to compare

        Returns:
            Similarity score between 0 and 1
        """
        # Combine concept information for richer comparison
        concept_text = f"{concept.name} {concept.description} {concept.content}"

        # Calculate base similarity
        base_similarity = self.calculate_similarity(node_content, concept_text)

        # Boost score if concept name appears in content (exact match bonus)
        if concept.name.lower() in node_content.lower():
            base_similarity = min(1.0, base_similarity + 0.2)

        # Consider concept salience as a weighting factor
        weighted_similarity = base_similarity * (0.5 + 0.5 * concept.salience_score)

        return weighted_similarity

    def find_similar_concepts(self,
                            node_content: str,
                            concepts: list[tuple[str, KnowledgeConcept]],
                            threshold: float = 0.3) -> list[tuple[KnowledgeConcept, float]]:
        """
        Find all concepts semantically similar to node content.

        Args:
            node_content: Content from conversation node
            concepts: List of (concept_id, concept) tuples
            threshold: Minimum similarity score to include

        Returns:
            List of (concept, similarity_score) tuples above threshold
        """
        similar_concepts = []

        for _concept_id, concept in concepts:
            similarity = self.calculate_concept_similarity(node_content, concept)

            if similarity >= threshold:
                similar_concepts.append((concept, similarity))

        # Sort by similarity score descending
        similar_concepts.sort(key=lambda x: x[1], reverse=True)

        return similar_concepts

    def batch_calculate_similarities(self,
                                   texts1: list[str],
                                   texts2: list[str]) -> np.ndarray:
        """
        Calculate similarities between all pairs of texts efficiently.

        Args:
            texts1: First list of texts
            texts2: Second list of texts

        Returns:
            Matrix of similarity scores
        """
        try:
            if not self.model:
                return np.zeros((len(texts1), len(texts2)))

            # Get all embeddings at once
            all_texts = texts1 + texts2
            embeddings = self.model.encode(all_texts)

            # Split back into two sets
            embeddings1 = embeddings[:len(texts1)]
            embeddings2 = embeddings[len(texts1):]

            # Calculate pairwise similarities
            similarities = np.zeros((len(texts1), len(texts2)))

            for i, emb1 in enumerate(embeddings1):
                for j, emb2 in enumerate(embeddings2):
                    similarities[i, j] = self._cosine_similarity(emb1, emb2)

            return similarities

        except Exception as e:
            logger.error(f"Error in batch similarity calculation: {e}")
            # Return zero matrix on error
            return np.zeros((len(texts1), len(texts2)))

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @staticmethod
    def _word_overlap_similarity(text1: str, text2: str) -> float:
        """
        Simple word overlap similarity as fallback.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Jaccard similarity between word sets
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)
