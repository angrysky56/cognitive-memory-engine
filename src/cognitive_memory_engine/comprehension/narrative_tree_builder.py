"""
RTM Narrative Tree Builder

Implements the Random Tree Model algorithm for hierarchical narrative compression
using local LLMs via Ollama. This is part of the Comprehension Module.

Based on research by Weishun Zhong et al. on mathematical models of meaningful memory.
"""

import json
import re
from datetime import datetime

from ..llm_providers import LLMProviderFactory
from ..types import (
    ConversationTurn,
    NodeType,
    RTMConfig,
    RTMNode,
    RTMTree,
    TemporalScale,
)


class NarrativeTreeBuilder:
    """
    Builds Random Tree Model hierarchies from conversations using local LLMs.

    Implements the core RTM algorithm:
    1. Segment conversation into meaningful clauses
    2. Recursively group and summarize until convergence
    3. Build tree structure with compression tracking
    """

    def __init__(
        self,
        cloud_config=None,
        rtm_store=None,
        config: RTMConfig | None = None
    ):
        self.cloud_config = cloud_config
        self.rtm_store = rtm_store
        self.config = config or RTMConfig()

        # Create LLM provider using factory
        self.provider = LLMProviderFactory.create(self.cloud_config)
        self.llm_model = self.provider.get_current_model()

    async def _call_llm(self, prompt: str, temperature: float = 0.1) -> str:
        """Call the configured LLM provider with the given prompt."""
        return await self.provider.generate(prompt, temperature=temperature)

    async def build_tree_from_conversation(
        self,
        conversation: list[ConversationTurn],
        session_id: str,
        temporal_scale: TemporalScale = TemporalScale.DAY
    ) -> RTMTree:
        """
        Build RTM tree from a conversation using the hierarchical algorithm.

        Args:
            conversation: List of conversation turns
            session_id: Session identifier
            temporal_scale: Time scale for this narrative

        Returns:
            RTMTree with hierarchical structure and compression stats
        """
        # Step 1: Segment conversation into meaningful clauses
        clauses = await self._segment_into_clauses(conversation)

        # Step 2: Build hierarchical tree using RTM algorithm
        tree = RTMTree(
            title=await self._generate_conversation_title(conversation),
            description=await self._generate_conversation_summary(conversation),
            session_id=session_id,
            temporal_scale=temporal_scale,
            total_clauses=len(clauses),
            max_branching_factor=self.config.max_branching_factor,
            max_recall_depth=self.config.max_recall_depth
        )

        # Step 3: Recursive tree building
        root_node = await self._build_rtm_hierarchy(clauses, tree)
        tree.root_node_id = root_node.node_id
        tree.nodes[root_node.node_id] = root_node

        # Step 4: Calculate compression statistics
        tree.node_count = len(tree.nodes)
        tree.compression_ratio = len(clauses) / tree.node_count if tree.node_count > 0 else 1.0

        return tree

    async def _segment_into_clauses(
        self,
        conversation: list[ConversationTurn]
    ) -> list[str]:
        """
        Segment conversation into meaningful clauses for RTM processing.

        Uses LLM to intelligently identify semantic boundaries rather than
        simple sentence splitting.
        """
        # Combine conversation into single text
        conversation_text = "\n".join([
            f"{turn.role}: {turn.content}" for turn in conversation
        ])

        prompt = f"""
        Segment this conversation into meaningful clauses for narrative memory processing.
        Each clause should represent a single coherent idea or exchange.

        Rules:
        - Preserve speaker attribution
        - Group related exchanges
        - Aim for {self.config.min_segment_size}-3 clauses per semantic unit
        - Return as JSON list of strings

        Conversation:
        {conversation_text}

        Return only the JSON array of clause strings:
        """

        try:
            response = await self._call_llm(prompt.strip(), temperature=0.1)

            # Extract JSON from response
            clauses_json = self._extract_json_from_response(response)
            clauses = json.loads(clauses_json)

            # Validation and fallback
            if not isinstance(clauses, list) or len(clauses) == 0:
                return self._fallback_segmentation(conversation)

            return clauses

        except Exception as e:
            print(f"LLM segmentation failed: {e}")
            return self._fallback_segmentation(conversation)

    async def _build_rtm_hierarchy(
        self,
        clauses: list[str],
        tree: RTMTree,
        depth: int = 0
    ) -> RTMNode:
        """
        Recursive RTM tree building algorithm.

        Based on the Random Tree Model:
        - If clauses <= K (branching factor), create leaf nodes
        - Otherwise, group into K super-chunks and recursively summarize
        """
        # Base case: small enough for leaf nodes
        if len(clauses) <= self.config.max_branching_factor:
            if len(clauses) == 1:
                # Single clause becomes a leaf
                node = RTMNode(
                    tree_id=tree.tree_id,
                    content=clauses[0],
                    summary=clauses[0][:self.config.max_summary_length],
                    node_type=NodeType.LEAF,
                    depth=depth,
                    temporal_scale=tree.temporal_scale
                )
                tree.nodes[node.node_id] = node
                return node
            else:
                # Multiple clauses become summary node with leaf children
                summary = await self._generate_summary(clauses)
                parent_node = RTMNode(
                    tree_id=tree.tree_id,
                    content=summary,
                    summary=summary[:self.config.max_summary_length],
                    node_type=NodeType.SUMMARY,
                    depth=depth,
                    temporal_scale=tree.temporal_scale
                )

                # Create leaf children
                for clause in clauses:
                    child_node = RTMNode(
                        tree_id=tree.tree_id,
                        content=clause,
                        summary=clause[:self.config.max_summary_length],
                        node_type=NodeType.LEAF,
                        parent_id=parent_node.node_id,
                        depth=depth + 1,
                        temporal_scale=tree.temporal_scale
                    )
                    tree.nodes[child_node.node_id] = child_node
                    parent_node.children_ids.append(child_node.node_id)

                tree.nodes[parent_node.node_id] = parent_node
                return parent_node

        # Recursive case: group into K super-chunks
        K = self.config.max_branching_factor
        chunk_size = len(clauses) // K
        remainder = len(clauses) % K

        chunks = []
        start_idx = 0

        for i in range(K):
            # Distribute remainder across first chunks
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            chunks.append(clauses[start_idx:end_idx])
            start_idx = end_idx

        # Generate summary for each chunk and recurse
        child_nodes = []
        for chunk in chunks:
            if chunk:  # Skip empty chunks
                child_node = await self._build_rtm_hierarchy(chunk, tree, depth + 1)
                child_nodes.append(child_node)

        # Create parent node summarizing all children
        child_summaries = [node.summary for node in child_nodes]
        parent_summary = await self._generate_summary(child_summaries)

        parent_node = RTMNode(
            tree_id=tree.tree_id,
            content=parent_summary,
            summary=parent_summary[:self.config.max_summary_length],
            node_type=NodeType.ROOT if depth == 0 else NodeType.SUMMARY,
            depth=depth,
            temporal_scale=tree.temporal_scale,
            children_ids=[node.node_id for node in child_nodes]
        )

        # Update children with parent reference
        for child_node in child_nodes:
            child_node.parent_id = parent_node.node_id
            tree.nodes[child_node.node_id] = child_node

        tree.nodes[parent_node.node_id] = parent_node
        return parent_node

    async def _generate_summary(self, content_items: list[str]) -> str:
        """Generate intelligent summary using local LLM"""
        combined_content = "\n".join(content_items)

        prompt = f"""
        Create a concise summary of the following content for narrative memory.
        Focus on the key themes, decisions, and relationships.
        Maximum {self.config.max_summary_length} characters.

        Content:
        {combined_content}

        Summary:
        """

        try:
            response = await self._call_llm(prompt.strip(), temperature=0.2)
            summary = response.strip()

            # Ensure length constraint
            if len(summary) > self.config.max_summary_length:
                summary = summary[:self.config.max_summary_length-3] + "..."

            return summary

        except Exception as e:
            print(f"Summary generation failed: {e}")
            # Fallback: truncate first item
            fallback = content_items[0] if content_items else "Summary unavailable"
            return fallback[:self.config.max_summary_length]

    async def _generate_conversation_title(self, conversation: list[ConversationTurn]) -> str:
        """Generate a title for the conversation"""
        # Take first few turns for context
        context_turns = conversation[:3]
        context_text = "\n".join([
            f"{turn.role}: {turn.content}" for turn in context_turns
        ])

        prompt = f"""
        Generate a brief, descriptive title for this conversation.
        Maximum 50 characters. Focus on the main topic or purpose.

        Conversation:
        {context_text}

        Title:
        """

        try:
            response = await self._call_llm(prompt.strip(), temperature=0.1)
            title = response.strip().strip('"\'')
            return title[:50] if len(title) > 50 else title

        except Exception as e:
            print(f"Title generation failed: {e}")
            # Fallback: use timestamp
            return f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    async def _generate_conversation_summary(self, conversation: list[ConversationTurn]) -> str:
        """Generate a summary description of the conversation"""
        context_text = "\n".join([
            f"{turn.role}: {turn.content}" for turn in conversation
        ])

        prompt = f"""
        Generate a one-sentence description of this conversation's purpose and outcome.

        Conversation:
        {context_text}

        Description:
        """

        try:
            response = await self._call_llm(prompt.strip(), temperature=0.2)
            return response.strip()

        except Exception as e:
            print(f"Description generation failed: {e}")
            return f"Conversation from {datetime.now().strftime('%Y-%m-%d')}"

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON array from LLM response"""
        # Look for JSON array pattern
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # If no array found, try to find quoted strings and make array
        string_matches = re.findall(r'"([^"]*)"', response)
        if string_matches:
            return json.dumps(string_matches)

        raise ValueError("No valid JSON found in response")

    def _fallback_segmentation(self, conversation: list[ConversationTurn]) -> list[str]:
        """Simple fallback segmentation when LLM fails"""
        clauses = []
        for turn in conversation:
            # Split by sentences and combine with speaker
            sentences = re.split(r'[.!?]+', turn.content)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    clauses.append(f"{turn.role}: {sentence}")

        return clauses if clauses else ["Empty conversation"]
