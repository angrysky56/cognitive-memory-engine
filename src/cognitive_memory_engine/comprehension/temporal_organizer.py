"""
Temporal Organizer

Implements the temporal books and shelves concept for organizing RTM trees
across multiple time scales with intelligent compression and archiving.

Based on the concept from kng-mcp-server but enhanced with RTM integration.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from ..types import (
    ConversationTurn,
    LibraryShelf,
    NeuralGainConfig,
    RTMTree,
    ShelfCategory,
    TemporalBook,
    TemporalScale,
)

logger = logging.getLogger(__name__)


class TemporalOrganizer:
    """
    Organizes RTM trees into temporal books and library shelves.

    Implements multi-scale temporal hierarchy:
    - Minute → Hour → Day → Week → Month → Year
    - Automatic compression as information ages
    - Intelligent shelf categorization (Active, Recent, Reference, Archived)
    - Theme persistence across time boundaries
    """

    def __init__(
        self,
        temporal_library=None,
        vector_manager=None,
        rtm_store=None,
        config: NeuralGainConfig | None = None
    ):
        self.temporal_library = temporal_library
        self.vector_manager = vector_manager
        self.rtm_store = rtm_store
        self.config = config or NeuralGainConfig()

        # Current active session tracking
        self.current_session_books: dict[str, str] = {}  # session_id -> book_id

        # Compression targets per temporal scale
        self.compression_targets = self.config.compression_targets

        # Shelf management
        self.shelf_categories = {
            ShelfCategory.ACTIVE: timedelta(days=1),      # < 1 day
            ShelfCategory.RECENT: timedelta(days=7),      # 1-7 days
            ShelfCategory.REFERENCE: None,                # Persistent themes
            ShelfCategory.ARCHIVED: timedelta(days=30)    # > 30 days
        }

    async def start_session(self, session_id: str) -> dict[str, Any]:
        """Initialize temporal organization for a new session"""
        if self.temporal_library is None:
            raise ValueError("temporal_library must be initialized")

        # Create or get today's active book
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        book_id = f"book_day_{today.strftime('%Y_%m_%d')}_{session_id[:8]}"

        # Check if book already exists
        existing_book = await self.temporal_library.get_book(book_id)

        if not existing_book:
            # Create new daily book
            book = TemporalBook(
                book_id=book_id,
                title=f"Daily Sessions - {today.strftime('%Y-%m-%d')}",
                description=f"Conversations and activities for {today.strftime('%B %d, %Y')}",
                temporal_scale=TemporalScale.DAY,
                start_time=today,
                end_time=today + timedelta(days=1),
                shelf_category=ShelfCategory.ACTIVE,
                session_ids=[session_id]
            )

            if self.temporal_library is None:
                raise ValueError("temporal_library must be initialized before storing books")
            await self.temporal_library.store_book(book)

            # Add to active shelf
            await self._add_book_to_shelf(book, ShelfCategory.ACTIVE)
        else:
            # Add session to existing book
            if session_id not in existing_book.session_ids:
                existing_book.session_ids.append(session_id)
                await self.temporal_library.store_book(existing_book)

        # Track current session
        self.current_session_books[session_id] = book_id

        return {
            "session_id": session_id,
            "book_id": book_id,
            "temporal_scale": "day",
            "shelf_category": "active"
        }

    async def organize_conversation(
        self,
        conversation: list[ConversationTurn],
        rtm_tree: RTMTree,
        session_id: str
    ) -> TemporalBook:
        """
        Organize a conversation's RTM tree into temporal structure.

        Process:
        1. Determine appropriate temporal book (usually daily)
        2. Add RTM tree to book
        3. Extract and update persistent themes
        4. Apply compression if needed
        5. Update shelf categorization
        """
        if self.temporal_library is None:
            raise ValueError("temporal_library must be initialized")

        # Get or create appropriate temporal book
        book_id = self.current_session_books.get(session_id)
        if not book_id:
            session_result = await self.start_session(session_id)
            book_id = session_result["book_id"]

        book = await self.temporal_library.get_book(book_id)
        if not book:
            raise ValueError(f"Book {book_id} not found")

        # Add RTM tree to book
        book.rtm_tree_ids.append(rtm_tree.tree_id)
        book.last_accessed = datetime.now()

        # Extract themes from conversation and RTM tree
        themes = await self._extract_themes(conversation, rtm_tree)

        # Update persistent themes
        for theme in themes:
            if theme not in book.persistent_themes:
                book.persistent_themes.append(theme)

        # Calculate compression ratio for book level
        total_clauses = sum([
            await self._get_tree_clause_count(tree_id)
            for tree_id in book.rtm_tree_ids
        ])

        book.compression_ratio = total_clauses / len(book.rtm_tree_ids) if book.rtm_tree_ids else 1.0
        book.narrative_depth = max(book.narrative_depth, rtm_tree.max_recall_depth)

        # Store updated book
        await self.temporal_library.store_book(book)

        # Check if book needs shelf reorganization
        await self._update_book_shelf_category(book)

        return book

    async def _extract_themes(
        self,
        conversation: list[ConversationTurn],
        rtm_tree: RTMTree
    ) -> list[str]:
        """
        Extract persistent themes from conversation and RTM tree.

        Themes are concepts that are likely to recur across conversations.
        We identify them from:
        1. Root and summary node content
        2. Repeated key terms
        3. Named entities (projects, people, etc.)
        """
        themes = set()

        # Extract from RTM tree summaries
        for node in rtm_tree.nodes.values():
            if node.node_type.value in ["root", "summary"]:
                # Simple keyword extraction from summaries
                words = node.summary.lower().split()

                # Look for potential themes (capitalized words, projects, etc.)
                for word in words:
                    if len(word) > 3 and (
                        word.istitle() or  # Capitalized
                        "project" in word.lower() or
                        "timeline" in word.lower() or
                        "api" in word.lower()
                    ):
                        themes.add(word.strip(".,!?"))

        # Extract from conversation content
        conversation_text = " ".join([turn.content for turn in conversation])

        # Simple theme extraction (could be enhanced with NLP)
        potential_themes = [
            "Phoenix", "project", "timeline", "Q3", "launch", "API",
            "integration", "backend", "frontend", "testing", "auth"
        ]

        for theme in potential_themes:
            if theme.lower() in conversation_text.lower():
                themes.add(theme)

        return list(themes)[:10]  # Limit to top themes

    async def _get_tree_clause_count(self, tree_id: str) -> int:
        """Get clause count for compression calculation"""
        # This should query the RTM store for tree statistics
        if self.rtm_store is None:
            raise ValueError("rtm_store must be initialized before loading trees")
        tree = await self.rtm_store.load_tree(tree_id)
        return tree.total_clauses if tree else 0

    async def _add_book_to_shelf(self, book: TemporalBook, category: ShelfCategory):
        """Add book to appropriate shelf"""
        if self.temporal_library is None:
            raise ValueError("temporal_library must be initialized before adding books to shelves")
        shelf_id = f"shelf_{category.value}_{book.temporal_scale.value}"

        # Get or create shelf
        shelf = await self.temporal_library.get_shelf(shelf_id)
        if not shelf:
            shelf = LibraryShelf(
                shelf_id=shelf_id,
                name=f"{category.value.title()} {book.temporal_scale.value.title()} Books",
                description=f"Books in {category.value} category at {book.temporal_scale.value} scale",
                category=category,
                primary_scale=book.temporal_scale,
                time_span_start=book.start_time,
                time_span_end=book.end_time
            )

        # Add book to shelf
        if book.book_id not in shelf.book_ids:
            shelf.book_ids.append(book.book_id)
            shelf.last_accessed = datetime.now()

            # Update shelf time span
            if book.start_time < shelf.time_span_start:
                shelf.time_span_start = book.start_time
            if book.end_time and (not shelf.time_span_end or book.end_time > shelf.time_span_end):
                shelf.time_span_end = book.end_time

        await self.temporal_library.store_shelf(shelf)

    async def _update_book_shelf_category(self, book: TemporalBook):
        """Update book's shelf category based on age and activity"""
        if self.temporal_library is None:
            raise ValueError("temporal_library must be initialized before updating book shelf category")

        now = datetime.now()
        book_age = now - book.created
        last_access_age = now - book.last_accessed

        # Determine new category
        new_category = book.shelf_category

        if last_access_age <= self.shelf_categories[ShelfCategory.ACTIVE]:
            new_category = ShelfCategory.ACTIVE
        elif book_age <= self.shelf_categories[ShelfCategory.RECENT]:
            new_category = ShelfCategory.RECENT
        elif book_age > self.shelf_categories[ShelfCategory.ARCHIVED]:
            new_category = ShelfCategory.ARCHIVED

        # Check for reference category (books with high theme persistence)
        if len(book.persistent_themes) >= 3:  # Rich in themes
            new_category = ShelfCategory.REFERENCE

        # Update if category changed
        if new_category != book.shelf_category:
            old_category = book.shelf_category
            book.shelf_category = new_category

            # Move between shelves
            await self._move_book_between_shelves(book, old_category, new_category)
            await self.temporal_library.store_book(book)

    async def _move_book_between_shelves(
        self,
        book: TemporalBook,
        old_category: ShelfCategory,
        new_category: ShelfCategory
    ):
        """Move a book from one shelf category to another"""
        if self.temporal_library is None:
            raise ValueError("temporal_library must be initialized before moving books between shelves")

        old_shelf_id = f"shelf_{old_category.value}_{book.temporal_scale.value}"

        # Remove from old shelf
        old_shelf = await self.temporal_library.get_shelf(old_shelf_id)
        if old_shelf and book.book_id in old_shelf.book_ids:
            old_shelf.book_ids.remove(book.book_id)
            await self.temporal_library.store_shelf(old_shelf)

        # Add to new shelf
        await self._add_book_to_shelf(book, new_category)

    async def compress_old_memories(
        self,
        cutoff_date: datetime,
        compression_factor: float = 0.1
    ) -> dict[str, Any]:
        """
        Compress memories older than cutoff date.

        Process:
        1. Find books older than cutoff
        2. Apply temporal compression to RTM trees
        3. Merge related books at higher temporal scales
        4. Move to archived shelves
        """
        if self.temporal_library is None:
            raise ValueError("temporal_library must be initialized before compressing memories")

        compressed_books = 0
        merged_books = 0
        freed_space = 0

        # Get all books older than cutoff
        old_books = await self.temporal_library.find_books_older_than(cutoff_date)

        for book in old_books:
            if book.shelf_category == ShelfCategory.ARCHIVED:
                continue  # Already compressed

            # Apply compression to the book
            original_size = len(book.rtm_tree_ids)

            # Compress RTM trees (keep only high-salience summaries)
            compressed_tree_ids = await self._compress_rtm_trees(
                book.rtm_tree_ids, compression_factor
            )

            book.rtm_tree_ids = compressed_tree_ids
            book.compression_ratio *= (1 / compression_factor)
            book.shelf_category = ShelfCategory.ARCHIVED

            await self.temporal_library.store_book(book)
            await self._update_book_shelf_category(book)

            compressed_books += 1
            freed_space += original_size - len(compressed_tree_ids)

        # Merge books at higher temporal scales
        merged_books = await self._merge_books_by_temporal_scale(old_books)

        return {
            "compressed_books": compressed_books,
            "merged_books": merged_books,
            "freed_rtm_trees": freed_space,
            "compression_factor": compression_factor
        }

    async def _compress_rtm_trees(
        self,
        tree_ids: list[str],
        compression_factor: float
    ) -> list[str]:
        """
        Compress RTM trees by keeping only high-salience nodes.

        This is where the neural gain mechanism helps - we can identify
        which trees and nodes are most important to preserve.
        """
        if not self.vector_manager or not tree_ids:
            # Fallback: keep a proportion based on compression factor
            keep_count = max(1, int(len(tree_ids) * compression_factor))
            return tree_ids[-keep_count:]

        # Get salience scores for each tree
        salient_trees = []
        for tree_id in tree_ids:
            # Use the conversation-level vector for the tree's salience
            try:
                results = await self.vector_manager.collections["conversations"].get(
                    ids=[f"conv_{tree_id}"],
                    include=["metadatas"]
                )
                if results and results["metadatas"]:
                    salience = results["metadatas"][0].get("salience_score", 0.0)
                    salient_trees.append((tree_id, salience))
            except Exception:
                # Handle cases where vector not found
                salient_trees.append((tree_id, 0.0))

        # Sort by salience and keep the top proportion
        salient_trees.sort(key=lambda x: x[1], reverse=True)
        keep_count = max(1, int(len(salient_trees) * compression_factor))

        return [tree_id for tree_id, salience in salient_trees[:keep_count]]

    async def _merge_books_by_temporal_scale(self, books: list[TemporalBook]) -> int:
        """
        Merge books into higher temporal scales (days → weeks → months).

        Groups related books and creates summary books at higher scales.
        """
        merged_count = 0

        # Group books by temporal scale and time period
        scale_groups = {}

        for book in books:
            scale = book.temporal_scale
            if scale not in scale_groups:
                scale_groups[scale] = []
            scale_groups[scale].append(book)

        # Merge daily books into weekly, weekly into monthly, etc.
        for scale, book_list in scale_groups.items():
            if scale == TemporalScale.DAY and len(book_list) >= 7:
                # Merge into weekly book
                weekly_book = await self._create_merged_book(
                    book_list[:7], TemporalScale.WEEK
                )
                if weekly_book:
                    merged_count += 1

            elif scale == TemporalScale.WEEK and len(book_list) >= 4:
                # Merge into monthly book
                monthly_book = await self._create_merged_book(
                    book_list[:4], TemporalScale.MONTH
                )
                if monthly_book:
                    merged_count += 1

        return merged_count

    async def _create_merged_book(
        self,
        books_to_merge: list[TemporalBook],
        target_scale: TemporalScale
    ) -> TemporalBook | None:
        """Create a merged book at higher temporal scale"""
        if not books_to_merge:
            return None

        # Combine all RTM trees
        all_tree_ids = []
        all_themes = set()
        all_session_ids = []

        for book in books_to_merge:
            all_tree_ids.extend(book.rtm_tree_ids)
            all_themes.update(book.persistent_themes)
            all_session_ids.extend(book.session_ids)

        # Create merged book
        start_time = min(book.start_time for book in books_to_merge)
        end_time = max(book.end_time for book in books_to_merge if book.end_time)

        merged_book = TemporalBook(
            book_id=f"merged_{target_scale.value}_{start_time.strftime('%Y_%m_%d')}",
            title=f"{target_scale.value.title()} Summary - {start_time.strftime('%Y-%m-%d')}",
            description=f"Merged {len(books_to_merge)} {books_to_merge[0].temporal_scale.value} books",
            temporal_scale=target_scale,
            start_time=start_time,
            end_time=end_time,
            rtm_tree_ids=all_tree_ids,
            persistent_themes=list(all_themes),
            shelf_category=ShelfCategory.ARCHIVED,
            session_ids=list(set(all_session_ids)),
            compression_ratio=sum(book.compression_ratio for book in books_to_merge) / len(books_to_merge)
        )

        if self.temporal_library:
            await self.temporal_library.store_book(merged_book)
            await self._add_book_to_shelf(merged_book, ShelfCategory.ARCHIVED)

            # Remove original books
            for book in books_to_merge:
                await self.temporal_library.delete_book(book.book_id)

        return merged_book

    async def get_relevant_books(
        self,
        query: str,
        temporal_scope: TemporalScale | None = None,
        max_books: int = 10
    ) -> list[TemporalBook]:
        """
        Find books relevant to a query using themes and temporal scope.

        This integrates with the vector manager to find semantically
        relevant books based on their themes and content.
        """
        if not self.temporal_library:
            return []

        # Get all books in scope
        if temporal_scope:
            candidate_books = await self.temporal_library.find_books_by_scale(temporal_scope)
        else:
            candidate_books = await self.temporal_library.get_all_books()

        # Score books by theme relevance
        scored_books = []
        query_lower = query.lower()

        for book in candidate_books:
            score = 0.0

            # Theme matching
            for theme in book.persistent_themes:
                if theme.lower() in query_lower:
                    score += 2.0

            # Title/description matching
            if query_lower in book.title.lower():
                score += 1.5
            if query_lower in book.description.lower():
                score += 1.0

            # Recency boost
            age_days = (datetime.now() - book.last_accessed).days
            recency_score = max(0, 1.0 - age_days / 30.0)  # Decay over 30 days
            score += recency_score

            # Category boost
            category_weights = {
                ShelfCategory.ACTIVE: 2.0,
                ShelfCategory.RECENT: 1.5,
                ShelfCategory.REFERENCE: 1.8,
                ShelfCategory.ARCHIVED: 0.5
            }
            score *= category_weights.get(book.shelf_category, 1.0)

            if score > 0:
                scored_books.append((score, book))

        # Sort by score and return top results
        scored_books.sort(key=lambda x: x[0], reverse=True)
        return [book for score, book in scored_books[:max_books]]

    async def get_statistics(self) -> dict[str, Any]:
        """Get temporal organization statistics"""
        if not self.temporal_library:
            return {}

        return await self.temporal_library.get_statistics()
