"""
Temporal Library

Handles persistent storage of temporal books and library shelves.
Provides the storage backend for the temporal organization system.
"""

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..types import LibraryShelf, ShelfCategory, TemporalBook, TemporalScale

logger = logging.getLogger(__name__)


class TemporalLibrary:
    """
    Persistent storage for temporal books and library shelves.

    Organizes storage by:
    - Books: Individual temporal containers
    - Shelves: Categorical organization
    - Indexes: Fast lookup by various criteria
    """

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Separate directories for different data types
        self.books_dir = self.storage_path / "books"
        self.shelves_dir = self.storage_path / "shelves"
        self.indexes_dir = self.storage_path / "indexes"

        for dir_path in [self.books_dir, self.shelves_dir, self.indexes_dir]:
            dir_path.mkdir(exist_ok=True)

        # In-memory caches for performance
        self._book_cache: dict[str, TemporalBook] = {}
        self._shelf_cache: dict[str, LibraryShelf] = {}

        # Indexes for fast lookups
        self._session_index: dict[str, list[str]] = {}  # session_id -> book_ids
        self._scale_index: dict[TemporalScale, list[str]] = {}  # scale -> book_ids
        self._category_index: dict[ShelfCategory, list[str]] = {}  # category -> book_ids

        # Load indexes on startup
        asyncio.create_task(self._load_indexes())

    async def store_book(self, book: TemporalBook) -> bool:
        """Store a temporal book"""
        try:
            book_path = self.books_dir / f"{book.book_id}.json"
            book_data = self._serialize_book(book)

            with open(book_path, 'w') as f:
                json.dump(book_data, f, indent=2)

            # Update cache
            self._book_cache[book.book_id] = book

            # Update indexes
            await self._update_book_indexes(book)

            return True

        except Exception as e:
            print(f"Error storing book {book.book_id}: {e}")
            return False

    async def get_book(self, book_id: str) -> TemporalBook | None:
        """Get a temporal book by ID"""
        # Check cache first
        if book_id in self._book_cache:
            return self._book_cache[book_id]

        book_path = self.books_dir / f"{book_id}.json"
        if not book_path.exists():
            return None

        try:
            with open(book_path) as f:
                book_data = json.load(f)

            book = self._deserialize_book(book_data)

            # Cache for future access
            self._book_cache[book_id] = book

            return book

        except Exception as e:
            print(f"Error loading book {book_id}: {e}")
            return None

    async def initialize(self) -> None:
        """Initialize the temporal library"""
        try:
            # Create directories if they don't exist
            for dir_path in [self.books_dir, self.shelves_dir, self.indexes_dir]:
                dir_path.mkdir(exist_ok=True)

            # Load existing indexes
            await self._load_indexes()
            logger.info("Temporal library initialized")

        except Exception as e:
            logger.error(f"Failed to initialize temporal library: {e}")
            raise

    async def _load_indexes(self) -> None:
        """Load indexes from disk"""
        try:
            # Load session index
            session_index_path = self.indexes_dir / "session_index.json"
            if session_index_path.exists():
                with open(session_index_path) as f:
                    self._session_index = json.load(f)

            # Load scale index
            scale_index_path = self.indexes_dir / "scale_index.json"
            if scale_index_path.exists():
                with open(scale_index_path) as f:
                    scale_data = json.load(f)
                    # Convert string keys back to enum
                    self._scale_index = {
                        TemporalScale(k): v for k, v in scale_data.items()
                    }

            # Load category index
            category_index_path = self.indexes_dir / "category_index.json"
            if category_index_path.exists():
                with open(category_index_path) as f:
                    category_data = json.load(f)
                    # Convert string keys back to enum
                    self._category_index = {
                        ShelfCategory(k): v for k, v in category_data.items()
                    }

        except Exception as e:
            logger.error(f"Error loading indexes: {e}")
            # Initialize empty indexes on error
            self._session_index = {}
            self._scale_index = {}
            self._category_index = {}

    async def find_books_by_scale(self, scale: TemporalScale) -> list[TemporalBook]:
        """Find all books at a specific temporal scale"""
        book_ids = self._scale_index.get(scale, [])
        books = []

        for book_id in book_ids:
            book = await self.get_book(book_id)
            if book:
                books.append(book)

        return books

    async def find_books_older_than(self, cutoff_date: datetime) -> list[TemporalBook]:
        """Find all books older than the cutoff date"""
        all_books = []

        # Iterate through all book files
        for book_file in self.books_dir.glob("*.json"):
            book = await self.get_book(book_file.stem)
            if book and book.last_accessed < cutoff_date:
                all_books.append(book)

        return all_books

    async def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics about the temporal library"""
        try:
            total_books = len(list(self.books_dir.glob("*.json")))
            total_shelves = len(list(self.shelves_dir.glob("*.json")))

            # Calculate storage size
            storage_size = sum(
                f.stat().st_size for f in self.storage_path.rglob("*") if f.is_file()
            )

            # Get books by category
            category_counts = {}
            for category in ShelfCategory:
                category_counts[category.value] = len(self._category_index.get(category, []))

            # Get books by scale
            scale_counts = {}
            for scale in TemporalScale:
                scale_counts[scale.value] = len(self._scale_index.get(scale, []))

            return {
                "total_books": total_books,
                "total_shelves": total_shelves,
                "storage_size_bytes": storage_size,
                "books_by_category": category_counts,
                "books_by_scale": scale_counts,
                "active_sessions": len(self._session_index)
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            # Save indexes before shutdown
            await self._save_indexes()
            logger.info("Temporal library cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def store_shelf(self, shelf: LibraryShelf) -> bool:
        """Store a library shelf"""
        try:
            shelf_path = self.shelves_dir / f"{shelf.shelf_id}.json"
            shelf_data = self._serialize_shelf(shelf)

            with open(shelf_path, 'w') as f:
                json.dump(shelf_data, f, indent=2)

            # Update cache
            self._shelf_cache[shelf.shelf_id] = shelf

            # Update indexes
            await self._update_shelf_indexes(shelf)

            return True

        except Exception as e:
            print(f"Error storing shelf {shelf.shelf_id}: {e}")
            return False

    async def get_shelf(self, shelf_id: str) -> LibraryShelf | None:
        """Get a library shelf by ID"""
        # Check cache first
        if shelf_id in self._shelf_cache:
            return self._shelf_cache[shelf_id]

        shelf_path = self.shelves_dir / f"{shelf_id}.json"
        if not shelf_path.exists():
            return None

        try:
            with open(shelf_path) as f:
                shelf_data = json.load(f)

            shelf = self._deserialize_shelf(shelf_data)

            # Cache for future access
            self._shelf_cache[shelf_id] = shelf

            return shelf

        except Exception as e:
            print(f"Error loading shelf {shelf_id}: {e}")
            return None

    async def find_books_by_session(self, session_id: str) -> List[TemporalBook]:
        """Find all books associated with a session"""
        book_ids = self._session_index.get(session_id, [])
        books = []

        for book_id in book_ids:
            book = await self.get_book(book_id)
            if book:
                books.append(book)

        return books

    async def find_books_by_category(self, category: ShelfCategory) -> List[TemporalBook]:
        """Find all books in a shelf category"""
        book_ids = self._category_index.get(category, [])
        books = []

        for book_id in book_ids:
            book = await self.get_book(book_id)
            if book:
                books.append(book)

        return books

    async def find_books_by_theme(self, theme: str) -> list[TemporalBook]:
        """Find books containing a specific theme"""
        matching_books = []
        theme_lower = theme.lower()

        for book_file in self.books_dir.glob("*.json"):
            try:
                with open(book_file) as f:
                    book_data = json.load(f)

                # Check if theme appears in persistent themes
                persistent_themes = [t.lower() for t in book_data.get("persistent_themes", [])]
                if theme_lower in persistent_themes:
                    book = self._deserialize_book(book_data)
                    matching_books.append(book)

            except Exception as e:
                print(f"Error checking book themes {book_file}: {e}")
                continue

        return matching_books

    async def get_all_books(self, limit: int | None = None) -> list[TemporalBook]:
        """Get all books, optionally limited"""
        books = []
        book_files = list(self.books_dir.glob("*.json"))

        if limit:
            book_files = book_files[:limit]

        for book_file in book_files:
            try:
                with open(book_file) as f:
                    book_data = json.load(f)

                book = self._deserialize_book(book_data)
                books.append(book)

            except Exception as e:
                print(f"Error loading book {book_file}: {e}")
                continue

        # Sort by last accessed (most recent first)
        books.sort(key=lambda b: b.last_accessed, reverse=True)

        return books

    async def get_all_shelves(self) -> list[LibraryShelf]:
        """Get all library shelves"""
        shelves = []

        for shelf_file in self.shelves_dir.glob("*.json"):
            try:
                with open(shelf_file) as f:
                    shelf_data = json.load(f)

                shelf = self._deserialize_shelf(shelf_data)
                shelves.append(shelf)

            except Exception as e:
                print(f"Error loading shelf {shelf_file}: {e}")
                continue

        # Sort by category and name
        shelves.sort(key=lambda s: (s.category.value, s.name))

        return shelves

    async def delete_book(self, book_id: str) -> bool:
        """Delete a book and update indexes"""
        try:
            book_path = self.books_dir / f"{book_id}.json"

            # Get book for index cleanup
            book = await self.get_book(book_id)

            # Remove file
            if book_path.exists():
                book_path.unlink()

            # Remove from cache
            self._book_cache.pop(book_id, None)

            # Update indexes
            if book:
                await self._remove_book_from_indexes(book)

            return True

        except Exception as e:
            print(f"Error deleting book {book_id}: {e}")
            return False

    async def delete_shelf(self, shelf_id: str) -> bool:
        """Delete a shelf"""
        try:
            shelf_path = self.shelves_dir / f"{shelf_id}.json"

            # Remove file
            if shelf_path.exists():
                shelf_path.unlink()

            # Remove from cache
            self._shelf_cache.pop(shelf_id, None)

            return True

        except Exception as e:
            print(f"Error deleting shelf {shelf_id}: {e}")
            return False

    async def _update_book_indexes(self, book: TemporalBook):
        """Update indexes when a book is added or modified"""
        book_id = book.book_id

        # Session index
        for session_id in book.session_ids:
            if session_id not in self._session_index:
                self._session_index[session_id] = []
            if book_id not in self._session_index[session_id]:
                self._session_index[session_id].append(book_id)

        # Scale index
        if book.temporal_scale not in self._scale_index:
            self._scale_index[book.temporal_scale] = []
        if book_id not in self._scale_index[book.temporal_scale]:
            self._scale_index[book.temporal_scale].append(book_id)

        # Category index
        if book.shelf_category not in self._category_index:
            self._category_index[book.shelf_category] = []
        if book_id not in self._category_index[book.shelf_category]:
            self._category_index[book.shelf_category].append(book_id)

        # Save indexes
        await self._save_indexes()

    async def _update_shelf_indexes(self, shelf: LibraryShelf):
        """Update indexes when a shelf is added or modified"""
        # Currently no specific shelf indexes needed
        pass

    async def _remove_book_from_indexes(self, book: TemporalBook):
        """Remove book from all indexes"""
        book_id = book.book_id

        # Session index
        for session_id in book.session_ids:
            if session_id in self._session_index and book_id in self._session_index[session_id]:
                self._session_index[session_id].remove(book_id)
                if not self._session_index[session_id]:
                    del self._session_index[session_id]

        # Scale index
        if book.temporal_scale in self._scale_index and book_id in self._scale_index[book.temporal_scale]:
            self._scale_index[book.temporal_scale].remove(book_id)
            if not self._scale_index[book.temporal_scale]:
                del self._scale_index[book.temporal_scale]

        # Category index
        if book.shelf_category in self._category_index and book_id in self._category_index[book.shelf_category]:
            self._category_index[book.shelf_category].remove(book_id)
            if not self._category_index[book.shelf_category]:
                del self._category_index[book.shelf_category]

        # Save indexes
        await self._save_indexes()

    async def _save_indexes(self):
        """Save indexes to disk"""
        try:
            session_index_path = self.indexes_dir / "session_index.json"
            with open(session_index_path, 'w') as f:
                json.dump(self._session_index, f, indent=2)
        except Exception as e:
            print(f"Error saving session index: {e}")

        try:
            scale_index_path = self.indexes_dir / "scale_index.json"
            # Convert enum keys to strings for JSON
            scale_data = {k.value: v for k, v in self._scale_index.items()}
            with open(scale_index_path, 'w') as f:
                json.dump(scale_data, f, indent=2)
        except Exception as e:
            print(f"Error saving scale index: {e}")

        try:
            category_index_path = self.indexes_dir / "category_index.json"
            # Convert enum keys to strings for JSON
            category_data = {k.value: v for k, v in self._category_index.items()}
            with open(category_index_path, 'w') as f:
                json.dump(category_data, f, indent=2)
        except Exception as e:
            print(f"Error saving category index: {e}")

    async def close(self):
        """Clean shutdown"""
        await self._save_indexes()
        self._book_cache.clear()
        self._shelf_cache.clear()

    def _serialize_book(self, book: TemporalBook) -> dict[str, Any]:
        """Serialize a TemporalBook to JSON-compatible dict"""
        book_dict = asdict(book)
        # Convert datetime objects to ISO format
        book_dict['created'] = book.created.isoformat()
        book_dict['last_accessed'] = book.last_accessed.isoformat()
        book_dict['start_time'] = book.start_time.isoformat()
        if book.end_time:
            book_dict['end_time'] = book.end_time.isoformat()
        # Convert enums to string values
        book_dict['temporal_scale'] = book.temporal_scale.value
        book_dict['shelf_category'] = book.shelf_category.value
        return book_dict

    def _deserialize_book(self, book_dict: dict[str, Any]) -> TemporalBook:
        """Deserialize a dict back to TemporalBook"""
        # Convert ISO format back to datetime
        book_dict['created'] = datetime.fromisoformat(book_dict['created'])
        book_dict['last_accessed'] = datetime.fromisoformat(book_dict['last_accessed'])
        book_dict['start_time'] = datetime.fromisoformat(book_dict['start_time'])
        if book_dict.get('end_time'):
            book_dict['end_time'] = datetime.fromisoformat(book_dict['end_time'])
        # Convert string values back to enums
        book_dict['temporal_scale'] = TemporalScale(book_dict['temporal_scale'])
        book_dict['shelf_category'] = ShelfCategory(book_dict['shelf_category'])
        return TemporalBook(**book_dict)

    def _serialize_shelf(self, shelf: LibraryShelf) -> dict[str, Any]:
        """Serialize a LibraryShelf to JSON-compatible dict"""
        shelf_dict = asdict(shelf)
        # Convert datetime objects to ISO format
        shelf_dict['created'] = shelf.created.isoformat()
        shelf_dict['last_accessed'] = shelf.last_accessed.isoformat()
        shelf_dict['time_span_start'] = shelf.time_span_start.isoformat()
        if shelf.time_span_end:
            shelf_dict['time_span_end'] = shelf.time_span_end.isoformat()
        # Convert enums to string values
        shelf_dict['category'] = shelf.category.value
        shelf_dict['primary_scale'] = shelf.primary_scale.value
        return shelf_dict

    def _deserialize_shelf(self, shelf_dict: dict[str, Any]) -> LibraryShelf:
        """Deserialize a dict back to LibraryShelf"""
        # Convert ISO format back to datetime
        shelf_dict['created'] = datetime.fromisoformat(shelf_dict['created'])
        shelf_dict['last_accessed'] = datetime.fromisoformat(shelf_dict['last_accessed'])
        shelf_dict['time_span_start'] = datetime.fromisoformat(shelf_dict['time_span_start'])
        if shelf_dict.get('time_span_end'):
            shelf_dict['time_span_end'] = datetime.fromisoformat(shelf_dict['time_span_end'])
        # Convert string values back to enums
        shelf_dict['category'] = ShelfCategory(shelf_dict['category'])
        shelf_dict['primary_scale'] = TemporalScale(shelf_dict['primary_scale'])
        return LibraryShelf(**shelf_dict)
