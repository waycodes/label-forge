"""
Local filesystem cache backend.

Uses SQLite for metadata and compressed JSON files for blobs.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from labelforge.cache.key import CacheKey
from labelforge.cache.store import CacheEntry, CacheStore
from labelforge.core.json_canonical import canonical_json_dumps, canonical_json_loads


class FilesystemCache(CacheStore):
    """
    Filesystem-based cache with SQLite metadata store.

    Structure:
        cache_dir/
            metadata.db          # SQLite database
            blobs/
                ab/cd/           # Sharded by key prefix
                    {hash}.json.gz
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize filesystem cache.

        Args:
            cache_dir: Root directory for cache storage.
        """
        self.cache_dir = Path(cache_dir)
        self.blobs_dir = self.cache_dir / "blobs"
        self.db_path = self.cache_dir / "metadata.db"

        # Ensure directories exist
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key_hash TEXT PRIMARY KEY,
                    key_json TEXT NOT NULL,
                    value_ref TEXT NOT NULL,
                    output_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    accessed_at TEXT,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER DEFAULT 0,
                    metadata_json TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_stage_name
                ON cache_entries (json_extract(key_json, '$.stage_name'))
            """)
            conn.commit()

    def _get_blob_path(self, key: CacheKey) -> Path:
        """Get blob file path for a key."""
        prefix = key.prefix
        # Two-level sharding
        return self.blobs_dir / prefix[:2] / prefix[2:4] / f"{key.hash}.json.gz"

    def get(self, key: CacheKey) -> CacheEntry | None:
        """Get cached entry by key."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM cache_entries WHERE key_hash = ?",
                (key.hash,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # Update access tracking
            conn.execute(
                """
                UPDATE cache_entries
                SET accessed_at = ?, access_count = access_count + 1
                WHERE key_hash = ?
                """,
                (datetime.utcnow().isoformat(), key.hash),
            )
            conn.commit()

            return self._row_to_entry(row)

    def put(
        self,
        key: CacheKey,
        value: Any,
        output_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        """Store value in cache."""
        # Write blob
        blob_path = self._get_blob_path(key)
        blob_path.parent.mkdir(parents=True, exist_ok=True)

        json_str = canonical_json_dumps(value)
        json_bytes = json_str.encode("utf-8")

        with gzip.open(blob_path, "wt", encoding="utf-8") as f:
            f.write(json_str)

        size_bytes = blob_path.stat().st_size
        created_at = datetime.utcnow()

        # Write metadata
        entry = CacheEntry(
            key=key,
            value_ref=str(blob_path),
            output_hash=output_hash,
            created_at=created_at,
            size_bytes=size_bytes,
            metadata=metadata or {},
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries
                (key_hash, key_json, value_ref, output_hash, created_at,
                 size_bytes, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key.hash,
                    canonical_json_dumps(key.to_dict()),
                    str(blob_path),
                    output_hash,
                    created_at.isoformat(),
                    size_bytes,
                    canonical_json_dumps(entry.metadata),
                ),
            )
            conn.commit()

        return entry

    def exists(self, key: CacheKey) -> bool:
        """Check if key exists in cache."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM cache_entries WHERE key_hash = ?",
                (key.hash,),
            )
            return cursor.fetchone() is not None

    def delete(self, key: CacheKey) -> bool:
        """Delete entry from cache."""
        # Get blob path first
        entry = self.get(key)
        if entry is None:
            return False

        # Delete blob
        blob_path = Path(entry.value_ref)
        if blob_path.exists():
            blob_path.unlink()

        # Delete metadata
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM cache_entries WHERE key_hash = ?",
                (key.hash,),
            )
            conn.commit()

        return True

    def get_value(self, entry: CacheEntry) -> Any:
        """Retrieve cached value from blob."""
        blob_path = Path(entry.value_ref)

        if not blob_path.exists():
            raise FileNotFoundError(f"Cache blob not found: {blob_path}")

        with gzip.open(blob_path, "rt", encoding="utf-8") as f:
            content = f.read()

        return canonical_json_loads(content)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Total entries
            cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
            total_entries = cursor.fetchone()[0]

            # Total size
            cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
            total_size = cursor.fetchone()[0] or 0

            # By stage
            cursor = conn.execute("""
                SELECT json_extract(key_json, '$.stage_name') as stage,
                       COUNT(*) as count,
                       SUM(size_bytes) as size
                FROM cache_entries
                GROUP BY stage
            """)
            by_stage = {row[0]: {"count": row[1], "size": row[2]} for row in cursor}

        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "by_stage": by_stage,
            "cache_dir": str(self.cache_dir),
        }

    def _row_to_entry(self, row: sqlite3.Row) -> CacheEntry:
        """Convert database row to CacheEntry."""
        key_data = canonical_json_loads(row["key_json"])
        metadata = (
            canonical_json_loads(row["metadata_json"])
            if row["metadata_json"]
            else {}
        )

        return CacheEntry(
            key=CacheKey.from_dict(key_data),
            value_ref=row["value_ref"],
            output_hash=row["output_hash"],
            created_at=datetime.fromisoformat(row["created_at"]),
            accessed_at=(
                datetime.fromisoformat(row["accessed_at"])
                if row["accessed_at"]
                else None
            ),
            access_count=row["access_count"],
            size_bytes=row["size_bytes"],
            metadata=metadata,
        )

    def vacuum(self) -> int:
        """
        Remove orphaned blobs and compact database.

        Returns:
            Number of orphaned blobs removed.
        """
        removed = 0

        # Get all blob paths from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value_ref FROM cache_entries")
            valid_paths = {row[0] for row in cursor}

        # Find and remove orphaned blobs
        for blob_path in self.blobs_dir.rglob("*.json.gz"):
            if str(blob_path) not in valid_paths:
                blob_path.unlink()
                removed += 1

        # Vacuum database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")

        return removed

    def clear_stage(self, stage_name: str) -> int:
        """
        Clear all cache entries for a stage.

        Args:
            stage_name: Name of the stage.

        Returns:
            Number of entries cleared.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get entries to delete
            cursor = conn.execute(
                """
                SELECT value_ref FROM cache_entries
                WHERE json_extract(key_json, '$.stage_name') = ?
                """,
                (stage_name,),
            )
            paths = [row[0] for row in cursor]

            # Delete blobs
            for path in paths:
                blob_path = Path(path)
                if blob_path.exists():
                    blob_path.unlink()

            # Delete metadata
            conn.execute(
                """
                DELETE FROM cache_entries
                WHERE json_extract(key_json, '$.stage_name') = ?
                """,
                (stage_name,),
            )
            conn.commit()

        return len(paths)
