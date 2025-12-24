"""PostgreSQL + pgvector storage backend for embeddings."""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager

import numpy as np
import numpy.typing as npt
from pgvector.psycopg2 import register_vector
from psycopg2 import pool
from psycopg2.extensions import connection as Connection
from tenacity import retry, stop_after_attempt, wait_exponential

from coral_vision.config import DatabaseConfig
from coral_vision.core.circuit_breaker import circuit_breaker
from coral_vision.core.exceptions import DatabaseError
from coral_vision.core.logger import get_logger

logger = get_logger("storage")


class PgVectorStorageBackend:
    """PostgreSQL + pgvector storage backend for face embeddings."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "coral_vision",
        user: str = "coral",
        password: str = "coral",
        min_connections: int = 1,
        max_connections: int = 20,
    ):
        """Initialize pgvector storage backend.

        Args:
            host: PostgreSQL host.
            port: PostgreSQL port.
            database: Database name.
            user: Database user.
            password: Database password.
            min_connections: Minimum number of connections in pool.
            max_connections: Maximum number of connections in pool.
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_connections = min_connections
        self.max_connections = max_connections
        self._pool: pool.ThreadedConnectionPool | None = None
        self._pool_lock = threading.Lock()

    def _get_pool(self) -> pool.ThreadedConnectionPool:
        """Get or create connection pool.

        Returns:
            Threaded connection pool.
        """
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    try:
                        # Get SSL configuration from environment
                        ssl_mode = os.getenv("DB_SSLMODE", "prefer")
                        connect_kwargs = {
                            "host": self.host,
                            "port": self.port,
                            "database": self.database,
                            "user": self.user,
                            "password": self.password,
                        }

                        # Add SSL configuration if provided
                        if ssl_mode != "disable":
                            connect_kwargs["sslmode"] = ssl_mode

                            ssl_cert = os.getenv("DB_SSL_CERT")
                            ssl_key = os.getenv("DB_SSL_KEY")
                            ssl_root_cert = os.getenv("DB_SSL_ROOT_CERT")

                            if ssl_cert:
                                connect_kwargs["sslcert"] = ssl_cert
                            if ssl_key:
                                connect_kwargs["sslkey"] = ssl_key
                            if ssl_root_cert:
                                connect_kwargs["sslrootcert"] = ssl_root_cert

                        self._pool = pool.ThreadedConnectionPool(
                            minconn=self.min_connections,
                            maxconn=self.max_connections,
                            **connect_kwargs,
                        )
                        logger.info(
                            f"Created connection pool: {self.min_connections}-{self.max_connections} connections"
                        )
                    except Exception as e:
                        logger.error(f"Failed to create connection pool: {e}")
                        raise DatabaseError(
                            f"Failed to create connection pool: {e}"
                        ) from e
        return self._pool

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    @circuit_breaker(
        failure_threshold=5,
        recovery_timeout=60.0,
        expected_exception=(DatabaseError, Exception),
    )
    def _get_connection(self) -> Connection:
        """Get a connection from the pool.

        Returns:
            Active database connection.

        Raises:
            DatabaseError: If connection cannot be obtained.
        """
        try:
            pool_instance = self._get_pool()
            conn = pool_instance.getconn()
            if conn is None:
                raise DatabaseError("Failed to get connection from pool")
            # Register pgvector extension on first use
            try:
                register_vector(conn)
            except Exception:
                # Already registered or not needed
                pass
            return conn
        except Exception as e:
            logger.error(f"Error getting connection: {e}")
            if isinstance(e, DatabaseError):
                raise
            raise DatabaseError(f"Failed to get database connection: {e}") from e

    def _put_connection(self, conn: Connection) -> None:
        """Return a connection to the pool.

        Args:
            conn: Connection to return.
        """
        if self._pool and conn:
            try:
                self._pool.putconn(conn)
            except Exception as e:
                logger.warning(f"Error returning connection to pool: {e}")

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions.

        Yields:
            Database connection.

        Raises:
            DatabaseError: If transaction fails.
        """
        conn = None
        try:
            conn = self._get_connection()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            logger.error(f"Transaction failed: {e}")
            raise DatabaseError(f"Transaction failed: {e}") from e
        finally:
            if conn:
                self._put_connection(conn)

    def ensure_initialized(self) -> None:
        """Create necessary database tables and extensions."""
        with self._transaction() as conn:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # Create people table
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS people (
                        person_id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Create embeddings table with vector column
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id SERIAL PRIMARY KEY,
                        person_id VARCHAR(255) NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
                        embedding vector(192) NOT NULL,
                        source_image VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Create index for fast similarity search using HNSW with optimized parameters
                # m=16: number of bi-directional links per node (higher = more accurate, slower)
                # ef_construction=64: size of candidate list during construction (higher = better quality, slower)
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS embeddings_vector_idx
                    ON embeddings USING hnsw (embedding vector_l2_ops)
                    WITH (m = 16, ef_construction = 64)
                """
                )

                # Create index on person_id for fast lookups
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS embeddings_person_id_idx
                    ON embeddings(person_id)
                """
                )
        logger.info("Database schema initialized")

    def load_people_index(self) -> dict[str, str]:
        """Load all people from database.

        Returns:
            Dictionary mapping person_id to name.
        """
        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT person_id, name FROM people")
                return {row[0]: row[1] for row in cur.fetchall()}

    def upsert_person(self, person_id: str, name: str) -> None:
        """Add or update a person in the database.

        Args:
            person_id: Unique identifier for the person.
            name: Display name for the person.
        """
        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO people (person_id, name, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (person_id)
                    DO UPDATE SET name = EXCLUDED.name, updated_at = CURRENT_TIMESTAMP
                    """,
                    (person_id, name),
                )

    def get_person_name(self, person_id: str) -> str | None:
        """Get person name by ID.

        Args:
            person_id: Unique identifier for the person.

        Returns:
            Person name or None if not found.
        """
        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT name FROM people WHERE person_id = %s", (person_id,)
                )
                result = cur.fetchone()
                return result[0] if result else None

    def add_embedding(
        self,
        person_id: str,
        embedding: npt.NDArray[np.float32],
        source_image: str | None = None,
    ) -> int:
        """Store an embedding in the database.

        Args:
            person_id: Unique identifier for the person.
            embedding: Face embedding vector (192-D).
            source_image: Optional source image filename.

        Returns:
            ID of the stored embedding.
        """
        # Convert numpy array to list for pgvector
        embedding_list = embedding.tolist()

        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO embeddings (person_id, embedding, source_image)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (person_id, embedding_list, source_image),
                )
                embedding_id = cur.fetchone()[0]
        return embedding_id

    def get_embeddings(self, person_id: str) -> list[npt.NDArray[np.float32]]:
        """Get all embeddings for a person.

        Args:
            person_id: Unique identifier for the person.

        Returns:
            List of embedding vectors.
        """
        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT embedding FROM embeddings WHERE person_id = %s ORDER BY id",
                    (person_id,),
                )
                return [np.array(row[0], dtype=np.float32) for row in cur.fetchall()]

    def get_all_embeddings(self) -> dict[str, list[npt.NDArray[np.float32]]]:
        """Get all embeddings for all people.

        Returns:
            Dictionary mapping person_id to list of embeddings.
        """
        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT person_id, embedding FROM embeddings ORDER BY person_id, id"
                )
                result: dict[str, list[npt.NDArray[np.float32]]] = {}
                for person_id, embedding in cur.fetchall():
                    if person_id not in result:
                        result[person_id] = []
                    result[person_id].append(np.array(embedding, dtype=np.float32))
                return result

    def delete_person(self, person_id: str) -> None:
        """Delete a person and all their embeddings (CASCADE).

        Args:
            person_id: Unique identifier for the person.
        """
        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM people WHERE person_id = %s", (person_id,))

    def get_embedding_count(self, person_id: str) -> int:
        """Get the number of embeddings for a person.

        Args:
            person_id: Unique identifier for the person.

        Returns:
            Number of stored embeddings.
        """
        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE person_id = %s", (person_id,)
                )
                return cur.fetchone()[0]

    def find_similar_embeddings(
        self,
        embedding: npt.NDArray[np.float32],
        limit: int = 10,
        threshold: float = 1.0,
    ) -> list[tuple[str, float]]:
        """Find most similar embeddings using vector similarity search.

        Args:
            embedding: Query embedding vector.
            limit: Maximum number of results to return.
            threshold: Maximum L2 distance threshold.

        Returns:
            List of (person_id, distance) tuples, sorted by distance.
        """
        embedding_list = embedding.tolist()

        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT person_id, embedding <-> %s AS distance
                    FROM embeddings
                    WHERE embedding <-> %s < %s
                    ORDER BY distance
                    LIMIT %s
                    """,
                    (embedding_list, embedding_list, threshold, limit),
                )
                return [(row[0], float(row[1])) for row in cur.fetchall()]

    def close(self) -> None:
        """Close all connections in the pool."""
        if self._pool:
            try:
                self._pool.closeall()
                logger.info("Connection pool closed")
            except Exception as e:
                logger.warning(f"Error closing connection pool: {e}")
            finally:
                self._pool = None

    def __del__(self) -> None:
        """Cleanup connection pool on deletion."""
        self.close()


def get_storage_backend_from_env() -> PgVectorStorageBackend:
    """Create and initialize pgvector storage backend from environment variables.

    Reads database configuration from environment variables:
    - DB_HOST (default: "localhost")
    - DB_PORT (default: "5432")
    - DB_NAME (default: "coral_vision")
    - DB_USER (default: "coral")
    - DB_PASSWORD (default: "coral")

    Returns:
        Initialized pgvector storage backend with database schema created.
    """
    db_config = DatabaseConfig(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "coral_vision"),
        user=os.getenv("DB_USER", "coral"),
        password=os.getenv("DB_PASSWORD", "coral"),
    )
    backend = PgVectorStorageBackend(
        host=db_config.host,
        port=db_config.port,
        database=db_config.database,
        user=db_config.user,
        password=db_config.password,
    )
    backend.ensure_initialized()
    return backend
