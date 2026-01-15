import os

import numpy as np

from app.core.config import settings


class DataStore:
    """
    Manages the persistent storage of vectors using memory-mapped files (mmap).
    Inspired by high-performance data storage patterns for large-scale retrieval.
    """

    def __init__(self, directory: str = settings.DATA_DIR) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.vectors: np.ndarray | None = None
        self.ids: np.ndarray | None = None
        self.dimension: int = 0
        self.count: int = 0

    def initialize(self, dimension: int, initial_capacity: int = 1000) -> None:
        """Initialize the datastore files."""
        self.dimension = dimension
        vector_path = os.path.join(self.directory, "vectors.npy")
        ids_path = os.path.join(self.directory, "ids.npy")

        # Create empty files with initial capacity
        self.vectors = np.memmap(
            vector_path, dtype="float32", mode="w+", shape=(initial_capacity, dimension)
        )
        self.ids = np.memmap(
            ids_path, dtype="int64", mode="w+", shape=(initial_capacity,)
        )
        self.count = 0

    def load(self, dimension: int) -> None:
        """Load existing datastore files if they exist."""
        self.dimension = dimension
        vector_path = os.path.join(self.directory, "vectors.npy")
        ids_path = os.path.join(self.directory, "ids.npy")

        if os.path.exists(vector_path) and os.path.exists(ids_path):
            # We don't know the capacity easily without checking file size,
            # but we can try to infer it. For simplicity, we'll use 'r+'
            self.vectors = np.memmap(vector_path, dtype="float32", mode="r+")
            # Reshape based on dimension
            num_elements = self.vectors.size
            capacity = num_elements // dimension
            self.vectors = self.vectors.reshape((capacity, dimension))

            self.ids = np.memmap(ids_path, dtype="int64", mode="r+")
            # We need a way to track 'count'. For now, let's assume it's stored or infer from IDs.
            # In a real system, we'd store metadata (count, dimension, etc.) in a separate JSON/Header.
            # For now, let's assume 'initialize' or 'add_vectors' sets it.
            self.count = capacity  # Placeholder: assumes full for now or needs metadata

    def _resize(self, new_capacity: int) -> None:
        """Resize the memory-mapped files to a new capacity."""
        if self.vectors is None or self.ids is None:
            return

        # 1. Flush and clear current mappings
        if isinstance(self.vectors, np.memmap):
            self.vectors.flush()
        if isinstance(self.ids, np.memmap):
            self.ids.flush()

        # Save current state
        current_vectors = np.array(self.vectors[: self.count])
        current_ids = np.array(self.ids[: self.count])

        # Close by deleting
        del self.vectors
        del self.ids

        # 2. Re-initialize with new capacity
        self.initialize(self.dimension, new_capacity)

        # 3. Restore data
        if self.vectors is not None and self.ids is not None:
            self.vectors[: self.count] = current_vectors
            self.ids[: self.count] = current_ids

    def add_vectors(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> None:
        """Append vectors to the datastore."""
        num_new = vectors.shape[0]
        if self.vectors is None:
            self.initialize(vectors.shape[1], max(1000, num_new))

        if self.vectors is None or self.ids is None:
            return

        current_capacity = self.vectors.shape[0]
        if self.count + num_new > current_capacity:
            # Grow by 2x or enough to fit new vectors
            new_capacity = max(current_capacity * 2, self.count + num_new)
            self._resize(new_capacity)

        # Now we definitely have space
        if self.vectors is not None and self.ids is not None:
            self.vectors[self.count : self.count + num_new] = vectors.astype("float32")
            if ids is not None:
                self.ids[self.count : self.count + num_new] = ids
            else:
                self.ids[self.count : self.count + num_new] = np.arange(
                    self.count, self.count + num_new
                )

            self.count += num_new
            if isinstance(self.vectors, np.memmap):
                self.vectors.flush()

    def get_vectors(self) -> np.ndarray:
        """Returns the active slice of vectors."""
        if self.vectors is None:
            return np.array([], dtype="float32")
        return self.vectors[: self.count]

    def reset(self) -> None:
        """Clear the datastore by deleting the underlying files."""
        if isinstance(self.vectors, np.memmap):
            del self.vectors
        if isinstance(self.ids, np.memmap):
            del self.ids

        vector_path = os.path.join(self.directory, "vectors.npy")
        ids_path = os.path.join(self.directory, "ids.npy")

        if os.path.exists(vector_path):
            os.remove(vector_path)
        if os.path.exists(ids_path):
            os.remove(ids_path)

        self.vectors = None
        self.ids = None
        self.count = 0
        self.dimension = 0


datastore = DataStore()
