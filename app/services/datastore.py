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

    def add_vectors(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> None:
        """Append vectors to the datastore."""
        num_new = vectors.shape[0]
        if self.vectors is None:
            self.initialize(vectors.shape[1], max(1000, num_new))

        # Check if we need to resize (simplified for now)
        if self.vectors is None or self.ids is None:
            return

        current_capacity = self.vectors.shape[0]
        if self.count + num_new > current_capacity:
            # In a production scenario, we'd handle mmap resizing here
            pass

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


datastore = DataStore()
