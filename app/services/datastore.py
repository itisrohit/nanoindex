import json
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
        self.norms: np.ndarray | None = None
        self.ids: np.ndarray | None = None
        self.dimension: int = 0
        self.count: int = 0
        self.meta_path = os.path.join(self.directory, "meta.json")

        # Try to auto-load if meta exists
        if os.path.exists(self.meta_path):
            self.load()

    def _save_meta(self) -> None:
        """Save metadata to disk."""
        with open(self.meta_path, "w") as f:
            json.dump({"count": self.count, "dimension": self.dimension}, f)

    def initialize(self, dimension: int, initial_capacity: int = 1000) -> None:
        """Initialize the datastore files."""
        self.dimension = dimension
        vector_path = os.path.join(self.directory, "vectors.npy")
        norms_path = os.path.join(self.directory, "norms.npy")
        ids_path = os.path.join(self.directory, "ids.npy")

        # Create empty files with initial capacity
        self.vectors = np.memmap(
            vector_path, dtype="float32", mode="w+", shape=(initial_capacity, dimension)
        )
        self.norms = np.memmap(
            norms_path, dtype="float32", mode="w+", shape=(initial_capacity,)
        )
        self.ids = np.memmap(
            ids_path, dtype="int64", mode="w+", shape=(initial_capacity,)
        )
        self.count = 0
        self._save_meta()

    def load(self) -> None:
        """Load existing datastore files using metadata."""
        if not os.path.exists(self.meta_path):
            return

        with open(self.meta_path) as f:
            meta = json.load(f)
            self.count = meta["count"]
            self.dimension = meta["dimension"]

        vector_path = os.path.join(self.directory, "vectors.npy")
        norms_path = os.path.join(self.directory, "norms.npy")
        ids_path = os.path.join(self.directory, "ids.npy")

        if os.path.exists(vector_path) and os.path.exists(ids_path):
            # Map existing files in read+ mode
            self.vectors = np.memmap(vector_path, dtype="float32", mode="r+")
            # Infer capacity from file size
            num_elements = self.vectors.size
            capacity = num_elements // self.dimension
            self.vectors = self.vectors.reshape((capacity, self.dimension))

            if os.path.exists(norms_path):
                self.norms = np.memmap(norms_path, dtype="float32", mode="r+")
            else:
                # Build norms if missing
                self.norms = np.memmap(
                    norms_path, dtype="float32", mode="w+", shape=(capacity,)
                )
                if self.count > 0:
                    self.norms[: self.count] = np.sum(
                        self.vectors[: self.count] ** 2, axis=1
                    )

            self.ids = np.memmap(ids_path, dtype="int64", mode="r+")

    def _resize(self, new_capacity: int) -> None:
        """Resize the memory-mapped files to a new capacity."""
        if self.vectors is None or self.ids is None:
            return

        # 1. Flush and clear current mappings
        if isinstance(self.vectors, np.memmap):
            self.vectors.flush()
        if isinstance(self.norms, np.memmap):
            self.norms.flush()
        if isinstance(self.ids, np.memmap):
            self.ids.flush()

        # Save current state
        current_count = self.count
        current_vectors = np.array(self.vectors[: self.count])
        current_norms = (
            np.array(self.norms[: self.count]) if self.norms is not None else None
        )
        current_ids = np.array(self.ids[: self.count])

        # Close by deleting
        del self.vectors
        del self.norms
        del self.ids

        # 2. Re-initialize with new capacity
        self.initialize(self.dimension, new_capacity)

        # 3. Restore data and count
        if self.vectors is not None and self.ids is not None:
            self.vectors[:current_count] = current_vectors
            if self.norms is not None and current_norms is not None:
                self.norms[:current_count] = current_norms
            self.ids[:current_count] = current_ids
            self.count = current_count  # Restore the original count
            self._save_meta()

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

            # After resize, ensure arrays are valid
            if self.vectors is None or self.ids is None or self.norms is None:
                raise RuntimeError("Failed to resize datastore arrays")

        # Now we definitely have space
        if self.vectors is not None and self.ids is not None and self.norms is not None:
            vectors_f32 = vectors.astype("float32")
            self.vectors[self.count : self.count + num_new] = vectors_f32
            # Cache squared norms
            self.norms[self.count : self.count + num_new] = np.sum(
                vectors_f32**2, axis=1
            )

            if ids is not None:
                self.ids[self.count : self.count + num_new] = ids
            else:
                self.ids[self.count : self.count + num_new] = np.arange(
                    self.count, self.count + num_new
                )

            self.count += num_new
            self._save_meta()
            v = self.vectors
            if isinstance(v, np.memmap):
                v.flush()
            n = self.norms
            if isinstance(n, np.memmap):
                n.flush()

    def get_vectors(self) -> np.ndarray:
        """Returns the active slice of vectors."""
        if self.vectors is None:
            return np.array([], dtype="float32")
        return self.vectors[: self.count]

    def get_norms(self) -> np.ndarray:
        """Returns the active slice of squared norms."""
        if self.norms is None:
            return np.array([], dtype="float32")
        return self.norms[: self.count]

    def reset(self) -> None:
        """Clear the datastore by deleting the underlying files."""
        if isinstance(self.vectors, np.memmap):
            del self.vectors
        if isinstance(self.norms, np.memmap):
            del self.norms
        if isinstance(self.ids, np.memmap):
            del self.ids

        vector_path = os.path.join(self.directory, "vectors.npy")
        norms_path = os.path.join(self.directory, "norms.npy")
        ids_path = os.path.join(self.directory, "ids.npy")

        for p in [vector_path, norms_path, ids_path, self.meta_path]:
            if os.path.exists(p):
                os.remove(p)

        self.vectors = None
        self.norms = None
        self.ids = None
        self.count = 0
        self.dimension = 0


datastore = DataStore()
