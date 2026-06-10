import numpy as np
from typing import Dict, List


class VectorIndex:
    """
    Dynamic vector index (Section A).

    Rules:
    - Dot-product similarity on L2-normalized vectors.
    - insert: succeeds iff ID does not exist; duplicate IDs in one batch must not occur in data.
    - delete: succeeds iff ID exists; non-existing IDs must not crash.
    - search: return shape (num_queries, min(k, n_active)); IDs sorted by descending dot product.
    - Each of insert/delete/search must be at most 20 physical lines (autograder-enforced).
    """

    def __init__(self, dim: int):
        """Initializes the vector index with pre-allocated tracking arrays."""
        self.dim = int(dim)
        self._cap = 8192  # Initial internal capacity buffer
        self._n = 0  # Track the actual number of active vectors currently stored

        # Pre-allocate memory blocks to avoid frequent overhead during smaller inserts
        self._mat = np.empty((self._cap, self.dim), dtype=np.float32)
        self._ids = np.empty(self._cap, dtype=np.int64)

        # Hash map for O(1) lookups translating global ID -> internal array index row
        self._id2pos: Dict[int, int] = {}

    def insert(self, batch: Dict[int, np.ndarray]) -> Dict[str, List[int]]:
        """
        Inserts a batch of unique vectors. If the internal pre-allocated memory
        buffers fill up, they will dynamically double in capacity.
        """

        failed = []
        new_ids = []
        new_vecs = []

        # Separate incoming duplicates from legitimate new insertions
        for v, vec in batch.items():
            v = int(v)
            if v in self._id2pos:
                failed.append(v)
            else:
                new_ids.append(v)
                new_vecs.append(vec)

        # If the batch contained entirely duplicates, return immediately
        if not new_ids:
            return {"succeeded": [], "failed": failed}

        num_new = len(new_ids)
        need = self._n + num_new

        # Ensure capacity (amortized doubling strategy)
        if need > self._cap:
            self._cap = max(need * 2, self._cap * 2)

            # Reallocate and copy the matrix data
            new_mat = np.empty((self._cap, self.dim), dtype=np.float32)
            new_mat[:self._n] = self._mat[:self._n]
            self._mat = new_mat

            # Reallocate and copy the tracking IDs array
            new_ids_arr = np.empty(self._cap, dtype=np.int64)
            new_ids_arr[:self._n] = self._ids[:self._n]
            self._ids = new_ids_arr

        # Write new elements into the continuous tail-end slice of our arrays
        self._mat[self._n: need] = np.array(new_vecs, dtype=np.float32, copy=False)
        self._ids[self._n: need] = new_ids

        # Map the new global IDs to their physical matrix row locations
        self._id2pos.update(zip(new_ids, range(self._n, need)))
        self._n = need

        return {"succeeded": new_ids, "failed": failed}

    def delete(self, ids: np.ndarray) -> Dict[str, List[int]]:
        """
        Deletes vector IDs from the index. To avoid costly array resizing or shifting,
        this utilizes a hole-filling strategy: moving valid items from the very end
        of the matrix into the indices left vacant by deletions.
        """
        succeeded, failed, valid_deletes = [], [], set()
        id_list = np.asarray(ids, dtype=np.int64).tolist()
        id2pos = self._id2pos

        # First, determine which IDs are valid for deletion and which are not
        for vid in id_list:
            if vid in id2pos and vid not in valid_deletes:
                valid_deletes.add(vid)
                succeeded.append(vid)
            else:
                failed.append(vid)

        # If there are no valid deletes, we can return early without modifying the index
        if not valid_deletes:
            return {"succeeded": succeeded, "failed": failed}

        # Get the internal positions of the valid deletes and compute the new size after deletion
        delete_positions = np.array([id2pos.pop(vid) for vid in valid_deletes], dtype=np.int64)
        new_n = self._n - len(valid_deletes)

        # Fill the holes left by deletions with valid items from the end of the active range
        holes = delete_positions[delete_positions < new_n]

        # If there are holes to fill, move items from the tail end of the active range into the holes
        if holes.size > 0:
            # Build a mask tracking elements in the "tail zone" that are NOT scheduled for deletion
            tail_mask = np.ones(self._n - new_n, dtype=bool)
            tail_mask[delete_positions[delete_positions >= new_n] - new_n] = False

            # Grab row positions of valid elements sitting in the tail zone
            fillers = np.arange(new_n, self._n, dtype=np.int64)[tail_mask]

            # Swap data: Copy rows from the fillers directly into the vacant holes
            self._mat[holes] = self._mat[fillers]
            self._ids[holes] = self._ids[fillers]

            # Re-map the moved tail elements to point to their brand new row positions
            id2pos.update(zip(self._ids[holes].tolist(), holes.tolist()))

        self._n = new_n
        return {"succeeded": succeeded, "failed": failed}

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """
        Performs high-speed batch similarity search against active vectors.
        Uses highly-optimized BLAS matrix multiplication for dot-products,
        combined with a partial quicksort strategy via argpartition.
        """
        if self._n == 0:
            return np.empty((queries.shape[0], 0), dtype=np.int64)

        q = np.asarray(queries, dtype=np.float32)
        k_eff = min(int(k), self._n)

        # Compute the full similarity scores matrix between queries and active vectors
        scores = q @ self._mat[:self._n].T

        # If we want everything, sort all elements completely
        if k_eff == self._n:
            return self._ids[:self._n][np.argsort(scores, axis=1)[:, ::-1]]
        
        # Otherwise, use argpartition to find the top-k candidates efficiently, then sort those candidates
        part = np.argpartition(scores, -k_eff, axis=1)[:, -k_eff:]
        part_scores = scores[np.arange(q.shape[0])[:, None], part]
        sorted_local = np.argsort(part_scores, axis=1)[:, ::-1]
        
        # Map the locally sorted top-k candidate indices back to the original IDs
        return self._ids[part[np.arange(q.shape[0])[:, None], sorted_local]]
