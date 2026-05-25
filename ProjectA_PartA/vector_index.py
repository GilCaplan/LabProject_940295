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
        self.dim = int(dim)
        self._cap = 8192
        self._n = 0
        self._mat = np.empty((self._cap, self.dim), dtype=np.float32)
        self._ids = np.empty(self._cap, dtype=np.int64)
        self._id2pos: Dict[int, int] = {}

    def insert(self, batch: Dict[int, np.ndarray]) -> Dict[str, List[int]]:
        failed = []
        new_ids = []
        new_vecs = []
        for v, vec in batch.items():
            v = int(v)
            if v in self._id2pos:
                failed.append(v)
            else:
                new_ids.append(v)
                new_vecs.append(vec)
        if not new_ids:
            return {"succeeded": [], "failed": failed}
        num_new = len(new_ids)
        need = self._n + num_new

        if need > self._cap:
            self._cap = max(need * 2, self._cap * 2)
            new_mat = np.empty((self._cap, self.dim), dtype=np.float32)
            new_mat[:self._n] = self._mat[:self._n]
            self._mat = new_mat
            new_ids_arr = np.empty(self._cap, dtype=np.int64)
            new_ids_arr[:self._n] = self._ids[:self._n]
            self._ids = new_ids_arr
        self._mat[self._n : need] = np.array(new_vecs, dtype=np.float32, copy=False)
        self._ids[self._n : need] = new_ids
        self._id2pos.update(zip(new_ids, range(self._n, need)))
        self._n = need
        return {"succeeded": new_ids, "failed": failed}

    def delete(self, ids: np.ndarray) -> Dict[str, List[int]]:
        succeeded, failed, valid_deletes = [], [], set()
        id_list = np.asarray(ids, dtype=np.int64).tolist()
        id2pos = self._id2pos
        for vid in id_list:
            if vid in id2pos and vid not in valid_deletes:
                valid_deletes.add(vid)
                succeeded.append(vid)
            else:
                failed.append(vid)
        if not valid_deletes:
            return {"succeeded": succeeded, "failed": failed}
        delete_positions = np.array([id2pos.pop(vid) for vid in valid_deletes], dtype=np.int64)
        new_n = self._n - len(valid_deletes)
        holes = delete_positions[delete_positions < new_n]
        if holes.size > 0:
            tail_mask = np.ones(self._n - new_n, dtype=bool)
            tail_mask[delete_positions[delete_positions >= new_n] - new_n] = False
            fillers = np.arange(new_n, self._n, dtype=np.int64)[tail_mask]
            self._mat[holes] = self._mat[fillers]
            self._ids[holes] = self._ids[fillers]
            id2pos.update(zip(self._ids[holes].tolist(), holes.tolist()))
        self._n = new_n
        return {"succeeded": succeeded, "failed": failed}

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        if self._n == 0: return np.empty((queries.shape[0], 0), dtype=np.int64)
        q = np.asarray(queries, dtype=np.float32)
        k_eff = min(int(k), self._n)
        scores = q @ self._mat[:self._n].T
        if k_eff == self._n: return self._ids[:self._n][np.argsort(scores, axis=1)[:, ::-1]]
        part = np.argpartition(scores, -k_eff, axis=1)[:, -k_eff:]
        part_scores = scores[np.arange(q.shape[0])[:, None], part]
        sorted_local = np.argsort(part_scores, axis=1)[:, ::-1]
        return self._ids[part[np.arange(q.shape[0])[:, None], sorted_local]]