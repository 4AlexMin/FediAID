from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel

from src.io import load_jsonl_dataset

def _normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True) + eps
    return vec / norm


# ============================================================
# Community Memory Bank (reference space)
# ============================================================

class CommunityMemory:
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.ids: List[str] = data["cids"].tolist()
        self.vecs: np.ndarray = _normalize(data["cents"].astype(np.float32))

    # def get_topk(self, query_vec: np.ndarray, k: int, *args, **kwargs) -> np.ndarray:
    #     q = _normalize(query_vec.reshape(1, -1))[0]
    #     sims = q @ self.vecs.T
    #     k = min(k, sims.shape[0])
    #     idx = np.argpartition(-sims, k - 1)[:k]
    #     idx = idx[np.argsort(-sims[idx])]
        
    #     # # Flipped
    #     # idx = np.argpartition(sims, k - 1)[:k]
    #     # idx = idx[np.argsort(sims[idx])]
        
    #     # ====== verbose ======
    #     top_indices = np.argsort(-sims)[:30]
    #     print("================= Top-k Retrieval ================")
    #     print("\t Top-k retrieval debug info:")
    #     print('\t chosen indices:', top_indices)
    #     print('\t communities:', [self.ids[i] for i in top_indices])
    #     print("\t similarity scores:", sims[top_indices])
    #     print("\t -----------------------------------------------")
    #     print("\t tail of similarity scores:")
    #     tail_indices = np.argsort(sims)[:30]
    #     print("\t tail indices:", tail_indices)
    #     print("\t tail communities:", [self.ids[i] for i in tail_indices])
    #     print("\t tail similarity scores:", sims[tail_indices])
    #     print("==================================================")
    #     # ====================
    #     return self.vecs[idx]


    def mmr_retrieval(
        self,
        query_vec: np.ndarray,
        k: int,
        lambda_mmr: float = 0.45,
        *args, **kwargs
    ) -> np.ndarray:
        """
        Maximal Marginal Relevance (MMR) retrieval.

        Args:
            query_vec: [D]
            k: number of communities to retrieve
            lambda_mmr: trade-off between relevance and diversity
        """
        q = _normalize(query_vec.reshape(1, -1))[0]
        sims_to_query = q @ self.vecs.T  # [N]

        selected = []
        selected_idx = []

        # candidate pool
        candidates = list(range(self.vecs.shape[0]))

        for _ in range(min(k, len(candidates))):
            if not selected:
                # first pick: most relevant
                i = int(np.argmax(sims_to_query))
            else:
                mmr_scores = []
                for i in candidates:
                    sim_q = sims_to_query[i]
                    sim_sel = max(
                        self.vecs[i] @ self.vecs[j] for j in selected_idx
                    )
                    mmr = lambda_mmr * sim_q - (1 - lambda_mmr) * sim_sel
                    mmr_scores.append(mmr)

                i = candidates[int(np.argmax(mmr_scores))]

            selected_idx.append(i)
            selected.append(self.vecs[i])
            candidates.remove(i)

        return np.stack(selected)

    def retrieve_communities(self, query_vec: np.ndarray, k: int, *args, **kwargs) -> np.ndarray:
        return self.mmr_retrieval(query_vec, k, *args, **kwargs)


# ============================================================
# Dataset
# ============================================================

class AIGTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        memory: CommunityMemory,
        tokenizer_name: Optional[str] = None,
        model_name: Optional[str] = None,
        k: int = 3,
        max_length: int = 256,
        device: Optional[torch.device] = None,
        max_centroid_samples: int = -1,
        lambda_mmr: float = 0.45,
    ):
        self.rows = load_jsonl_dataset(jsonl_path)
        self.memory = memory
        self.k = k
        self.device = device
        self.max_length = max_length
        self.max_centroid_samples = max_centroid_samples
        self.lambda_mmr = lambda_mmr

        self.has_community = "community_id" in self.rows[0]
        self.use_precomputed = "post_emb" in self.rows[0]
        if self.use_precomputed:
            print("Using precomputed post embeddings.")
        else:
            print("Using on-the-fly text encoding for post embeddings.")

        if not self.use_precomputed:
            assert tokenizer_name and model_name
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.encoder = AutoModel.from_pretrained(model_name).to(device)
            self.encoder.eval()

        # Build dataset centroid bank
        self.dataset_centroids = self._build_dataset_centroids()

        # Build retrieval cache (community → retrieved centroids)
        if self.k:
            self.communities_retrieved = self._build_retrieval_cache(self.k)

    # --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.rows)

    # --------------------------------------------------------

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.k is None:
            raise ValueError("k must be set to retrieve entries.")
        
        row = self.rows[idx]

        vec = (
            np.asarray(row["post_emb"], dtype=np.float32)
            if self.use_precomputed
            else self._embed_text(row["text"])
        )
        vec = _normalize(vec)

        if self.k == 0:
            # Ablation/baseline path: no retrieval, so return an empty centroid set
            centroids = np.zeros((0, vec.shape[0]), dtype=np.float32)
        else:
            cid = row.get("community_id") if self.has_community else None
            centroids = self.communities_retrieved[cid]

        return (
            torch.tensor(vec, dtype=torch.float32),
            torch.tensor(centroids, dtype=torch.float32) if centroids is not None else None,
            torch.tensor(int(row["label"]), dtype=torch.long),
        )

    # ========================================================
    # Embedding
    # ========================================================

    def _embed_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            outputs = self.encoder(**tokens)
            hidden = outputs.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
            return pooled.squeeze(0).cpu().numpy()

    # ========================================================
    # Dataset centroid bank (QUERY SPACE)
    # ========================================================
    def _build_dataset_centroids(self) -> Dict[Optional[str], np.ndarray]:
        if not self.has_community:
            return {None: self._global_centroid()}

        groups: Dict[str, List[np.ndarray]] = {}

        for r in self.rows:
            cid = r["community_id"]

            if self.use_precomputed:
                vec = np.asarray(r["post_emb"], dtype=np.float32)
            else:
                vec = self._embed_text(r["text"])

            groups.setdefault(cid, []).append(vec)

        centroids = {}
        for cid, vecs in groups.items():
            if self.max_centroid_samples > 0:
                vecs = vecs[: self.max_centroid_samples]
            arr = np.stack(vecs)
            centroids[cid] = _normalize(arr.mean(axis=0))

        print(f"Built dataset centroid bank for {len(centroids)} communities.")
        return centroids
    
    def _global_centroid(self) -> np.ndarray:
        vecs = []

        for r in self.rows:
            if self.use_precomputed:
                vecs.append(np.asarray(r["post_emb"], dtype=np.float32))
            else:
                vecs.append(self._embed_text(r["text"]))

        if not vecs:
            raise ValueError("No data available to build global centroid.")

        return _normalize(np.stack(vecs).mean(axis=0))


    # ========================================================
    # Retrieval cache (RETRIEVAL SPACE)
    # ========================================================
    def _build_retrieval_cache(self, k) -> Dict[Optional[str], np.ndarray]:
        cache = {}
        for cid, query_vec in self.dataset_centroids.items():
            cache[cid] = self.memory.retrieve_communities(
                query_vec,
                k,
                lambda_mmr=self.lambda_mmr,
            )
        return cache
    
    def set_k(self, k: int) -> None:
        self.k = k
        if self.k:
            self.communities_retrieved = self._build_retrieval_cache(self.k)
        else:
            self.communities_retrieved = {}
    


