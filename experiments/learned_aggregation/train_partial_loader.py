
"""Resumable document-batch loader for learned-aggregation train partials.

Important protocol rules
------------------------
1. CPC labels are never loaded by this module.
2. The full-train depth normalization divisor is fitted on train only:
       normalized_depth = depth / 55.0
   The same fixed divisor must be used for DEV and TEST inference.
3. Existing DEV-only checkpoints retain their recovered divisor of 27.
4. Document-order RNG and augmentation RNG are separate streams.
5. Epoch plans are materialized before training and can be resumed using
   ``start_document``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F


TRAIN_DEPTH_DIVISOR = 55.0
EXPECTED_SHARDS = 496
EXPECTED_DOCUMENTS = 49_599
EXPECTED_CLAIMS = 844_636
EXPECTED_DIRECTED_EDGES = 1_446_960


def derive_seed(
    base_seed: int,
    *,
    stream: int,
    epoch: int = 0,
    index: int = 0,
    view: int = 0,
) -> int:
    """Derive an independent, deterministic uint32 seed."""
    ss = np.random.SeedSequence(
        [
            int(base_seed),
            int(stream),
            int(epoch),
            int(index),
            int(view),
        ]
    )
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def augmentation_seed(
    training_seed: int,
    *,
    epoch: int,
    batch_index: int,
    view: int,
) -> int:
    """Seed for stochastic training views, separate from document order."""
    return derive_seed(
        training_seed,
        stream=20_001,
        epoch=epoch,
        index=batch_index,
        view=view,
    )


@dataclass(frozen=True)
class EpochPlan:
    training_seed: int
    epoch: int
    shard_indices: np.ndarray
    document_indices: np.ndarray
    patent_ids: np.ndarray
    order_sha256: str
    npz_path: Path
    metadata_path: Path

    @property
    def n_documents(self) -> int:
        return int(self.document_indices.shape[0])


@dataclass
class GraphBatch:
    x: torch.Tensor
    depths: torch.Tensor
    structural_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor
    node_ptr: torch.Tensor
    patent_ids: List[str]
    training_seed: int
    epoch: int
    batch_index: int
    start_document: int
    stop_document: int
    augmentation_seed_view1: int
    augmentation_seed_view2: int

    @property
    def n_documents(self) -> int:
        return len(self.patent_ids)

    @property
    def n_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to(self, device: torch.device | str) -> "GraphBatch":
        self.x = self.x.to(device, non_blocking=True)
        self.depths = self.depths.to(device, non_blocking=True)
        self.structural_features = self.structural_features.to(
            device, non_blocking=True
        )
        self.edge_index = self.edge_index.to(device, non_blocking=True)
        self.edge_type = self.edge_type.to(device, non_blocking=True)
        self.edge_attr = self.edge_attr.to(device, non_blocking=True)
        self.batch = self.batch.to(device, non_blocking=True)
        self.node_ptr = self.node_ptr.to(device, non_blocking=True)
        return self


class TrainPartialDataset:
    """Loader for shard-based patent claim graphs."""

    def __init__(
        self,
        partial_dir: str | Path,
        plan_dir: str | Path,
        *,
        depth_divisor: float = TRAIN_DEPTH_DIVISOR,
        expected_shards: int = EXPECTED_SHARDS,
    ) -> None:
        self.partial_dir = Path(partial_dir)
        self.plan_dir = Path(plan_dir)
        self.depth_divisor = float(depth_divisor)

        if self.depth_divisor <= 0:
            raise ValueError("depth_divisor must be positive")

        self.shard_paths = sorted(
            self.partial_dir.glob("shard_*_graphs.npz")
        )
        if len(self.shard_paths) != expected_shards:
            raise RuntimeError(
                f"Expected {expected_shards} shards, "
                f"found {len(self.shard_paths)}"
            )

        self.plan_dir.mkdir(parents=True, exist_ok=True)

        self.documents_per_shard = np.empty(
            len(self.shard_paths), dtype=np.int32
        )
        self.claims_per_shard = np.empty(
            len(self.shard_paths), dtype=np.int64
        )
        self.edges_per_shard = np.empty(
            len(self.shard_paths), dtype=np.int64
        )

        for shard_index, path in enumerate(self.shard_paths):
            with np.load(path, allow_pickle=False) as z:
                node_ptr = z["node_ptr"]
                edge_ptr = z["edge_ptr"]
                self.documents_per_shard[shard_index] = len(node_ptr) - 1
                self.claims_per_shard[shard_index] = int(node_ptr[-1])
                self.edges_per_shard[shard_index] = int(edge_ptr[-1])

        self.n_documents = int(self.documents_per_shard.sum())
        self.n_claims = int(self.claims_per_shard.sum())
        self.n_edges = int(self.edges_per_shard.sum())

        if self.n_documents != EXPECTED_DOCUMENTS:
            raise RuntimeError(
                f"Document mismatch: {self.n_documents} "
                f"!= {EXPECTED_DOCUMENTS}"
            )
        if self.n_claims != EXPECTED_CLAIMS:
            raise RuntimeError(
                f"Claim mismatch: {self.n_claims} != {EXPECTED_CLAIMS}"
            )
        if self.n_edges != EXPECTED_DIRECTED_EDGES:
            raise RuntimeError(
                f"Edge mismatch: {self.n_edges} "
                f"!= {EXPECTED_DIRECTED_EDGES}"
            )

    def _plan_paths(
        self,
        training_seed: int,
        epoch: int,
    ) -> Tuple[Path, Path]:
        stem = f"train_seed_{training_seed:03d}_epoch_{epoch:03d}"
        return (
            self.plan_dir / f"{stem}.npz",
            self.plan_dir / f"{stem}.json",
        )

    def create_epoch_plan(
        self,
        training_seed: int,
        epoch: int,
        *,
        overwrite: bool = False,
    ) -> EpochPlan:
        """Create an efficient deterministic order.

        Shards are shuffled, and documents are independently shuffled inside
        each shard. Documents from a shard remain contiguous so each large
        NPZ file is opened approximately once per epoch.
        """
        npz_path, metadata_path = self._plan_paths(training_seed, epoch)

        if npz_path.exists() and metadata_path.exists() and not overwrite:
            return self.load_epoch_plan(training_seed, epoch)

        shard_rng = np.random.default_rng(
            derive_seed(
                training_seed,
                stream=10_001,
                epoch=epoch,
            )
        )
        shard_order = shard_rng.permutation(len(self.shard_paths))

        shard_parts: List[np.ndarray] = []
        document_parts: List[np.ndarray] = []
        patent_parts: List[np.ndarray] = []

        for shard_index in shard_order:
            shard_index = int(shard_index)
            n_docs = int(self.documents_per_shard[shard_index])

            doc_rng = np.random.default_rng(
                derive_seed(
                    training_seed,
                    stream=10_002,
                    epoch=epoch,
                    index=shard_index,
                )
            )
            local_order = doc_rng.permutation(n_docs).astype(
                np.int32, copy=False
            )

            with np.load(
                self.shard_paths[shard_index],
                allow_pickle=False,
            ) as z:
                # Intentionally read only patent IDs, never CPC labels.
                patent_ids = np.asarray(z["patent_ids"])[local_order]

            shard_parts.append(
                np.full(n_docs, shard_index, dtype=np.int16)
            )
            document_parts.append(local_order)
            patent_parts.append(patent_ids.astype(str, copy=False))

        shard_indices = np.concatenate(shard_parts)
        document_indices = np.concatenate(document_parts)
        patent_ids = np.concatenate(patent_parts)

        if len(document_indices) != self.n_documents:
            raise RuntimeError("Incomplete epoch plan")

        hash_input = "\n".join(patent_ids.tolist()).encode("utf-8")
        order_sha256 = hashlib.sha256(hash_input).hexdigest()

        np.savez_compressed(
            npz_path,
            shard_indices=shard_indices,
            document_indices=document_indices,
            patent_ids=patent_ids,
        )

        metadata = {
            "schema_version": "train_document_order_v1",
            "training_seed": int(training_seed),
            "epoch": int(epoch),
            "n_documents": int(len(document_indices)),
            "order_sha256": order_sha256,
            "ordering_policy": (
                "shuffle_shards_then_shuffle_documents_within_each_shard"
            ),
            "document_order_stream": 10001,
            "within_shard_order_stream": 10002,
            "augmentation_stream": 20001,
            "depth_divisor": self.depth_divisor,
            "plan_file": str(npz_path),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        return EpochPlan(
            training_seed=int(training_seed),
            epoch=int(epoch),
            shard_indices=shard_indices,
            document_indices=document_indices,
            patent_ids=patent_ids,
            order_sha256=order_sha256,
            npz_path=npz_path,
            metadata_path=metadata_path,
        )

    def load_epoch_plan(
        self,
        training_seed: int,
        epoch: int,
    ) -> EpochPlan:
        npz_path, metadata_path = self._plan_paths(training_seed, epoch)

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as z:
            shard_indices = z["shard_indices"].copy()
            document_indices = z["document_indices"].copy()
            patent_ids = z["patent_ids"].copy()

        hash_input = "\n".join(
            patent_ids.astype(str).tolist()
        ).encode("utf-8")
        actual_hash = hashlib.sha256(hash_input).hexdigest()

        if actual_hash != metadata["order_sha256"]:
            raise RuntimeError(
                f"Epoch-plan hash mismatch: {npz_path}"
            )

        return EpochPlan(
            training_seed=int(training_seed),
            epoch=int(epoch),
            shard_indices=shard_indices,
            document_indices=document_indices,
            patent_ids=patent_ids,
            order_sha256=actual_hash,
            npz_path=npz_path,
            metadata_path=metadata_path,
        )

    @staticmethod
    def _localize_edges(
        src: np.ndarray,
        dst: np.ndarray,
        node_start: int,
        node_stop: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_nodes = node_stop - node_start

        if src.size == 0:
            return (
                src.astype(np.int64, copy=False),
                dst.astype(np.int64, copy=False),
            )

        shard_global = (
            int(src.min()) >= node_start
            and int(dst.min()) >= node_start
            and int(src.max()) < node_stop
            and int(dst.max()) < node_stop
        )
        document_local = (
            int(src.min()) >= 0
            and int(dst.min()) >= 0
            and int(src.max()) < n_nodes
            and int(dst.max()) < n_nodes
        )

        if shard_global:
            src = src.astype(np.int64, copy=False) - node_start
            dst = dst.astype(np.int64, copy=False) - node_start
        elif document_local:
            src = src.astype(np.int64, copy=False)
            dst = dst.astype(np.int64, copy=False)
        else:
            raise RuntimeError(
                "Edge indices are neither document-local nor "
                f"valid shard-global indices: nodes=[{node_start}, "
                f"{node_stop})"
            )

        return src, dst

    def iter_epoch(
        self,
        training_seed: int,
        epoch: int,
        *,
        batch_size: int = 256,
        start_document: int = 0,
        reverse_edge_features: bool = False,
        pin_memory: bool = False,
    ) -> Iterator[GraphBatch]:
        """Yield graph batches without loading label arrays.

        ``start_document`` should normally be a previously recorded batch's
        ``stop_document`` value.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        plan = self.create_epoch_plan(training_seed, epoch)

        if not 0 <= start_document <= plan.n_documents:
            raise ValueError(
                f"Invalid start_document={start_document}"
            )

        current_shard_index: Optional[int] = None
        current_npz = None

        docs_x: List[np.ndarray] = []
        docs_depth: List[np.ndarray] = []
        docs_src: List[np.ndarray] = []
        docs_dst: List[np.ndarray] = []
        docs_edge_type: List[np.ndarray] = []
        docs_patent_id: List[str] = []
        batch_start = start_document
        emitted_batch_index = start_document // batch_size

        def build_batch(
            start: int,
            stop: int,
            batch_index: int,
        ) -> GraphBatch:
            node_counts = np.asarray(
                [len(x) for x in docs_x],
                dtype=np.int64,
            )
            node_ptr = np.concatenate(
                [
                    np.zeros(1, dtype=np.int64),
                    np.cumsum(node_counts, dtype=np.int64),
                ]
            )

            x = np.concatenate(docs_x, axis=0).astype(
                np.float32, copy=False
            )
            depths = np.concatenate(docs_depth).astype(
                np.int64, copy=False
            )

            root_indicator = (depths == 0).astype(np.float32)
            normalized_depth = (
                depths.astype(np.float32) / self.depth_divisor
            )
            structural = np.stack(
                [normalized_depth, root_indicator],
                axis=1,
            )

            batch_vector = np.repeat(
                np.arange(len(docs_x), dtype=np.int64),
                node_counts,
            )

            shifted_src: List[np.ndarray] = []
            shifted_dst: List[np.ndarray] = []
            for doc_i, (src, dst) in enumerate(
                zip(docs_src, docs_dst)
            ):
                offset = int(node_ptr[doc_i])
                shifted_src.append(src + offset)
                shifted_dst.append(dst + offset)

            if shifted_src:
                all_src = np.concatenate(shifted_src)
                all_dst = np.concatenate(shifted_dst)
                edge_type_np = np.concatenate(docs_edge_type).astype(
                    np.int64, copy=False
                )
            else:
                all_src = np.empty(0, dtype=np.int64)
                all_dst = np.empty(0, dtype=np.int64)
                edge_type_np = np.empty(0, dtype=np.int64)

            edge_index_np = np.stack([all_src, all_dst], axis=0)

            # Preserve canonical edge_type and separately construct the
            # recovered model's edge feature order.
            edge_feature_type = (
                1 - edge_type_np
                if reverse_edge_features
                else edge_type_np
            )

            batch_obj = GraphBatch(
                x=torch.from_numpy(x),
                depths=torch.from_numpy(depths),
                structural_features=torch.from_numpy(structural),
                edge_index=torch.from_numpy(edge_index_np),
                edge_type=torch.from_numpy(edge_type_np),
                edge_attr=F.one_hot(
                    torch.from_numpy(edge_feature_type),
                    num_classes=2,
                ).to(torch.float32),
                batch=torch.from_numpy(batch_vector),
                node_ptr=torch.from_numpy(node_ptr),
                patent_ids=list(docs_patent_id),
                training_seed=int(training_seed),
                epoch=int(epoch),
                batch_index=int(batch_index),
                start_document=int(start),
                stop_document=int(stop),
                augmentation_seed_view1=augmentation_seed(
                    training_seed,
                    epoch=epoch,
                    batch_index=batch_index,
                    view=1,
                ),
                augmentation_seed_view2=augmentation_seed(
                    training_seed,
                    epoch=epoch,
                    batch_index=batch_index,
                    view=2,
                ),
            )

            if pin_memory:
                batch_obj.x = batch_obj.x.pin_memory()
                batch_obj.depths = batch_obj.depths.pin_memory()
                batch_obj.structural_features = (
                    batch_obj.structural_features.pin_memory()
                )
                batch_obj.edge_index = batch_obj.edge_index.pin_memory()
                batch_obj.edge_type = batch_obj.edge_type.pin_memory()
                batch_obj.edge_attr = batch_obj.edge_attr.pin_memory()
                batch_obj.batch = batch_obj.batch.pin_memory()
                batch_obj.node_ptr = batch_obj.node_ptr.pin_memory()

            return batch_obj

        try:
            for position in range(start_document, plan.n_documents):
                shard_index = int(plan.shard_indices[position])
                document_index = int(plan.document_indices[position])

                if shard_index != current_shard_index:
                    if current_npz is not None:
                        current_npz.close()
                    current_npz = np.load(
                        self.shard_paths[shard_index],
                        allow_pickle=False,
                    )
                    current_shard_index = shard_index

                node_ptr = current_npz["node_ptr"]
                edge_ptr = current_npz["edge_ptr"]

                node_start = int(node_ptr[document_index])
                node_stop = int(node_ptr[document_index + 1])
                edge_start = int(edge_ptr[document_index])
                edge_stop = int(edge_ptr[document_index + 1])

                x_doc = np.asarray(
                    current_npz["embeddings"][node_start:node_stop],
                    dtype=np.float32,
                )
                depth_doc = np.asarray(
                    current_npz["depths"][node_start:node_stop],
                    dtype=np.int64,
                )

                src = np.asarray(
                    current_npz["edge_src"][edge_start:edge_stop]
                )
                dst = np.asarray(
                    current_npz["edge_dst"][edge_start:edge_stop]
                )
                src, dst = self._localize_edges(
                    src,
                    dst,
                    node_start,
                    node_stop,
                )

                edge_type = np.asarray(
                    current_npz["edge_type"][edge_start:edge_stop],
                    dtype=np.int64,
                )

                if np.any((edge_type < 0) | (edge_type > 1)):
                    raise RuntimeError("edge_type must contain only 0/1")

                docs_x.append(x_doc)
                docs_depth.append(depth_doc)
                docs_src.append(src)
                docs_dst.append(dst)
                docs_edge_type.append(edge_type)
                docs_patent_id.append(str(plan.patent_ids[position]))

                reached_batch_size = len(docs_x) == batch_size
                reached_epoch_end = position + 1 == plan.n_documents

                if reached_batch_size or reached_epoch_end:
                    stop = position + 1
                    yield build_batch(
                        batch_start,
                        stop,
                        emitted_batch_index,
                    )

                    docs_x.clear()
                    docs_depth.clear()
                    docs_src.clear()
                    docs_dst.clear()
                    docs_edge_type.clear()
                    docs_patent_id.clear()

                    batch_start = stop
                    emitted_batch_index += 1
        finally:
            if current_npz is not None:
                current_npz.close()


def make_torch_generator(
    seed: int,
    device: str = "cpu",
) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator
